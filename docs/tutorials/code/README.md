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
| `mmdit_block.py` | 双流 MMDiT block（joint attention + AdaLN-Zero + per-stream FFN） | [image_generation_systems_tutorial.md](../image_generation_systems_tutorial.md) | <5s |
| `toy_mmdit_t2i_pipeline.py` | End-to-end skeleton（toy text encoder + VAE + MMDiT + Euler scheduler + true CFG） | [image_generation_systems_tutorial.md](../image_generation_systems_tutorial.md) | <10s |
| `lora.py` | `LoRALinear`（B=0 起点 · α/r 与 rsLoRA 缩放 · merge/unmerge）+ `DoRALinear`（幅度-方向分解）+ 6 个 assert | [lora_peft_tutorial.md](../lora_peft_tutorial.md) | <5s |
| `rag_embedding.py` | 双塔 `DualEncoder` + InfoNCE（in-batch / 难负例）+ BM25 + RRF 混合召回 + 6 个 assert | [rag_embedding_retrieval_tutorial.md](../rag_embedding_retrieval_tutorial.md) | <5s |
| `linear_sparse_attention.py` | linear attention 递推/chunkwise 等价 + delta rule（DeltaNet 改写式更新）+ block-sparse top-k（NSA/MoBA select）+ 6 个 assert | [linear_sparse_attention_tutorial.md](../linear_sparse_attention_tutorial.md) | <5s |
| `normalization.py` | LayerNorm/RMSNorm from-scratch vs `torch` + RMSNorm 去 re-centering + BatchNorm train≠eval + Pre/Post-LN 梯度 top-heavy + Kaiming/Xavier 二阶矩 E[y²] + GPT-2 残差 1/√(2N) + 6 个 assert | [normalization_init_tutorial.md](../normalization_init_tutorial.md) | <5s |
| `optimizer_lr_schedule.py` | SGD-momentum/Adam/AdamW from-scratch vs `torch.optim` + **AdamW≠Adam+L2**(解耦≠耦合) + bias correction 方向(首步 3.16× 偏大) + cosine-warmup 调度 + 动量加速病态二次 + 6 个 assert | [optimizer_lr_schedule_tutorial.md](../optimizer_lr_schedule_tutorial.md) | <5s |

## 运行

```bash
cd docs/tutorials/code
python mha.py
python axial_attention.py
python flow_matching.py          # 需要 matplotlib（可选，没装会跳过画图）
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

