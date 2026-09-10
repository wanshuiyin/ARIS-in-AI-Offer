> 希望大家秋招的时候轻松一点 🌱
>
> 📱 全部教程一页收齐：[wanshuiyin.github.io/ARIS-in-AI-Offer](https://wanshuiyin.github.io/ARIS-in-AI-Offer/)

## TL;DR

这是一个 **中文 ML / LLM / 多模态 / 生成式 / Agent 面试 cheat sheet 合集**，目标读者是准备 2026 年 AI 秋招的同学（国内 LLM 实验室 / 大厂 AI 岗位 / 海外 ML internship）。每篇 cheat sheet 包含：

- **公式推导**（从数学第一性原理走完整链路，不是只给结论）
- **从零开始的 PyTorch 代码**（每段都过 cross-model codex review，确保能跑）
- **25 高频面试题**（L1 必会 / L2 进阶 / L3 顶级 lab，每题展开标准答案 + 容易踩的坑）

所有教程通过 [**ARIS — Auto Research in Sleep**](https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep) 的 `/interview-cheatsheet` + `/render-html` workflow **自动生成 + 跨模型审查**。

## HTML 排版哪里都能读

地铁上掏手机、咖啡馆开 iPad、图书馆开笔记本——同一个 HTML 链接打开都能读：

> 🧮 **MathJax** 渲染所有 LaTeX 公式（不是截图 — 可缩放、可复制、可选中）
>
> 💻 **highlight.js** 给 PyTorch 代码上色
>
> 📐 **响应式排版** 自动适配窗口宽度
>
> 📑 **Sticky TOC** 长文里随时跳转章节
>
> 💾 **单文件 HTML**，下载就能离线读，不依赖任何后端

## 这套 workflow 是怎么工作的

```
                  ┌─────────────────────────────┐
                  │ /interview-cheatsheet skill │
                  └──────────────┬──────────────┘
                                 │
                                 ▼
        ┌──────────────────────────────────────────────┐
        │  Step 1: Plan 12-14 sections                 │
        │  Step 2: Draft 600-1500 lines Chinese MD     │
        │  Step 3: Cross-model codex review (10 checks)│
        │    ┌────────────────────────────────┐        │
        │    │ codex gpt-5.5 xhigh fresh thread        │
        │    │ • formula correctness                   │
        │    │ • code runnability                      │
        │    │ • citation accuracy                     │
        │    │ • personal info redaction               │
        │    │ • style discipline                      │
        │    └────────────────────────────────┘        │
        │  Step 4: Fix loop (no hard cap)              │
        │  Step 5: /render-html → academic template    │
        │  Step 6: Render review (13 checks)           │
        │  Step 7: Write .review.json audit trail      │
        └──────────────────────────────────────────────┘
                                 │
                                 ▼
                ┌──────────────────────────────┐
                │  Single-file HTML (~80KB)    │
                │  ━━━━━━━━━━━━━━━━━━━━━━━━━━ │
                │  ✓ MathJax (live)            │
                │  ✓ highlight.js (live)       │
                │  ✓ Sticky TOC                │
                │  ✓ Responsive layout         │
                │  ✓ Offline-readable          │
                └──────────────────────────────┘
```

## ARIS 的 "跨模型对抗审查" 协议

ARIS 的核心方法论：**executor 和 reviewer 必须不同模型家族**（这里是 Claude × GPT-5.5 xhigh × Gemini）—— LLM 不能自己审自己。这套协议直接复用到面试 cheat sheet 生成：

> 💡 **每篇里的每个公式 / 代码 / 引用都过了独立审查**
>
> 完整审计 trail 见每篇旁边的 `.review.json` —— 包含 reviewer thread ID、抓到的真实 bug、应用的修复。

具体案例（来自仓库 audit trail）：

- **VLM 教程**：M-RoPE `mrope_section=[16,24,24]` 单位被纠正（half-dim pair 不是 dim count）；BLIP-2 stage 数 3 → 2；Qwen2-VL vision tower 攻击 DFN-derived 非自训练
- **MoE 教程**：Q20 Mixtral 8×7B 算术从「7×8+2.3≈47B」改成 shared 2.4B + 8×4.8B FFN ≈ 40.8B（路由增长到 46.7B）；§4.4 router gradient 补齐 softmax/renorm 完整 Jacobian；8×H100 80G = 640GB ≠ 671B FP8 修正
- **Agent Foundations 教程**：跑了 9 轮 codex review，抓到 30+ 个真错（ReAct Fever 数字、SWE-bench Verified OpenAI attribution、MCP 2025-11-25 DCR demoted to MAY、A2A v1.0 SCREAMING_SNAKE_CASE、Anthropic Tool Use GA 2024-05-30 等）

## 教程分类

| Category | Topics |
|---|---|
| 🧠 General / 基础 | Attention · KL Divergence in RLHF (k1/k2/k3 · placement gradient bias) |
| 🎯 Post-Training & Reasoning | RLHF / DPO / GRPO / PPO · Reasoning Models (o1 / R1) · LLM On-Policy Distillation (MiniLLM / GKD / Qwen3 / Tinker) |
| 🏛️ LLM Architecture & Systems | MoE (DeepSeek-V3) · Long Context (RoPE / YaRN / MLA) · KV Cache + Speculative Decoding · Quantization (GPTQ / AWQ / FP8 / NVFP4) · Distributed Training (FSDP2 / ZeRO / TP / PP / EP) |
| 🌊 Generative Models — Theory & Tokenizers | Flow Matching · Diffusion Foundations · VAE / VQ-VAE / VQ-GAN / FSQ |
| 🎨 Generation Systems | Image Gen (SD3 / FLUX / ControlNet) · Video Gen (Sora / Hunyuan-Video / Wan) · 3D Gen (NeRF / 3DGS / SDS) · Diffusion Post-Training (DDPO / DPOK / Diffusion-DPO / Flow-GRPO) · Diffusion / Flow Distillation (CM / sCM / LCM / DMD / DMD2 / ADD) |
| 👁️ Multimodal | VLM (CLIP / LLaVA / Qwen-VL) |
| 🤖 Agents | Agent Foundations (ReAct / MCP / A2A) · Agentic RL (AgentTuning / ToolRL / RAGEN / WebRL) · Multi-Agent & Long-Horizon (CAMEL / AutoGen / MoA / MemGPT / LATS) · Self-Evolving Agents (Ctx2Skill / Voyager / Reflexion / STaR) |

> 🌐 **双语版本**：23 篇 cheat sheet 每篇都有中文 + 英文 HTML 两个版本（文件名 `*_tutorial.html` / `*_tutorial_en.html`）。

## 怎么贡献

读 [CONTRIBUTING.md](../CONTRIBUTING.md)（[中文](../CONTRIBUTING_CN.md)）。

一句话总结：用 ARIS 的 `/interview-cheatsheet` 跑出来 → PR 三个文件（MD / HTML / review.json） → 更新两个 README → 等 merge。社区可以一起把这份资料持续维护下去。

## License

MIT — 用、改、传、二开都行。秋招加油 💪
