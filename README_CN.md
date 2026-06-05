<p align="center">
  <img src="assets/aris_logo.svg" alt="ARIS — Auto Research in Sleep" width="640">
</p>

# ARIS-in-AI-Offer (ARIS in 秋招)

> 希望大家秋招的时候轻松一点 🌱
>
> 📖 **English version (default)**: [README.md](README.md)

**📚 按方向直达** —— 23 篇 first-party cheat sheet，7 大方向 + 1 个社区贡献方向：

[🧠 General / 基础](#-general--基础) · [🎯 Post-Training & Reasoning](#-post-training--reasoning) · [🏛️ LLM Architecture & Systems](#-llm-architecture--systems) · [🌊 Generative Models — 理论 & Tokenizers](#-generative-models--理论--tokenizers) · [🎨 Generation Systems（图像 / 视频 / 3D / Diffusion 后训练）](#-generation-systems--图像--视频--3d--diffusion-后训练) · [👁️ Multimodal](#-multimodal) · [🤖 Agents](#-agents) · [🦾 Embodied AI / 具身智能](#-embodied-ai--具身智能)

或浏览完整 [📚 教程清单 ↓](#-教程清单) · 跳到 [🌐 ARIS-Homepage ↓](#-aris-homepage--fact-checked-学术主页生成器)。

[![Stars](https://img.shields.io/github/stars/wanshuiyin/Auto-claude-code-research-in-sleep?style=flat&logo=github&logoColor=white&color=gold&label=ARIS%20Stars)](https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep/stargazers) · [![arXiv](https://img.shields.io/badge/arXiv-2605.03042-b31b1b?style=flat&logo=arxiv)](https://huggingface.co/papers/2605.03042) · [![HF Daily #1](https://img.shields.io/badge/HF%20Daily%20Papers-%231-yellow?style=flat)](https://huggingface.co/papers/2605.03042) · [![PaperWeekly](https://img.shields.io/badge/Featured%20on-PaperWeekly-red?style=flat)](https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep) · [![awesome-agent-skills](https://img.shields.io/badge/Featured%20in-awesome--agent--skills-blue?style=flat&logo=github)](https://github.com/VoltAgent/awesome-agent-skills) · [![Project of the Day](https://img.shields.io/badge/AI%20Digital%20Crew-Project%20of%20the%20Day-orange?style=flat)](https://aidigitalcrew.com)

> 🏆 **建立在已经验证的方法论上** —— [**ARIS 主仓**](https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep) 拿了 ~10k GitHub stars，HuggingFace Daily Papers #1，AI Digital Crew Project of the Day，74+ 个 research skill 跑在 7+ 平台上。这里不是 vaporware preview —— **每一篇 cheat sheet 都是同一个 `/interview-cheatsheet` + `/render-html` workflow 的产出**，跟科研生产里用的是同一套。

**双语**（中文 + English）ML / LLM / 多模态 / diffusion / agent / 生成式面试 cheat sheet 合集，由 **[ARIS — Auto Research in Sleep](https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep)** 的 `/render-html` workflow 自动生成。

每篇都是一份长文 + 公式 + 从零开始的 PyTorch 代码 + 25 高频面试题（L1 必会 · L2 进阶 · L3 顶级 lab）。

> 🔥 **新 feature** · 🌐 **本仓库另一个 feature** —— [**ARIS-Homepage**](#-aris-homepage--fact-checked-学术主页生成器)：用 **CV + 你选的 GitHub repos** 一键生成 fact-checked 学术主页，单文件 HTML 输出。v1.1 新增 `--from-repos owner/repo,...` —— `gh` CLI 抓 stars / releases / READMEs，把 repo 时间线 merge 进主页。[**🔥 Live demo 在 wanshuiyin.github.io →**](https://wanshuiyin.github.io/)

<p align="center">
  <img src="assets/preview_strip.jpg" alt="ARIS-in-AI-Offer 预览 — 基础知识 + 面试题 + 实际代码，截自一篇代表性 cheat sheet" width="100%">
</p>

> 📖 **预览**（上图）：每个 pillar 一张截图，全部来自 [Diffusion Foundations 教程](https://wanshuiyin.github.io/ARIS-in-AI-Offer/tutorials/diffusion_foundations_tutorial.html) —— ① **基础知识点**（公式推导 + 直觉 + TL;DR），② **面试题**（25 高频题分层 L1/L2/L3），③ **实际代码**（可跑的 PyTorch，含 CFG 训练 + DDIM 采样）。这个三柱结构在本仓库每篇 cheat sheet 里都一样。

<p align="center">
  <img src="assets/homepage_preview_strip.jpg" alt="ARIS-Homepage 预览 —— Header & Bio、ARIS Featured section (hero SVG 右浮动)、Publications (topic group + 缩略图)" width="100%">
</p>

> 🌐 **ARIS-Homepage 预览**（上图）：同一套 `/render-html` workflow，把 CV 变成 fact-checked 学术主页。Live demo：[wanshuiyin.github.io](https://wanshuiyin.github.io/)。详情 + 流程图见 [ARIS-Homepage section ↓](#-aris-homepage--fact-checked-学术主页生成器)。

<p align="center">
  <a href="https://wanshuiyin.github.io/ARIS-in-AI-Offer/blogs/continuous_dlm_representation_perspective.html">
    <img src="assets/blog_continuous_dlm_preview_strip.jpg" alt="Continuous DLM 综述 blog 预览 —— 引言 + 训练 pipeline + 推理结果，3 panel 取自长文综述" width="100%" />
  </a>
</p>

> 📝 **长文 blog 预览**（上图）：独立手写的长文技术综述。*Continuous DLM 综述*（2026 上半年 6 篇）—— 作者 **杨若峰 (Ruofeng Yang)**（上海交大），通过跨模型讨论端到端撰写（Claude Opus 4.7 + Codex GPT-5.5 xhigh + Gemini auto-gemini-3）。[**📖 阅读完整 blog ↗**](https://wanshuiyin.github.io/ARIS-in-AI-Offer/blogs/continuous_dlm_representation_perspective.html)。

### 📱 HTML 格式哪里都能读，清清楚楚

地铁上掏手机、咖啡馆开 iPad、图书馆开笔记本——同一个 HTML 链接打开都能读：

- 🧮 **MathJax** 渲染所有 LaTeX 公式（**不是截图**，可缩放、可复制、可选中）
- 💻 **highlight.js** 给 PyTorch 代码高亮上色
- 📐 **响应式排版** 自动适配窗口宽度，不糊不溢出
- 📑 **Sticky TOC** 长文里随时跳转章节
- 💾 **单文件 HTML**，下载就能离线读，不依赖任何后端

---

## 📢 最新动态

- **2026-06-02** — ![NEW](https://img.shields.io/badge/NEW-red?style=flat-square) 📝 **两篇新 blog —— Continuous DLM（表征视角 v2）+ 扩散 × 表征 × 流形** —— Continuous DLM 综述升级为表征视角的扩写版（取代早先的 v1），并新增一篇姊妹篇梳理图像/视频扩散里「借表征 / 用流形」两条脉络（SSL / Consistency / REPA / RAE / JiT / V-JEPA2）。两篇均手写、跨模型审过，且互相引用。线上：[`continuous_dlm_representation_perspective.html`](docs/blogs/continuous_dlm_representation_perspective.html) · [`diffusion_representation_manifold.html`](docs/blogs/diffusion_representation_manifold.html)。见 [Blog 清单 ↓](#-blog-清单)。
- **2026-06-01** — ![NEW](https://img.shields.io/badge/NEW-red?style=flat-square) 📝 **新 blog：NVIDIA Cosmos 3 深度导读（omnimodal world model · MoT 架构）** —— 对 138 页 Cosmos 3 technical report 的中文长文科普（15 节 · 12 图）：把理解与生成缝进一个 Transformer（Mixture-of-Transformers）、训练配方、规模化训练与 serving、三个尺寸。经 5 轮 Codex GPT-5.5 xhigh 跨模型逐段审（数字/机制核对、overclaim 收敛）。自包含单文件 [`docs/blogs/cosmos3_mot_guide.html`](docs/blogs/cosmos3_mot_guide.html) —— 手写导读（ELF blog 格式），不走受审的 `/render-html` 管线。全文图表数字均出自 NVIDIA Cosmos 3 报告（[github.com/nvidia/cosmos](https://github.com/nvidia/cosmos)）；版权归原作者，此处作注明来源的个人科普导读。
- **2026-05-31** — ![QUALITY](https://img.shields.io/badge/QUALITY-2ea44f?style=flat-square) 🔍 **跨模型审计 pass —— 全集抓出并修复真实错误，并接入 CI 强制** —— 用 fresh Codex GPT-5.5 xhigh 把每篇 cheat sheet（中文 + EN）重审一遍，挖出第一轮漏掉的真技术错误并修正：DeepSeek-V3 的 FP8 GEMM 用 **FP32 累加、而非 bf16**；Qwen2-VL M-RoPE base 是 **1e6**；Molmo 视觉塔是 **CLIP、而非 SigLIP**；TensoRF 复杂度 **O(N³)**；一个渲染坏掉的 StreamingLLM callout；外加 10 处 EN 翻译保真修复（语义反转、漏译框架清单）。现在每个 `docs/` 产物都带可追溯的跨模型审，新增 CI 门（[`tools/verify_reviews.py`](tools/verify_reviews.py) `--mode strict --reproduce`，见 [`review-audit.yml`](.github/workflows/review-audit.yml)）会拦下任何 HTML 非"可从源 byte 复现的已审视图"的 PR —— 未审或手改的内容进不来。**今天还一并 merge 了** Focus-dim 阅读模式（用于 `--blog-mode` 页面：dim 非当前 section · 浮动 🎯 按钮 · ↑/↓ 切节），由 dllm 的 blog 工作回流进 academic 模板——对 cheat sheet **休眠**（gate 在 `.aris-blog` 下），blog-mode 下才点亮，为即将上线的 blog 备好。（[5aae952](https://github.com/wanshuiyin/ARIS-in-AI-Offer/commit/5aae952) · [498ecf2](https://github.com/wanshuiyin/ARIS-in-AI-Offer/commit/498ecf2)）
<details>
<summary>📋 更早的更新</summary>

- **2026-05-28** — ![NEW](https://img.shields.io/badge/NEW-red?style=flat-square) 📝 **首篇 blog 上线：*Continuous DLM 综述*（2026 上半年 6 篇）** —— 长篇双语技术综述，作者 **杨若峰 (Ruofeng Yang)**（上海交大），通过跨模型讨论端到端撰写（Claude Opus 4.7 + Codex GPT-5.5 xhigh + Gemini auto-gemini-3）。覆盖：离散 DLM 已有问题、ELF "已知未知" 连续空间核心想法、Flow-Matching 家族 5 paper 同期对比、ELF 训练 pipeline、架构/参数量/shape 最详细一节、推理代码+时间网格+Tab 6/7 全数字、去噪轨迹定性效果、与字节 Cola-DLM 的 Field Landscape 对比、Q&A、引用资源。文件在 [`docs/blogs/continuous_dlm_representation_perspective.html`](docs/blogs/continuous_dlm_representation_perspective.html)（1.7 MB 自包含单文件，无 build）—— 一份手写长文 HTML，尚未纳入受审的 [`/render-html`](skills/render-html/SKILL.md) 流程。**预览图 + Live 链接见 README 顶部**。（[8475a2d](https://github.com/wanshuiyin/ARIS-in-AI-Offer/commit/8475a2d)）
- **2026-05-28** — ![POLISH](https://img.shields.io/badge/POLISH-blue?style=flat-square) ⚡ **`render-html` P0 polish + 23 个 tutorial 全量 regenerate** —— academic 模板加了 7 个交互特性：print degradation 修复（PDF 导出不再丢 `<details>` 内容）、TOC 侧栏 scrollspy（当前 section 跟随滚动高亮）、figure lightbox（原生 `<dialog>` + focus trap + Esc）、长代码自动折叠（`<pre>` ≥30 行包成 `<details class="code-card">`，可用 ` ```python {collapsed}` / `{open}` fence flag 单块 override）、论文引用 popover（新增 `[[key]]` MD 语法 + `--papers <papers.json>` sidecar）、eyebrow 风格清理（marketing 大写 → 正文衬线灰）、`--blog-mode` 基础设施（opt-in `aris-blog` body class）。XSS 加固：`json_for_script()` 把 `<` / `>` / `&` / U+2028 等 escape 成 `\\uXXXX`，paper title 含 `</script>` 不会 break out。23 对双语 tutorial（共 46 个 HTML）全量 regenerate 吃到新模板，源 MD 没动。Codex GPT-5.5 xhigh 4 轮审（设计 ×2 → 代码 ×1 → spot-check ×1）。试试：滚动 [`attention_tutorial.html`](docs/tutorials/attention_tutorial.html) 看左侧 TOC 跟随高亮（[b79c57d](https://github.com/wanshuiyin/ARIS-in-AI-Offer/commit/b79c57d)、[8793f40](https://github.com/wanshuiyin/ARIS-in-AI-Offer/commit/8793f40)）。
- **2026-05-26** — ![NEW](https://img.shields.io/badge/NEW-red?style=flat-square) 🐍 **5 个可跑的 PyTorch 教学脚本** —— [`docs/tutorials/code/`](docs/tutorials/code/) 是仓库第一批 runnable-code 贡献：[`mha.py`](docs/tutorials/code/mha.py)（MHA + causal mask）、[`axial_attention.py`](docs/tutorials/code/axial_attention.py)（H/W axial + 复杂度表）、[`flow_matching.py`](docs/tutorials/code/flow_matching.py)（2D moons 上的 Rectified Flow）、[`mmdit_block.py`](docs/tutorials/code/mmdit_block.py)（双流 MMDiT block）、[`toy_mmdit_t2i_pipeline.py`](docs/tutorials/code/toy_mmdit_t2i_pipeline.py)（端到端 toy T2I pipeline）。纯 PyTorch，CPU 几秒到几十秒跑完，每个脚本自带 `assert` sanity check（shape 对齐、必要时跟 `nn.MultiheadAttention` 数值校准）。配合 [`attention_tutorial.md`](docs/tutorials/attention_tutorial.md) / [`flow_matching_tutorial.md`](docs/tutorials/flow_matching_tutorial.md) / [`image_generation_systems_tutorial.md`](docs/tutorials/image_generation_systems_tutorial.md) 阅读（[f63f468](https://github.com/wanshuiyin/ARIS-in-AI-Offer/commit/f63f468)）。
- **2026-05-24** — ![NEW](https://img.shields.io/badge/NEW-red?style=flat-square) 🐙 **ARIS-Homepage v1.1：`--from-repos`** — 用户选定 `owner/repo` 列表通过 `gh` CLI 抓快照，调用方 LLM agent 把 repo 时间线 merge 进主页 News + `featured_projects[].github`。默认拒绝 private repo。Closes [#2](https://github.com/wanshuiyin/ARIS-in-AI-Offer/issues/2) by [@Yafei-Liu99](https://github.com/Yafei-Liu99)（[cdcf9a2](https://github.com/wanshuiyin/ARIS-in-AI-Offer/commit/cdcf9a2)）。
- **2026-05-23** — ![NEW](https://img.shields.io/badge/NEW-red?style=flat-square) 🌐 **ARIS-Homepage v1 上线** — CV → fact-checked 学术主页（DBLP / arXiv 审 venue/年份/作者，错的硬阻断 ship）。单文件 HTML；Codex / Gemini review 可选。Live demo：[wanshuiyin.github.io](https://wanshuiyin.github.io/)。Skill：[`skills/homepage-generator/SKILL.md`](skills/homepage-generator/SKILL.md)（[b818c1d](https://github.com/wanshuiyin/ARIS-in-AI-Offer/commit/b818c1d)）。
- **2026-05-22** — ![NEW](https://img.shields.io/badge/NEW-red?style=flat-square) 🦾 **社区贡献 feature：具身智能高频面试题库** by [@WinstonJQ](https://github.com/WinstonJQ) —— 413 题 8 卷（VLA / 模仿学习 / RL / 世界模型 / 工程落地 / 腿足控制 / 3D 感知 / 系统设计），外部托管，从 Tutorial Index 新增的 "🦾 Embodied AI" 分类里链接过去（[b1ebb6f](https://github.com/wanshuiyin/ARIS-in-AI-Offer/commit/b1ebb6f)）。
- **2026-05** — ![NEW](https://img.shields.io/badge/NEW-red?style=flat-square) 📚 **4 篇新双语 cheat sheet**：KL Divergence in RLHF（k1/k2/k3 · placement gradient bias）、LLM On-Policy Distillation（MiniLLM / GKD / Qwen3 / Tinker）、Diffusion Post-Training（DDPO / DPOK / DRaFT / AlignProp / Diffusion-DPO / Flow-GRPO）、Diffusion / Flow Distillation（CM / iCT / sCM / CTM / LCM / DMD/DMD2 / ADD/LADD）。总数：**23 篇 first-party cheat sheet**。
- **2026-05** — ![DOCS](https://img.shields.io/badge/DOCS-blue?style=flat-square) 📖 **README 重构** —— 预览图 banner、ARIS 战绩前置（badges + 10K-star credibility paragraph）、与 [ARIS 主仓](https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep)共享微信群二维码。
</details>

---

## 📝 Blog 清单

长文技术 blog —— 手写、跨模型审过；不走受审的 `/render-html` 管线（图表版权归原作者，注明来源使用）。

| Blog | 内容 | |
|---|---|---|
| **NVIDIA Cosmos 3 —— MoT 架构深度导读**（中文） | omnimodal world model · Mixture-of-Transformers · 138 页 Cosmos 3 technical report 精读 | [📄 阅读](https://wanshuiyin.github.io/ARIS-in-AI-Offer/blogs/cosmos3_mot_guide.html) |
| **Continuous DLM 综述 —— 表征视角**（中文） | 从表征视角看连续扩散语言模型 · ELF / 字节 Cola-DLM / Flow-Matching 家族（2026 上半年） | [📄 阅读](https://wanshuiyin.github.io/ARIS-in-AI-Offer/blogs/continuous_dlm_representation_perspective.html) |
| **扩散 × 表征 × 流形**（中文） | 图像/视频扩散的「借表征 / 用流形」两条脉络 · SSL / Consistency / REPA / RAE / JiT / V-JEPA2 · 与 continuous DLM 互引 | [📄 阅读](https://wanshuiyin.github.io/ARIS-in-AI-Offer/blogs/diffusion_representation_manifold.html) |

---

## 📚 教程清单

> 🌐 **双语版**：每一篇 cheat sheet 都同时提供中文（默认）和英文 HTML 版本——文件名 `*_tutorial.html`（中文）和 `*_tutorial_en.html`（英文），下方表格的 HTML 列直接给出对应链接。

### 🧠 General / 基础

| Topic | HTML 中文 | HTML EN | MD |
|---|---|---|---|
| **Attention 面试 Cheat Sheet** | [📄 中](https://wanshuiyin.github.io/ARIS-in-AI-Offer/tutorials/attention_tutorial.html) | [📄 EN](https://wanshuiyin.github.io/ARIS-in-AI-Offer/tutorials/attention_tutorial_en.html) | [MD](docs/tutorials/attention_tutorial.md) |
| **KL Divergence in RLHF (k1/k2/k3 · placement gradient bias)** | [📄 中](https://wanshuiyin.github.io/ARIS-in-AI-Offer/tutorials/kl_divergence_rlhf_tutorial.html) | [📄 EN](https://wanshuiyin.github.io/ARIS-in-AI-Offer/tutorials/kl_divergence_rlhf_tutorial_en.html) | [MD](docs/tutorials/kl_divergence_rlhf_tutorial.md) |

### 🎯 Post-Training & Reasoning

| Topic | HTML 中文 | HTML EN | MD |
|---|---|---|---|
| **RLHF / DPO / GRPO / PPO** | [📄 中](https://wanshuiyin.github.io/ARIS-in-AI-Offer/tutorials/rlhf_dpo_grpo_ppo_tutorial.html) | [📄 EN](https://wanshuiyin.github.io/ARIS-in-AI-Offer/tutorials/rlhf_dpo_grpo_ppo_tutorial_en.html) | [MD](docs/tutorials/rlhf_dpo_grpo_ppo_tutorial.md) |
| **Reasoning Models (o1 / R1 / Test-Time Compute / PRM)** | [📄 中](https://wanshuiyin.github.io/ARIS-in-AI-Offer/tutorials/reasoning_models_tutorial.html) | [📄 EN](https://wanshuiyin.github.io/ARIS-in-AI-Offer/tutorials/reasoning_models_tutorial_en.html) | [MD](docs/tutorials/reasoning_models_tutorial.md) |
| **LLM On-Policy Distillation (MiniLLM / GKD / Qwen3 / Tinker)** | [📄 中](https://wanshuiyin.github.io/ARIS-in-AI-Offer/tutorials/llm_opd_tutorial.html) | [📄 EN](https://wanshuiyin.github.io/ARIS-in-AI-Offer/tutorials/llm_opd_tutorial_en.html) | [MD](docs/tutorials/llm_opd_tutorial.md) |

### 🏛️ LLM Architecture & Systems

| Topic | HTML 中文 | HTML EN | MD |
|---|---|---|---|
| **MoE (DeepSeek-V3 / Mixtral / Llama 4)** | [📄 中](https://wanshuiyin.github.io/ARIS-in-AI-Offer/tutorials/moe_tutorial.html) | [📄 EN](https://wanshuiyin.github.io/ARIS-in-AI-Offer/tutorials/moe_tutorial_en.html) | [MD](docs/tutorials/moe_tutorial.md) |
| **Long Context (RoPE / YaRN / NTK / MLA / StreamingLLM)** | [📄 中](https://wanshuiyin.github.io/ARIS-in-AI-Offer/tutorials/long_context_rope_yarn_mla_tutorial.html) | [📄 EN](https://wanshuiyin.github.io/ARIS-in-AI-Offer/tutorials/long_context_rope_yarn_mla_tutorial_en.html) | [MD](docs/tutorials/long_context_rope_yarn_mla_tutorial.md) |
| **KV Cache + Speculative Decoding (Medusa / EAGLE / MLA)** | [📄 中](https://wanshuiyin.github.io/ARIS-in-AI-Offer/tutorials/kv_cache_speculative_decoding_tutorial.html) | [📄 EN](https://wanshuiyin.github.io/ARIS-in-AI-Offer/tutorials/kv_cache_speculative_decoding_tutorial_en.html) | [MD](docs/tutorials/kv_cache_speculative_decoding_tutorial.md) |
| **Quantization (GPTQ / AWQ / FP8 / NVFP4 / SmoothQuant)** | [📄 中](https://wanshuiyin.github.io/ARIS-in-AI-Offer/tutorials/quantization_tutorial.html) | [📄 EN](https://wanshuiyin.github.io/ARIS-in-AI-Offer/tutorials/quantization_tutorial_en.html) | [MD](docs/tutorials/quantization_tutorial.md) |
| **Distributed Training (DDP / FSDP2 / ZeRO / TP / PP / EP / SP)** | [📄 中](https://wanshuiyin.github.io/ARIS-in-AI-Offer/tutorials/distributed_training_tutorial.html) | [📄 EN](https://wanshuiyin.github.io/ARIS-in-AI-Offer/tutorials/distributed_training_tutorial_en.html) | [MD](docs/tutorials/distributed_training_tutorial.md) |

### 🌊 Generative Models — 理论 & Tokenizers

| Topic | HTML 中文 | HTML EN | MD |
|---|---|---|---|
| **Flow Matching Quick Reference** | [📄 中](https://wanshuiyin.github.io/ARIS-in-AI-Offer/tutorials/flow_matching_tutorial.html) | [📄 EN](https://wanshuiyin.github.io/ARIS-in-AI-Offer/tutorials/flow_matching_tutorial_en.html) | [MD](docs/tutorials/flow_matching_tutorial.md) |
| **Diffusion Foundations (DDPM / Score / DDIM / EDM / CFG)** | [📄 中](https://wanshuiyin.github.io/ARIS-in-AI-Offer/tutorials/diffusion_foundations_tutorial.html) | [📄 EN](https://wanshuiyin.github.io/ARIS-in-AI-Offer/tutorials/diffusion_foundations_tutorial_en.html) | [MD](docs/tutorials/diffusion_foundations_tutorial.md) |
| **VAE / VQ-VAE / VQ-GAN / FSQ** | [📄 中](https://wanshuiyin.github.io/ARIS-in-AI-Offer/tutorials/vae_vqvae_vqgan_tutorial.html) | [📄 EN](https://wanshuiyin.github.io/ARIS-in-AI-Offer/tutorials/vae_vqvae_vqgan_tutorial_en.html) | [MD](docs/tutorials/vae_vqvae_vqgan_tutorial.md) |

### 🎨 Generation Systems — 图像 / 视频 / 3D / Diffusion 后训练

| Topic | HTML 中文 | HTML EN | MD |
|---|---|---|---|
| **Image Gen Systems (LDM / SD / SDXL / SD3 / FLUX / ControlNet)** | [📄 中](https://wanshuiyin.github.io/ARIS-in-AI-Offer/tutorials/image_generation_systems_tutorial.html) | [📄 EN](https://wanshuiyin.github.io/ARIS-in-AI-Offer/tutorials/image_generation_systems_tutorial_en.html) | [MD](docs/tutorials/image_generation_systems_tutorial.md) |
| **Video Gen (Sora / Hunyuan-Video / Kling / Wan / Movie Gen)** | [📄 中](https://wanshuiyin.github.io/ARIS-in-AI-Offer/tutorials/video_generation_tutorial.html) | [📄 EN](https://wanshuiyin.github.io/ARIS-in-AI-Offer/tutorials/video_generation_tutorial_en.html) | [MD](docs/tutorials/video_generation_tutorial.md) |
| **3D Gen (NeRF / Instant-NGP / 3DGS / SDS / Trellis)** | [📄 中](https://wanshuiyin.github.io/ARIS-in-AI-Offer/tutorials/3d_generation_tutorial.html) | [📄 EN](https://wanshuiyin.github.io/ARIS-in-AI-Offer/tutorials/3d_generation_tutorial_en.html) | [MD](docs/tutorials/3d_generation_tutorial.md) |
| **Diffusion Post-Training (DDPO / DPOK / DRaFT / AlignProp / Diffusion-DPO / Flow-GRPO)** | [📄 中](https://wanshuiyin.github.io/ARIS-in-AI-Offer/tutorials/diffusion_post_training_tutorial.html) | [📄 EN](https://wanshuiyin.github.io/ARIS-in-AI-Offer/tutorials/diffusion_post_training_tutorial_en.html) | [MD](docs/tutorials/diffusion_post_training_tutorial.md) |
| **Diffusion / Flow Distillation (CM / iCT / sCM / CTM / LCM / DMD/DMD2 / ADD/LADD)** | [📄 中](https://wanshuiyin.github.io/ARIS-in-AI-Offer/tutorials/diffusion_distillation_tutorial.html) | [📄 EN](https://wanshuiyin.github.io/ARIS-in-AI-Offer/tutorials/diffusion_distillation_tutorial_en.html) | [MD](docs/tutorials/diffusion_distillation_tutorial.md) |

### 👁️ Multimodal

| Topic | HTML 中文 | HTML EN | MD |
|---|---|---|---|
| **VLM (CLIP / LLaVA / Qwen-VL / DeepSeek-VL)** | [📄 中](https://wanshuiyin.github.io/ARIS-in-AI-Offer/tutorials/vlm_multimodal_tutorial.html) | [📄 EN](https://wanshuiyin.github.io/ARIS-in-AI-Offer/tutorials/vlm_multimodal_tutorial_en.html) | [MD](docs/tutorials/vlm_multimodal_tutorial.md) |

### 🤖 Agents

| Topic | HTML 中文 | HTML EN | MD |
|---|---|---|---|
| **Agent Foundations (ReAct / MCP / A2A / SWE-bench / GAIA / OSWorld)** | [📄 中](https://wanshuiyin.github.io/ARIS-in-AI-Offer/tutorials/agent_foundations_tutorial.html) | [📄 EN](https://wanshuiyin.github.io/ARIS-in-AI-Offer/tutorials/agent_foundations_tutorial_en.html) | [MD](docs/tutorials/agent_foundations_tutorial.md) |
| **Agentic RL (AgentTuning / ToolRL / RAGEN / WebRL / SWE-RL / GRPO for tool use)** | [📄 中](https://wanshuiyin.github.io/ARIS-in-AI-Offer/tutorials/agentic_rl_tutorial.html) | [📄 EN](https://wanshuiyin.github.io/ARIS-in-AI-Offer/tutorials/agentic_rl_tutorial_en.html) | [MD](docs/tutorials/agentic_rl_tutorial.md) |
| **Multi-Agent & Long-Horizon (CAMEL / AutoGen / MetaGPT / MoA / Debate / MemGPT / LATS)** | [📄 中](https://wanshuiyin.github.io/ARIS-in-AI-Offer/tutorials/multi_agent_long_horizon_tutorial.html) | [📄 EN](https://wanshuiyin.github.io/ARIS-in-AI-Offer/tutorials/multi_agent_long_horizon_tutorial_en.html) | [MD](docs/tutorials/multi_agent_long_horizon_tutorial.md) |
| **Self-Evolving Agents (Ctx2Skill / Native Evolution / A²RD / Voyager / Reflexion / STaR)** | [📄 中](https://wanshuiyin.github.io/ARIS-in-AI-Offer/tutorials/self_evolving_agents_tutorial.html) | [📄 EN](https://wanshuiyin.github.io/ARIS-in-AI-Offer/tutorials/self_evolving_agents_tutorial_en.html) | [MD](docs/tutorials/self_evolving_agents_tutorial.md) |

> 🎉 **23 篇 cheat sheet 全部就位**（2026-05），每篇都有中文 + 英文双语版本，覆盖 General / Post-Training / Architecture / Generative / Multimodal / Agents / Diffusion 后训练 七大类。本轮新增 4 篇：KL Divergence in RLHF、LLM On-Policy Distillation、Diffusion Post-Training、Diffusion Distillation。后续 Flow-OPD / Audio Gen / 更新 SOTA 等会逐步加 —— **PR 欢迎**（详见 [CONTRIBUTING](CONTRIBUTING.md)）。

### 🦾 Embodied AI / 具身智能

> 🌟 **社区贡献** by [@WinstonJQ](https://github.com/WinstonJQ) —— 这是社区独立维护的一份题库，托管在他自己的仓库，慷慨开源给大家。如果对你秋招有帮助，请去 ⭐ 一下原仓库，给作者一点鼓励 🙏

| Topic | HTML 中文 | Source |
|---|---|---|
| **具身智能高频面试题库** (VLA / 模仿学习 / RL / 世界模型 / 工程落地 / 腿足控制 / 3D 感知 / LeetCode·系统设计 — 413 题，8 卷) | [📄 中 (在线)](https://winstonjq.github.io/embodied-interview-qa/) | [@WinstonJQ/embodied-interview-qa](https://github.com/WinstonJQ/embodied-interview-qa) |

---

## 🤖 这些教程是怎么生成的

每篇都用了 [ARIS](https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep) 的 `/interview-cheatsheet` skill：

1. **Plan** — 12-14 节（TL;DR · 直觉 · 公式 · 代码 · 变体 · 复杂度 · 25 高频题）
2. **Draft** — ~600-1000 行中文 + 真能跑的 PyTorch
3. **Cross-model review** — 跨模型 codex GPT-5.5 xhigh 审 10 项（公式正确性 / 代码可运行 / 引用真实 / 表格 pipe 转义 / callout 风格 / 个人信息泄漏…）
4. **Fix 循环**（trajectory-based，FAIL 集在收敛就继续，同一问题反复出现或 ~6 轮没收敛就停）
5. **`/render-html`** 渲染 + 13 项渲染审查（信息保真 / TOC / 公式 / 代码高亮 / 安全 / 隐私…）
6. **`.review.json`** 完整审计 trail

跨模型对抗审查（executor != reviewer 家族）是 ARIS 的核心不变量——LLM 自己审自己等于没审。

---

## 🌐 ARIS-Homepage —— fact-checked 学术主页生成器

> **唯一一个发布前 fact-check 你 CV 的个人主页生成器。**

本仓库新增的 skill：`/homepage-generator` 把你的 CV（`.docx` / `.pdf` / `.txt`）变成一个 polished 单文件学术主页。跨模型 fact-check 对 DBLP / arXiv 审核——错 venue / 错年份 / 错作者 / 编造奖项都硬阻断 ship，除非你显式 override。

**Live demo**：[wanshuiyin.github.io](https://wanshuiyin.github.io/) —— 这就是用这个 skill 跑 CV + 维护者之前 manual 主页（作 editorial reference）生成的。预览 strip 在 README 顶部。

### 工作流程

```
                       ARIS-Homepage Pipeline

   📄 CV (.docx/.pdf/.txt)    🌐 Manual Homepage URL    🖼 素材目录
   事实源                     编辑源（可选）            视觉源（可选）
         │                            │                      │
         ▼                            │                      │
   ┌──────────┐                       │                      │
   │ init     │                       │                      │
   │ CV→text  │                       │                      │
   └─────┬────┘                       │                      │
         ▼                            ▼                      ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │ 🤖 调用方 LLM agent 读 EXTRACTION_HANDOFF.md + 可选 manual      │
   │    homepage URL + 素材目录作 context                             │
   │ → 写 .aris-homepage/extraction.json                             │
   └─────────────────────────┬───────────────────────────────────────┘
                             ▼
                       ┌──────────┐
                       │ finalize │
                       └─────┬────┘
                             ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │ ✋ 可编辑源文件（事实在这，IDE 里改）：                          │
   │   profile.yml · publications.bib · bio.md · news.md             │
   │   EXTRACTION_REVIEW.md（审 LLM 不确定的抽取项）                 │
   └─────────────────────────┬───────────────────────────────────────┘
                             ▼
                  ┌────────────────────────┐
                  │ render                 │
                  │   --persona            │
                  │     theory-minimal     │
                  └───────────┬────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        ┌──────────┐    ┌──────────┐    ┌──────────────┐
        │ Layer-1  │    │ Layer-2  │    │ Layer-2      │
        │ DBLP /   │    │ Codex MCP│    │ Gemini       │
        │ arXiv    │    │ 对抗 rev │    │ 视觉 critique│
        │ 事实审   │    │（可选）  │    │（可选）      │
        │（默认）  │    │          │    │              │
        └─────┬────┘    └──────────┘    └──────────────┘
              │
              ▼
        ┌──────────────┐
        │ index.html + │
        │ audit-report │ ──▶ 🚀 部署：GitHub Pages · S3 · 邮箱 · 任意
        │   .md        │
        └──────────────┘

   典型流程（7 步，~5 分钟）：
     1. aris-homepage init --from-cv ./cv.pdf --out ./site
     2.（调用方 agent）读 .aris-homepage/EXTRACTION_HANDOFF.md
        → 填 .aris-homepage/extraction.json
     3. aris-homepage finalize
     4. $EDITOR profile.yml publications.bib bio.md news.md
     5. aris-homepage check --strict        # 只做 fact-check
     6. aris-homepage render --persona theory-minimal
     7. 看 audit-report.md；修了 → 重 render 或 --override-all

   最小运行时：Python + 调用方 LLM agent
   Codex MCP 可选（跨模型对抗 review）
   Gemini 可选（多模态视觉 critique）
```

- **Skill 契约**：[`skills/homepage-generator/SKILL.md`](skills/homepage-generator/SKILL.md)
- **完整 schema**：[`skills/homepage-generator/PROFILE_SCHEMA.md`](skills/homepage-generator/PROFILE_SCHEMA.md)
- **实现**：[`tools/aris_homepage.py`](tools/aris_homepage.py)（纯 stdlib Python，只需 `pip install pyyaml`）
- **模板**：[`tools/templates/homepage-theory-minimal.html`](tools/templates/homepage-theory-minimal.html)

### 快速开始

```bash
aris-homepage init --from-cv ./cv.pdf --out ./site
cd ./site
# 调用方 agent 读 .aris-homepage/EXTRACTION_HANDOFF.md，填 extraction.json
aris-homepage finalize
$EDITOR profile.yml             # 调整编辑选择
aris-homepage render --persona theory-minimal
```

输出：`index.html` + `audit-report.md`。HTML 扔 GitHub Pages、S3、学校 `~user/public_html/`、邮箱附件都行——零 build server。**最小运行时只要 Python + 调用方 LLM agent**；Codex MCP 可选（增强 adversarial 跨模型 review），Gemini 可选（多模态视觉 critique）。

---

## 🤝 欢迎贡献

一个人的力量有限，希望靠大家一起把这套教程做得更完善。

完整贡献指南见 [**CONTRIBUTING.md**](CONTRIBUTING.md)（[English](CONTRIBUTING.md) · [中文](CONTRIBUTING_CN.md)）—— 含 ARIS workflow 调用、严格风格指南（headings / math / tables / callouts / 个人信息 banlist）、PR checklist。

**TL;DR**：用 ARIS 的 [`/interview-cheatsheet`](skills/interview-cheatsheet/SKILL.md) + [`/render-html`](skills/render-html/SKILL.md) workflow 跑出来再提 PR；两个 skill 都内置跨模型 codex 5.5 xhigh 审查 gate（数学 / 代码 / 引用 / 渲染保真），过了就有质量底线。Skill 源码和 `tools/render_html.py` 都在这个仓库里，可以直接 fork。

**坦白说**：现有教程把 HTML 基础结构（公式 / 代码 / 表格 / callout / TOC / 响应式）做扎实了，但某些主题最前沿（2025 下半年才出的方法、某些细分领域最新论文）大概率没全覆盖。发现哪里过时或有错，PR / issue 都欢迎。

---

## 💬 社区

**与 [ARIS 主仓](https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep) 共享一个社区** —— 同一个微信群既讨论 ARIS skill workflow，也讨论这份秋招 cheat sheet 合集。进群可以讨论面试准备、提新的 cheat sheet 主题需求、或反馈勘误 / 贡献：

<p align="center">
  <img src="assets/wechat_group.jpg" alt="WeChat 群二维码（与 ARIS 主仓共享）" width="300">
</p>

### 🔭 社区衍生

社区基于这套教程做的二次创作（本仓库 MIT license，欢迎保留署名的转载与整合）：

- **[大模型秋招教程 (ARIS-in-AI-Offer & Hello-Agents)](https://qizishi.github.io/Autumn-Recruitment-Tutorials-for-LLM/)** by [@QiZishi](https://github.com/QiZishi) —— 把这里的 23 篇中文教程与 [Datawhale Hello-Agents](https://github.com/datawhalechina/hello-agents) 的大模型面试问答整合成可跳转的教程卡片索引站，方便在线随时阅读（[仓库](https://github.com/QiZishi/Autumn-Recruitment-Tutorials-for-LLM) · 来自 [#3](https://github.com/wanshuiyin/ARIS-in-AI-Offer/issues/3)）。

基于这套教程做了新东西？开个 issue，我们把它列在这里。

---

## 🌟 ARIS 是什么 — 顺便安利一下

[**ARIS — Auto Research in Sleep**](https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep) 是 2025-2026 年最受关注的 AI 科研 agent skill 平台之一。这个仓库生成所用的 `/interview-cheatsheet` + `/render-html` 是 ARIS 74+ 个 skill 中的两个。

[![Stars](https://img.shields.io/github/stars/wanshuiyin/Auto-claude-code-research-in-sleep?style=flat&logo=github&logoColor=white&color=gold&label=Stars)](https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep/stargazers) · [![arXiv](https://img.shields.io/badge/arXiv-2605.03042-b31b1b?style=flat&logo=arxiv)](https://huggingface.co/papers/2605.03042) · [![HF Daily #1](https://img.shields.io/badge/HF%20Daily%20Papers-%231-yellow?style=flat)](https://huggingface.co/papers/2605.03042) · [![PaperWeekly](https://img.shields.io/badge/Featured%20on-PaperWeekly-red?style=flat)](https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep) · [![awesome-agent-skills](https://img.shields.io/badge/Featured%20in-awesome--agent--skills-blue?style=flat&logo=github)](https://github.com/VoltAgent/awesome-agent-skills) · [![Project of the Day](https://img.shields.io/badge/AI%20Digital%20Crew-Project%20of%20the%20Day-orange?style=flat)](https://aidigitalcrew.com)

- ⭐ **~10k GitHub stars** — top trending AI agent repo
- 🥇 **HuggingFace Daily Papers #1** — top of the day, paper [arXiv:2605.03042](https://huggingface.co/papers/2605.03042)
- 🏆 **AI Digital Crew · Project of the Day** (2026.03.14)
- 📰 **Featured on PaperWeekly** + **VoltAgent/awesome-agent-skills**
- 🛠️ **74+ research skills** — 从找 idea → 跑实验 → 写论文 → rebuttal → 做 talk slides 的全流程
- 🌐 **7+ 平台支持** — Claude Code · Codex CLI · Cursor · Trae · Antigravity · GitHub Copilot CLI · OpenClaw
- 🔧 **ARIS-Code 独立 CLI** — 不想绑定 Claude Code 也行，自带 multi-provider runtime

核心方法论：**跨模型对抗审查**——executor 和 reviewer 必须不同模型家族（Claude × GPT-5.5 xhigh × Gemini），不让 LLM 自己审自己。这套协议复用到面试 cheat sheet 生成上，就保证了每篇里的公式 / 代码 / 引用都过了一遍独立审查（详见每篇旁边的 `.review.json` 审计 trail）。

👉 **ARIS 主仓库**：https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep

---

## 📖 引用 ARIS

如果这份合集——或里面任何一篇 cheat sheet——对你的面试准备 / 科研 / 论文有帮助，欢迎引用底层 ARIS 方法论论文：

```bibtex
@article{yang2026aris,
  title={ARIS: Autonomous Research via Adversarial Multi-Agent Collaboration},
  author={Yang, Ruofeng and Li, Yongcan and Li, Shuai},
  journal={arXiv preprint arXiv:2605.03042},
  year={2026}
}
```

本仓库每一篇教程都由 ARIS 的 `/interview-cheatsheet` + `/render-html` workflow 跑跨模型对抗审查（Claude × GPT-5.5 xhigh × Gemini）端到端生成。引用是为了支持 workflow 背后的方法论，不只是这份合集本身。

---

## License

[MIT](LICENSE) — 用、改、传、二开都行。希望对正在准备秋招的你有帮助。加油 💪
