#!/usr/bin/env python3
"""build_index.py — generate docs/index.html, the one-link hub for every tutorial.

The catalogue lives in this file (CATALOG below). Adding a tutorial = one entry
here + `python3 tools/build_index.py`. The page is a single static HTML: every
card is written into the markup (readable with JavaScript off); the script layer
only adds search, the 中/EN switch, dark mode, "抽一篇", and per-reader 已读 marks
kept in localStorage. No CDN, no fonts, no build system.

Run from the repo root:
    python3 tools/build_index.py
"""
from __future__ import annotations

import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "index.html"
SITE = "https://wanshuiyin.github.io/ARIS-in-AI-Offer"
REPO = "https://github.com/wanshuiyin/ARIS-in-AI-Offer"

# id, 中文, English, emoji
CATEGORIES = [
    ("found", "基础", "Foundations", "🧠"),
    ("post", "后训练与推理", "Post-Training & Reasoning", "🎯"),
    ("arch", "架构与系统", "Architecture & Systems", "🏛️"),
    ("gentheory", "生成模型理论", "Generative Theory", "🌊"),
    ("gensys", "生成系统", "Generation Systems", "🎨"),
    ("mm", "多模态", "Multimodal", "👁️"),
    ("agent", "Agent", "Agents", "🤖"),
    ("embodied", "具身与世界模型", "Embodied & World Models", "🦾"),
    ("blog", "长文 Blog", "Long-form Blogs", "📝"),
]

# slug, cat, 中文标题, English title, tags, questions, code scripts
def T(slug, cat, cn, en, tags, q=25, code=(), tags_en=None):
    return dict(slug=slug, cat=cat, cn=cn, en=en, tags=tags, tags_en=tags_en or tags, q=q, code=list(code))
CATALOG = [
    T("attention_tutorial", "found", "Attention", "Attention", "MHA · fused QKV+chunk · MQA/GQA · KV cache · FlashAttention · RoPE", 25, ("mha.py", "axial_attention.py")),
    T("transformer_block_tutorial", "found", "Transformer Block", "Transformer Block", "Pre/Post-LN · 残差拓扑 · MHA/MQA/GQA/MLA · Dense FFN vs MoE · GPT-2→Llama", 30, ("transformer_block.py",), tags_en="Pre/Post-LN · residual topologies · MHA/MQA/GQA/MLA · Dense FFN vs MoE · GPT-2→Llama"),
    T("llm_evaluation_benchmarking_tutorial", "found", "LLM 评测与 Benchmark", "LLM Evaluation & Benchmarking", "pass@k 无偏估计 · LLM-as-judge · 污染检测 · Elo/Bradley-Terry", 25, ("llm_eval_metrics.py",), tags_en="unbiased pass@k · LLM-as-judge · contamination checks · Elo/Bradley-Terry"),
    T("normalization_init_tutorial", "found", "归一化 / 残差 / 初始化", "Normalization / Residual / Init", "BatchNorm/LayerNorm/RMSNorm · Pre-vs-Post-LN · DeepNorm · QK-Norm · Kaiming/Xavier · μP", 25, ("normalization.py",)),
    T("optimizer_lr_schedule_tutorial", "found", "优化器 / LR Schedule", "Optimizers & LR Schedules", "SGD·Momentum · Adam·AdamW · Muon·Lion·Shampoo·SOAP · warmup·cosine·WSD", 25, ("optimizer_lr_schedule.py",)),
    T("tokenization_tutorial", "found", "Tokenization / 分词器", "Tokenization", "BPE · WordPiece · Unigram·SentencePiece · byte-level · 词表五量 · fertility·BPB", 25, ("tokenization.py",), tags_en="BPE · WordPiece · Unigram·SentencePiece · byte-level · the five vocab sizes · fertility·BPB"),
    T("kl_divergence_rlhf_tutorial", "found", "RLHF 里的 KL 散度", "KL Divergence in RLHF", "k1/k2/k3 估计 · KL 放在 reward 还是 loss · 梯度偏差", 25, tags_en="k1/k2/k3 estimators · KL in the reward vs in the loss · gradient bias"),
    T("rlhf_dpo_grpo_ppo_tutorial", "post", "RLHF / DPO / GRPO / PPO", "RLHF / DPO / GRPO / PPO", "reward model · PPO-clip · DPO 推导 · GRPO 组内 baseline · KL 正则", 25, tags_en="reward model · PPO-clip · DPO derivation · GRPO group baseline · KL regularisation"),
    T("reasoning_models_tutorial", "post", "推理模型", "Reasoning Models", "o1 / R1 · test-time compute · PRM vs ORM · 长 CoT RL", 25, tags_en="o1 / R1 · test-time compute · PRM vs ORM · long-CoT RL"),
    T("llm_opd_tutorial", "post", "LLM On-Policy 蒸馏", "LLM On-Policy Distillation", "MiniLLM · GKD · Qwen3 · Tinker · reverse-KL vs forward-KL", 25),
    T("lora_peft_tutorial", "post", "LoRA / PEFT", "LoRA / PEFT", "LoRA · QLoRA · DoRA · rsLoRA · PiSSA · AdaLoRA · (IA)³", 25, ("lora.py",)),
    T("moe_tutorial", "arch", "MoE 混合专家", "MoE (Mixture-of-Experts)", "DeepSeek-V3 · Mixtral · Llama 4 · 路由/负载均衡 · 专家并行", 25, tags_en="DeepSeek-V3 · Mixtral · Llama 4 · routing / load balancing · expert parallelism"),
    T("long_context_rope_yarn_mla_tutorial", "arch", "长上下文", "Long Context", "RoPE · YaRN · NTK · MLA · StreamingLLM · 位置外推", 25, tags_en="RoPE · YaRN · NTK · MLA · StreamingLLM · position extrapolation"),
    T("linear_sparse_attention_tutorial", "arch", "线性 / 稀疏注意力", "Linear / Sparse Attention", "Linear Attn · SSM·Mamba · Mamba-2·SSD · DeltaNet · NSA·MoBA · Hybrid", 25, ("linear_sparse_attention.py",)),
    T("kv_cache_speculative_decoding_tutorial", "arch", "KV Cache + 投机解码", "KV Cache + Speculative Decoding", "PagedAttention · Medusa · EAGLE · MLA · 接受率与加速比", 25, tags_en="PagedAttention · Medusa · EAGLE · MLA · acceptance rate and speed-up"),
    T("quantization_tutorial", "arch", "量化", "Quantization", "GPTQ · AWQ · SmoothQuant · FlatQuant · FP8/NVFP4 · QAT: LSQ·ParetoQ·Gemma-QAT · FP4 训练", 25, tags_en="GPTQ · AWQ · SmoothQuant · FlatQuant · FP8/NVFP4 · QAT: LSQ·ParetoQ·Gemma-QAT · FP4 training"),
    T("distributed_training_tutorial", "arch", "分布式训练", "Distributed Training", "DDP · FSDP2 · ZeRO · TP · PP · EP · SP · 通信量推导", 25, tags_en="DDP · FSDP2 · ZeRO · TP · PP · EP · SP · communication-volume derivations"),
    T("llm_pretraining_pipeline_tutorial", "arch", "LLM 预训练流水线", "LLM Pretraining Pipeline", "Kaplan vs Chinchilla · 数据工厂 · 去重/质量过滤 · 课程与退火", 26, ("pretraining_pipeline.py",), tags_en="Kaplan vs Chinchilla · corpus factory · dedup / quality filtering · curriculum and annealing"),
    T("llm_inference_serving_tutorial", "arch", "LLM 推理与 Serving", "LLM Inference & Serving Stack", "请求状态机 · 采样算子 · PagedAttention · continuous batching · prefill/decode 分离", 25, ("inference_serving.py",), tags_en="request state machine · sampling operators · PagedAttention · continuous batching · prefill/decode disaggregation"),
    T("flow_matching_tutorial", "gentheory", "Flow Matching 速查", "Flow Matching Quick Reference", "Rectified Flow · 条件路径 vs 边缘路径 · Euler/Heun 采样 · CFG", 0, ("flow_matching.py",), tags_en="Rectified Flow · conditional vs marginal paths · Euler/Heun sampling · CFG"),
    T("diffusion_foundations_tutorial", "gentheory", "Diffusion 基础", "Diffusion Foundations", "DDPM · Score · DDIM · EDM · CFG · 训练/采样统一视角", 25, tags_en="DDPM · Score · DDIM · EDM · CFG · one view of training and sampling"),
    T("vae_vqvae_vqgan_tutorial", "gentheory", "VAE / VQ-VAE / VQ-GAN / FSQ", "VAE / VQ-VAE / VQ-GAN / FSQ", "ELBO · 重参数化 · codebook collapse · FSQ/LFQ · 感知损失", 25, tags_en="ELBO · reparameterisation · codebook collapse · FSQ/LFQ · perceptual loss"),
    T("image_generation_systems_tutorial", "gensys", "图像生成系统", "Image Generation Systems", "LDM · SD · SDXL · SD3 · FLUX · ControlNet · MMDiT", 25, ("mmdit_block.py", "toy_mmdit_t2i_pipeline.py")),
    T("video_generation_tutorial", "gensys", "视频生成", "Video Generation", "Sora · Hunyuan-Video · Kling · Wan · Movie Gen · 时空 token 化", 25, tags_en="Sora · Hunyuan-Video · Kling · Wan · Movie Gen · spatio-temporal tokenisation"),
    T("3d_generation_tutorial", "gensys", "3D 生成", "3D Generation", "NeRF · Instant-NGP · 3DGS · SDS · DUSt3R→VGGT · 原生 mesh · TRELLIS/Hunyuan3D 2.x · sim-ready", 25, tags_en="NeRF · Instant-NGP · 3DGS · SDS · DUSt3R→VGGT · native mesh · TRELLIS/Hunyuan3D 2.x · sim-ready"),
    T("diffusion_post_training_tutorial", "gensys", "Diffusion 后训练", "Diffusion Post-Training", "DDPO · DPOK · DRaFT · AlignProp · ReFL · Diffusion-DPO · reward hacking", 25),
    T("modern_diffusion_post_training_tutorial", "gensys", "现代 Diffusion 后训练", "Modern Diffusion Post-Training", "Flow-GRPO · DGPO · DiffusionNFT · reverse SDE · log Z 抵消 · CFG-free", 25, ("diffusion_online_rl.py",), tags_en="Flow-GRPO · DGPO · DiffusionNFT · reverse SDE · log Z cancellation · CFG-free"),
    T("diffusion_distillation_tutorial", "gensys", "Diffusion / Flow 蒸馏", "Diffusion / Flow Distillation", "CM · iCT · sCM · CTM · LCM · DMD/DMD2 · ADD/LADD", 25),
    T("vlm_multimodal_tutorial", "mm", "VLM / 多模态", "VLM / Multimodal", "CLIP · LLaVA · Qwen-VL · DeepSeek-VL · 视觉 token · 对齐训练三阶段", 25, tags_en="CLIP · LLaVA · Qwen-VL · DeepSeek-VL · visual tokens · three-stage alignment"),
    T("agent_foundations_tutorial", "agent", "Agent 基础", "Agent Foundations", "ReAct · MCP · A2A · SWE-bench · GAIA · OSWorld · 工具调用", 25, tags_en="ReAct · MCP · A2A · SWE-bench · GAIA · OSWorld · tool calling"),
    T("agentic_rl_tutorial", "agent", "Agentic RL", "Agentic RL", "AgentTuning · ToolRL · RAGEN · WebRL · SWE-RL · 多轮 GRPO", 25, tags_en="AgentTuning · ToolRL · RAGEN · WebRL · SWE-RL · multi-turn GRPO"),
    T("multi_agent_long_horizon_tutorial", "agent", "多智能体与长程", "Multi-Agent & Long-Horizon", "CAMEL · AutoGen · MetaGPT · MoA · Debate · MemGPT · LATS", 25),
    T("self_evolving_agents_tutorial", "agent", "自进化 Agent", "Self-Evolving Agents", "Ctx2Skill · Native Evolution · A²RD · Voyager · Reflexion · STaR", 25),
    T("world_models_tutorial", "embodied", "World Models / 世界模型", "World Models", "RSSM·Dreamer·MuZero·TD-MPC2 · JEPA·V-JEPA 2 · Genie·GameNGen·Cosmos 3·Atlas · 具身三条路 · 评测", 25, ("world_models_toy.py",), tags_en="RSSM·Dreamer·MuZero·TD-MPC2 · JEPA·V-JEPA 2 · Genie·GameNGen·Cosmos 3·Atlas · three embodied routes · evaluation"),
    T("rag_embedding_retrieval_tutorial", "agent", "RAG + 向量检索", "RAG + Embedding / Retrieval", "InfoNCE · 难负例 · Matryoshka · BM25 · RRF · ColBERT · GraphRAG", 25, ("rag_embedding.py",), tags_en="InfoNCE · hard negatives · Matryoshka · BM25 · RRF · ColBERT · GraphRAG"),
]

EXTERNAL = [  # community-contributed, hosted elsewhere
    dict(cat="embodied", cn="具身智能高频面试题库", en="Embodied AI Interview Q&A Bank",
         tags="VLA · 模仿学习 · RL · 世界模型 · 腿足控制 · 3D 感知 · 413 题 / 8 卷", tags_en="VLA · imitation learning · RL · world models · legged control · 3D perception · 413 questions / 8 volumes",
         url="https://winstonjq.github.io/embodied-interview-qa/",
         src="https://github.com/WinstonJQ/embodied-interview-qa", author="@WinstonJQ"),
]

BLOGS = [
    dict(cn="NVIDIA Cosmos 3 — MoT 架构深度导读", en="NVIDIA Cosmos 3 — MoT Architecture Deep-Dive",
         tags="omnimodal world model · Mixture-of-Transformers · 138 页技术报告走读", tags_en="omnimodal world model · Mixture-of-Transformers · a walk through the 138-page report", file="blogs/cosmos3_mot_guide.html"),
    dict(cn="连续 DLM 综述 — 表征视角", en="A Survey on Continuous DLM — Representation Perspective",
         tags="ELF · Cola-DLM · Flow-Matching 家族 · 2026 H1", tags_en="ELF · Cola-DLM · the Flow-Matching family · 2026 H1", file="blogs/continuous_dlm_representation_perspective.html"),
    dict(cn="Diffusion × 表征 × 流形", en="Diffusion × Representation × Manifold",
         tags="SSL · Consistency · REPA · RAE · JiT · V-JEPA2", tags_en="SSL · Consistency · REPA · RAE · JiT · V-JEPA2", file="blogs/diffusion_representation_manifold.html"),
]

def e(s: str) -> str:
    return html.escape(s, quote=True)

def card(t: dict) -> str:
    cn_url = f"tutorials/{t['slug']}.html"
    en_url = f"tutorials/{t['slug']}_en.html"
    md_cn = f"{REPO}/blob/main/docs/tutorials/{t['slug']}.md"
    md_en = f"{REPO}/blob/main/docs/tutorials/{t['slug']}_en.md"
    q = (f'<span class="b">{t["q"]} <span class="s-cn">题</span><span class="s-en">Q</span></span>' if t["q"]
         else '<span class="b"><span class="s-cn">速查</span><span class="s-en">quick ref</span></span>')
    code = "".join(f'<a class="b b-code" href="{REPO}/blob/main/docs/tutorials/code/{c}" title="{e(c)}">{e(c)}</a>' for c in t["code"])
    return (f'<article class="card" data-id="{e(t["slug"])}" data-cat="{t["cat"]}" data-cn="{e(cn_url)}" data-en="{e(en_url)}" '
            f'data-text="{e((t["cn"] + " " + t["en"] + " " + t["tags"] + " " + t["tags_en"]).lower())}">'
            f'<a class="title" href="{e(cn_url)}"><span class="t-cn">{e(t["cn"])}</span><span class="t-en">{e(t["en"])}</span></a>'
            f'<p class="tags"><span class="s-cn">{e(t["tags"])}</span><span class="s-en">{e(t["tags_en"])}</span></p>'
            f'<div class="meta">{q}{code}<a class="b" href="{e(en_url)}" data-alt="en">EN</a><a class="b" href="{e(cn_url)}" data-alt="cn">中</a>'
            f'<a class="b" href="{e(md_cn)}" data-alt="cn">MD</a><a class="b" href="{e(md_en)}" data-alt="en">MD</a>'
            f'<label class="read js-only"><input type="checkbox" data-read="{e(t["slug"])}"><span class="s-cn">已读</span><span class="s-en">read</span></label></div>'
            f'</article>')

def ext_card(x: dict) -> str:
    return (f'<article class="card ext" data-cat="{x["cat"]}" data-cn="{e(x["url"])}" data-en="{e(x["url"])}" data-text="{e((x["cn"]+" "+x["en"]+" "+x["tags"]).lower())}">'
            f'<a class="title" href="{e(x["url"])}"><span class="t-cn">{e(x["cn"])}</span><span class="t-en">{e(x["en"])}</span></a>'
            f'<p class="tags"><span class="s-cn">{e(x["tags"])}</span><span class="s-en">{e(x["tags_en"])}</span></p>'
            f'<div class="meta"><span class="b b-ext"><span class="s-cn">社区贡献</span><span class="s-en">community</span> · {e(x["author"])}</span>'
            f'<a class="b" href="{e(x["src"])}">⭐ <span class="s-cn">给作者点星</span><span class="s-en">star the source</span></a></div>'
            f'</article>')

def blog_card(b: dict) -> str:
    return (f'<article class="card" data-cat="blog" data-cn="{e(b["file"])}" data-en="{e(b["file"])}" data-text="{e((b["cn"]+" "+b["en"]+" "+b["tags"]).lower())}">'
            f'<a class="title" href="{e(b["file"])}"><span class="t-cn">{e(b["cn"])}</span><span class="t-en">{e(b["en"])}</span></a>'
            f'<p class="tags"><span class="s-cn">{e(b["tags"])}</span><span class="s-en">{e(b["tags_en"])}</span></p>'
            f'<div class="meta"><span class="b"><span class="s-cn">中文长文</span><span class="s-en">Chinese long-form</span></span></div></article>')

def build() -> str:
    n = len(CATALOG)
    sections = []
    for cid, cn, en, emoji in CATEGORIES:
        if cid == "blog":
            items = [blog_card(b) for b in BLOGS]
        elif cid == "embodied":
            items = [card(t) for t in CATALOG if t["cat"] == cid] + [ext_card(x) for x in EXTERNAL]
        else:
            items = [card(t) for t in CATALOG if t["cat"] == cid]
        note = ""
        if cid == "embodied":
            note = ('<p class="note"><span class="s-cn">第一方的 World Models 篇在前；具身题库由社区作者维护、外站托管，有帮助的话去源仓库点个 ⭐ 谢谢作者。</span>'
                    '<span class="s-en">The World Models sheet is first-party; the embodied Q&A bank is community-maintained and hosted elsewhere — star the source repo to thank the author.</span></p>')
        if cid == "blog":
            note = ('<p class="note"><span class="s-cn">三篇中文长文：世界模型架构、连续扩散语言模型综述、扩散与表征学习。</span>'
                    '<span class="s-en">Three Chinese long reads: a world-model architecture, a survey of continuous diffusion LMs, diffusion meets representation learning.</span></p>')
        sections.append(f'<section class="cat" data-cat="{cid}"><h2 id="{cid}">{emoji} <span class="s-cn">{e(cn)}</span><span class="s-en">{e(en)}</span> <small>{len(items)}</small></h2>{note}<div class="grid">{"".join(items)}</div></section>')
    chips = "".join(f'<button class="chip" data-chip="{cid}">{emoji} <span class="s-cn">{e(cn)}</span><span class="s-en">{e(en)}</span></button>' for cid, cn, en, emoji in CATEGORIES)
    tpl = (ROOT / "tools" / "templates" / "index.html").read_text(encoding="utf-8")
    return (tpl.replace("{{N}}", str(n)).replace("{{NCAT}}", str(len(CATEGORIES) - 1))
               .replace("{{CHIPS}}", chips).replace("{{SECTIONS}}", "".join(sections))
               .replace("{{SITE}}", SITE).replace("{{REPO}}", REPO))

if __name__ == "__main__":
    OUT.write_text(build(), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size:,} bytes, {len(CATALOG)} tutorials)")
