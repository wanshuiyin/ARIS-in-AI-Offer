## §0 TL;DR Cheat Sheet

> 💡 **9 句话搞定 Multi-Agent & Long-Horizon Agent** — 2024-2026 LLM agent 第二次浪潮，一页拿下面试核心。

1. **范式定位**：单 LLM = "一次 forward 的 system 1"；agent = "把推理拆成 perceive → plan → act → reflect 的 system 2"。**multi-agent** = 多个 LLM role-play 不同身份协作；**long-horizon** = 同一 agent 跨多 turn / 多天保持目标。两者正交但常一起出现（如 ChatDev, MetaGPT）。

2. **multi-agent 三大原型**：
   - **role-play 对话**（CAMEL, Li et al. NeurIPS 2023, arXiv 2303.17760）：assistant-user 双角色 inception prompting；
   - **SOP / 流水线**（MetaGPT, Hong et al. ICLR 2024, arXiv 2308.00352）：把软件开发拆成 PM / Architect / Engineer / QA 七个固定角色；
   - **debate / aggregation**（Du et al. ICML 2024 arXiv 2305.14325 + Liang et al. EMNLP 2024 arXiv 2305.19118）：N agent 独立解答 → 互看 → 多轮迭代到一致。

3. **MoA (Mixture-of-Agents, Wang et al. ICLR 2025 Spotlight, arXiv 2406.04692)**：N 个 proposer 各出回答，aggregator 把 N 个回答 concat 进 prompt 再合成。开源 6.5B-70B 组合在 AlpacaEval 2.0 上**超过 GPT-4 Omni**。

4. **debate 收敛——经验观察 + 理论近似**：把每个 agent 的 update 视作对 peer 答案分布的混合算子，若该算子是 **doubly-stochastic averaging** 形（majority-vote softmax + temperature β），按 consensus dynamics 理论（线性 averaging 算子在 doubly-stochastic graph 上）会收敛到 **fixed-point set**（一致分布或离散一致 cluster），**而非唯一 fixed point**（Banach contraction 一般不成立——因为 averaging 算子有特征值 1）。经验上 N=3 与 N=5 在 GSM8K 上 accuracy gap ≈ 1-2pp（diminishing return，见 Du 2023 Fig 4 + Liang 2024 Table 3）。

5. **MemGPT (Packer et al., arXiv 2310.08560, 2023-10)**：把 LLM context 类比 OS RAM，把外部存储类比 disk。**page fault** = 模型 function-call "search archival memory" 触发的 retrieval。延迟代价：一次 page fault ≈ 一次额外 LLM call（数百 ms ~ 数秒），但避免了 context overflow。后续以 **Letta** 名义工程化（2024+）。

6. **GraphRAG (Microsoft 2024, arXiv 2404.16130)**：vector RAG 在 multi-hop 上崩；GraphRAG 先用 LLM 抽 entity-relation 三元组，建知识图谱 + community detection，按层级摘要后再做 retrieval-augmented generation。**global query**（"给整篇文档总结主题"）比 vector RAG 强 70-80%。

7. **tree-search agent**：
   - **ToT (Yao et al., NeurIPS 2023, arXiv 2305.10601)** = BFS/DFS + LLM evaluator；
   - **RAP (Hao et al., EMNLP 2023, arXiv 2305.14992)** = MCTS + world-model rollout；
   - **LATS (Zhou et al., ICML 2024, arXiv 2310.04406)** = MCTS + 自反思 (Reflexion) + value 估计；
   - **Agent-Q (Putta et al., 2024, arXiv 2408.07199)** = MCTS + DPO 离线训练 policy。共性：用 **PUCT** 公式 $a^* = \arg\max_a Q(s,a) + c_\text{puct} P(s,a) \sqrt{N(s)} / (1+N(s,a))$ 平衡 exploit / explore。

8. **long-horizon benchmark 三件套**：**TAU-bench**（Yao et al. 2024 arXiv 2406.12045，customer service multi-turn）、**OSWorld** (Xie et al. NeurIPS 2024 arXiv 2404.07972，real OS GUI)、**SWE-bench** (Jimenez et al., ICLR 2024, arXiv 2310.06770，真实 GitHub issue 修复)。SOTA 2026-05 在 SWE-bench Verified ≈ 75%（Claude 4.6 Sonnet / o3 等），但 OSWorld 仍 < 60%。

9. **易踩坑**：multi-agent 不是越多越好（cost ∝ N, accuracy gain ∝ log N）；long-CoT context overflow 不是写更长 context window 能解（**lost-in-the-middle**, Liu et al. TACL 2024 arXiv 2307.03172）；memory stale 比 memory missing 更危险（错误 retrieval 会产生 confident wrong answer）；sub-agent blame-shifting（"那是 worker A 的错"）在 hierarchical orchestrator 模式中常见。

## §1 直觉：从 single LLM 到 agent system

### 1.1　为什么 single LLM 不够

把 LLM 当成 $y = f_\theta(x)$ 一次 forward 的 oracle，有三个本质限制：

- **状态有界**：context window 是物理上限（即使 1M token 也有限），无法长期保持目标
- **决策一锤定音**：一次 sampling 错了就错了，没有"undo / replan" 机制
- **能力切片**：单 prompt 既要 reason 又要 act 又要 verify，**role drift**——前段 reasoning 与后段 verification 互相干扰

**agent** 的本质是把这三个问题外部化：

| LLM 内置 | Agent 外置 |
|---|---|
| context window | persistent memory（vector store / KG / disk） |
| one-shot sampling | iterative loop: perceive → plan → act → reflect |
| single role | multi-agent role specialization |

> 💡 **mental model — agent = LLM × harness × memory × tools** — 一个 agent 不是一个 LLM，而是 LLM 套在 outer loop 上。harness 决定 LLM 什么时候被 call、call 什么、怎么把 output 喂回去。Claude Code Agent / Cursor / Devin / SWE-Agent 本质都是 **不同 harness 的差异**——同样的 LLM，harness 设计差距能拉开 30-50pp 的 task success rate。

### 1.2　multi-agent vs long-horizon vs agentic

这三个词常被混用，但**精确含义不同**：

| 概念 | 关注点 | 典型代表 |
|---|---|---|
| **multi-agent** | 多个 LLM role-play 不同身份**协作 / 对抗** | CAMEL, AutoGen, MetaGPT, debate |
| **long-horizon** | 同一 agent 跨**多 turn / 长时间**保持目标 | MemGPT, Voyager, OSWorld, SWE-Lancer |
| **agentic** | LLM 具备**自主 plan + tool use + reflection** 能力（≥1 agent，≥1 step） | Toolformer, ReAct, AutoGPT |

→ 一个真实系统通常**三个都涉及**（比如 Devin = agentic（自主决策）+ long-horizon（跨小时调试）+ 多 sub-agent（planner / executor / debugger））。

### 1.3　two-thread 心法（每道面试题先 disambiguate）

面试官问 "你怎么理解 multi-agent system？" 时，**第一句话先 disambiguate**：

> "Multi-agent 字面上是多个 LLM 协作，但语义上分两条线：
> （1）**协作型**（CAMEL / AutoGen / MetaGPT），多 agent 不同身份共同完成一个目标；
> （2）**对抗型 / 共识型**（debate），多 agent 独立解答再 reconcile。
> 工程实践里前者更常见，理论分析里后者更多。"

这个 disambiguate 在 RLHF / Diffusion / RAG 等其他主题里也通用——**用 30 秒 framing 拿到 reviewer 信任**，比直接背 paper 名字加分。

## §2 multi-agent 协作核心范式

### 2.1　CAMEL：role-play inception prompting

CAMEL (Communicative Agents for "Mind" Exploration of Large scale Language model society, Li et al., NeurIPS 2023, arXiv 2303.17760) 是**第一个被广泛引用的 LLM multi-agent paper**（不是第一个 multi-agent 想法，但第一个把 prompt 工程做到 systematic）。

核心机制：**两个 frozen LLM**，一个扮演 user（task initiator），一个扮演 assistant（task solver），通过 **inception prompt** 锁定角色不漂移：

```
[user prompt 模板]
你是 <ROLE_USER>，你正在和 <ROLE_ASSISTANT> 合作完成 <TASK>。
绝不直接给答案，只给 instruction，等 assistant 回答后再给下一步 instruction。

[assistant prompt 模板]
你是 <ROLE_ASSISTANT>，按 <ROLE_USER> 的 instruction 执行。
绝不主动提问或追加 task。每次回答后说 "Next request."
```

**关键贡献**：发现了 **role flipping**（assistant 不知不觉变 user）和 **task drift**（任务范围越聊越大）两个失败模式，并通过 prompt 工程缓解。

> ⚠️ **CAMEL 不是 cooperative game** — 表面上是 user-assistant 对话，本质上 user 和 assistant 都受同一个 task 锁定，**没有 reward 冲突**——不是 multi-agent RL 意义上的 game theory。常见误用是把 CAMEL 和 self-play RL（如 AlphaGo）混为一谈。

### 2.2　AutoGen：GroupChat / 通用 multi-agent 框架

AutoGen (Wu et al., arXiv 2308.08155, 2023-08, Microsoft Research; later published at **COLM 2024** + presented at **ICLR 2024 LLM Agents Workshop**) 把 multi-agent 抽象成 **conversable agent**——每个 agent 是一个对象，有 `send / receive / generate_reply` 三个方法。`GroupChat` 是其中最有影响力的模式：N 个 agent 在一个共享的 message queue 上轮流发言，一个 `GroupChatManager` 决定下一个发言的人。

伪代码（concept）：

```python
class GroupChat:
    """ N 个 agent 共享 message history，按 selector 策略轮流发言 """
    def __init__(self, agents, max_round=10):
        self.agents = agents
        self.messages = []
        self.max_round = max_round

    def select_speaker(self, last_speaker):
        # 三种策略: round_robin / random / llm_selector (LLM 自己决定下一个)
        # 实际产品常用 llm_selector: 给 manager LLM 看 history + 角色描述, 选下一个
        ...

    def run(self, init_msg):
        self.messages.append(init_msg)
        speaker = self.agents[0]
        for _ in range(self.max_round):
            reply = speaker.generate_reply(self.messages)
            self.messages.append(reply)
            if self._is_terminated(reply):
                break
            speaker = self.select_speaker(speaker)
        return self.messages
```

> 💡 **selector 设计是 AutoGen 的成败点** — round-robin 简单但容易让弱 agent 拖后腿；random 缺乏控制；**llm_selector** 让 manager LLM 看 history 选下一个发言，是 production 默认（也是 cost 高的根源——多了一次 manager call）。

### 2.3　MetaGPT：SOP-driven 软件公司模拟

MetaGPT (Hong et al., ICLR 2024 (oral), arXiv 2308.00352) 是把 multi-agent 推向**结构化 workflow** 的代表。核心 insight：通用 GroupChat 容易 chaos——让 PM / Architect / Engineer 自由对话，往往话题跑偏；不如**写死 SOP**（standard operating procedure）：

```
ProductManager  → PRD (Product Requirement Doc)
Architect       → Tech Design (class diagram, API)
ProjectManager  → Task List
Engineer        → Code (multiple .py files)
QAEngineer      → Test Cases
```

每个角色**只接收上一节点的 structured output（不是自由对话）**，输出也是 structured（文档/代码/测试）。

> 💡 **structured vs unstructured 是 multi-agent 工程化的最大分水岭** — CAMEL / 早期 AutoGen 是 unstructured（自由聊天）；MetaGPT / ChatDev / 现代 production agent 都是 structured（pipeline of artifacts）。原因：unstructured 容易 hallucinate "我们已经完成了"，structured 强制每一步产出可 inspect 的 artifact。

### 2.4　ChatDev / AgentVerse / Generative Agents

- **ChatDev** (Qian et al., ACL 2024, arXiv 2307.07924)：和 MetaGPT 思路类似，软件开发 multi-agent，**chat chain** 串起 design / coding / testing 三阶段。代码量小（~7B token），但开源得早，被广泛 fork。
- **AgentVerse** (Chen et al., ICLR 2024, arXiv 2308.10848)：提出 expert recruitment + collaborative decision-making + individual action + evaluation 四阶段框架。expert 角色由 LLM **动态生成**（不是 fixed pool），适合任务多样场景。
- **Generative Agents** (Park et al., UIST 2023, arXiv 2304.03442, Stanford + Google)：25 个 agent 在 sandbox 小镇里**自由生活**，每个 agent 有 daily routine + memory stream + reflection。涌现出生日 party、约会、竞选市长等社会行为——**第一次让 LLM agent 表现出 social emergence**。

> ⚠️ **Generative Agents 的 emergence 不等于真智能** — Park 2023 强调这是 "believable behavior"（让人觉得像真人），不是有意识或有 agency。Reddit / Twitter 上常被夸大成 "AI village"，面试时要克制：**这是一个 prompt 工程极致的 case study，不是 AGI 雏形**。

### 2.5　Mixture-of-Agents (MoA) — 2024 NeurIPS 最 catchy 工作

MoA (Wang et al., ICLR 2025 Spotlight, arXiv 2406.04692, Together AI) 的核心 idea **简单到没人会信**：

```
        ┌─ proposer_1 (Qwen2-72B)
prompt ─┼─ proposer_2 (LLaMA3-70B)  ──→ aggregator (Qwen2-72B):
        ├─ proposer_3 (WizardLM)         "把这 N 个 response 综合
        └─ proposer_N (Mixtral)            成最终答案"
```

**关键观察**（Wang 2024 fig 3）：把 N 个 proposer 的 response 直接 concat 进 aggregator 的 prompt，比 best-of-N（选最高 reward 一个）效果好。原因：**多个不完美回答的互补信号**比单个最优回答的信息量大。

**MoA 在 AlpacaEval 2.0 上 LC win rate**：

| 系统 | LC Win Rate |
|---|---|
| GPT-4 Omni (May 2024) | 57.5% |
| MoA (6× open-source models) | **65.1%** |
| MoA-Lite (3× layer) | 59.3% |

> 💡 **MoA 的理论解释（自己加分）** — 这不是 ensemble averaging（不是简单 vote），而是 **LLM-as-aggregator** 在 prompt 空间做 latent reasoning：把 N 个不同 perspective 的 response 当成 "expert evidence"，让 aggregator 做一次 in-context reasoning + synthesis。本质是 **Wang et al. 2022 self-consistency 的 LLM-aggregator 推广**（self-consistency 用 majority vote，MoA 用 LLM）。

```python
def moa_inference(prompt, proposers, aggregator, num_layers=3):
    """ Wang 2024 MoA 多层版本 (paper 默认 3 层) """
    responses = [p(prompt) for p in proposers]
    for layer in range(num_layers - 1):
        # 每一层把上一层 N 个 response 喂给同一批 proposer 再生成
        aggregate_prompt = build_moa_prompt(prompt, responses)
        responses = [p(aggregate_prompt) for p in proposers]
    # 最后一层用 aggregator 而不是 proposer
    final_prompt = build_moa_prompt(prompt, responses)
    return aggregator(final_prompt)

def build_moa_prompt(query, responses):
    instructions = (
        "You have been provided with a set of responses from various open-source "
        "models to the latest user query. Your task is to synthesize these responses "
        "into a single, high-quality response. Critically evaluate the information, "
        "recognize that some may be biased or incorrect.\n\n"
    )
    refs = "\n".join(f"[Model {i}]\n{r}" for i, r in enumerate(responses))
    return f"{instructions}{refs}\n\n[User Query]\n{query}"
```

## §3 multi-agent debate：从 Society of Mind 到现代实现

### 3.1　历史脉络

- **Minsky 1986《Society of Mind》**：人类智能不是单一过程，而是多个 simple "agent" 在心智社会里**互相批评、互相补充**。这是 multi-agent debate 的**思想原型**，但 1986 没有可执行算法。
- **Du et al. ICML 2024 (arXiv 2305.14325, MIT)**："Improving Factuality and Reasoning in Language Models through Multiagent Debate"——**第一个**把 Society of Mind 用 LLM 实现：N 个 agent 各自给答案，**互看对方答案后再修订**，2-3 轮收敛。GSM8K +6-10pp。
- **Liang et al. EMNLP 2024 (arXiv 2305.19118, Tencent AI Lab)**："Encouraging Divergent Thinking in LLMs through Multi-Agent Debate"——加入 **judge** 角色，affirmative / negative agent 对抗，judge 仲裁。在翻译 / counter-intuitive 推理任务上 +5-8pp。

### 3.2　最小可运行实现

```python
import re
from collections import Counter

def extract_answer(response: str) -> str:
    """从 LLM response 提取 final answer。生产中要更鲁棒（处理 \\boxed{}、LaTeX、单位）。"""
    m = re.search(r"(?:final answer|答案)[:：]\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # fallback: 取最后一行
    return response.strip().split("\n")[-1].strip()

def majority_vote(answers: list[str]) -> str:
    """简单多数投票；平局时返回第一个 mode。"""
    if not answers:
        return ""
    return Counter(answers).most_common(1)[0][0]

def multi_agent_debate(query, agents, num_rounds=3):
    """
    Du et al. 2023/ICML 2024 风格的 debate.

    Args:
        agents: List of N callable LLMs (same or different)
        num_rounds: 通常 2-3 轮足够 (diminishing return)
    Returns:
        最终多数答案
    """
    # Round 0: 各自独立回答
    responses = [a(query) for a in agents]

    for r in range(num_rounds):
        new_responses = []
        for i, a in enumerate(agents):
            # 把 *其他* agent 的回答展示给 agent i
            others = "\n".join(
                f"Agent {j}: {responses[j]}"
                for j in range(len(agents)) if j != i
            )
            debate_prompt = (
                f"Question: {query}\n\n"
                f"Other agents have proposed:\n{others}\n\n"
                f"Your previous answer: {responses[i]}\n\n"
                f"Critically examine other answers. If they convince you, update "
                f"your answer; if they are wrong, defend yours with reasoning. "
                f"Give a final answer at the end."
            )
            new_responses.append(a(debate_prompt))
        responses = new_responses

    # 最终: majority vote over extracted answers
    return majority_vote([extract_answer(r) for r in responses])
```

### 3.3　收敛性：什么条件下 debate "收敛"？

> ⚠️ **常见错误：直接套 Banach contraction** — debate 的每个 agent update $T_i$ 把 N 个 peer answer 映到自己新答案；联合 update $T = (T_1, \dots, T_N)$ 在 "答案空间" 上通常是 **doubly-stochastic averaging** 风格（softmax + majority vote）。这类 averaging 算子在 doubly-stochastic graph 上**特征值 1 总是存在**（对应 consensus 方向），所以**严格 Banach contraction 一般不成立** —— 不能直接说 "唯一不动点 + 几何收敛"。

> ✅ **正确说法（consensus dynamics）** — 应该说收敛到 **fixed-point set**（一致 cluster），而非唯一 fixed point：

1. **Linear averaging case**：若 $T$ 是线性 doubly-stochastic operator $A$（即 $A \mathbf{1} = \mathbf{1}, A^\top \mathbf{1} = \mathbf{1}$），按 Perron-Frobenius，iterates $x_k = A^k x_0 \to \pi^\top x_0 \cdot \mathbf{1}$，即收敛到 **consensus value**（所有 agent 同答案），但**起点不同 consensus value 不同** —— fixed-point set 是 $\{c \cdot \mathbf{1} : c \in \mathbb{R}\}$
2. **非线性 case（softmax / majority vote）**：fixed-point set 通常是若干离散 consensus cluster（每个对应一个候选答案的全员同意）；从不同 initial 出发收敛到哪个 cluster 取决于初始多数派
3. **真要 Banach 不动点**：需要额外引入外部 contraction（如固定 reference answer 的 anchor agent），但这等价于退化为非 debate 算法

**实践对收敛的影响**：

- **majority pull** 强（peer 一致时容易 sway agent）→ fixed-point set 中 majority cluster 吸引盆变大
- **temperature 低**（agent 倾向 majority answer）→ 收敛快但易陷局部 consensus（即使错答案）
- **agent heterogeneous**（不同 model / prompt）→ fixed-point set 可能完全不重叠，无 consensus

**Liang 2024 的 affirmative-negative 反例**：刻意设计对抗 agent，averaging 结构被破坏 —— 严格说不是 "$\beta \ge 1$"，而是 **没有公共 fixed-point set**（两个 cluster 互推），所以必须加 judge 当**外部决策器**强行裁决。

### 3.4　N=3 vs N=5：边际收益分析

**经验观察**（Du 2023 / Liang 2024 多组实验综合）：

| N | 相对 baseline 提升（GSM8K / MMLU 类推理任务） | Cost (× LLM call) |
|---|---|---|
| 1 (no debate) | baseline | 1× |
| 3 | +5-10pp (典型) | 3 × R rounds |
| 5 | +1-2pp 在 N=3 之上 | 5 × R rounds |
| 7+ | <1pp 增量 | 7 × R rounds |

→ **N=3 到 N=5 边际收益急剧下降**，N=3, R=2-3 已经吃到大部分收益。具体数字依任务而变（math vs trivia vs translation 表现不同）。

> ⚠️ **cost-accuracy 关系不是线性** — N 增加收益递减，根本原因是 **agent 之间 correlation 上升**（同 model family 的 N 个 agent 给出高度相关的回答，independent ensemble 假设崩塌；按 Condorcet 定理收敛速率受 $\rho$ 影响，$\rho$ 大则增益小）。面试常问"为什么不 N=100"——除了 cost，是 correlation。

## §4 协调协议：A2A / 黑板架构 / hierarchical orchestrator

### 4.1　通信协议层级

| 层 | 名字 | 典型代表 |
|---|---|---|
| 应用层 | 任务语义 | "你修 bug，我写 test" |
| 协调层 | message routing | AutoGen GroupChat, GroupChatManager |
| 协议层 | message format | **A2A (Google 2025)**, **MCP (Anthropic 2024)**, OpenAI Agents SDK |
| 传输层 | RPC / WebSocket / stdio | HTTP, gRPC, stdio |

- **MCP (Model Context Protocol, Anthropic Nov 2024)**：**LLM ↔ tool** 的协议，不是 agent ↔ agent。但很多 agent framework 用 MCP 暴露 sub-agent 作为 tool（"sub-agent as tool"）。
- **A2A (Agent-to-Agent, Google April 2025)**：明确针对**多 agent 互联**，定义 agent card / task / message / artifact 四个核心对象，HTTP+JSON-RPC。截至 2026-05 仍是 spec 阶段，业界用得少。
- **OpenAI Agents SDK** (2025) 提供 handoff / guardrails / tracing 抽象，事实上是 OpenAI 私有协议。

### 4.2　黑板架构（Blackboard architecture）

经典 AI 架构（HEARSAY-II 1980，DARPA 项目），**multi-agent 共享一个全局可读写的"黑板"**，每个 agent 看黑板决定要不要发言。在 LLM 时代它的变体是：

```
┌─────────────────────────────────────────────┐
│  Shared Blackboard (vector store / KG)      │
│  - claim_1: "x > 0"                          │
│  - assumption_1: "f convex"                  │
│  - subgoal_1: "show g(x) bounded"            │
└─────────────────────────────────────────────┘
        ↑                ↑                ↑
        │                │                │
    Reasoner          Verifier         Skeptic
   (写 claim)        (验 claim)       (反 claim)
```

**优点**：解耦——agent 不需要知道别的 agent 存在，只关心黑板上的状态。**缺点**：need a control unit 决定谁发言（否则一片混乱）。AutoGen GroupChat 本质就是黑板 + LLM-as-control-unit。

### 4.3　Hierarchical orchestrator + worker（Claude Code 模式）

2024-2025 production agent 的事实标准：**1 个 orchestrator + N 个 worker**。

```

           ┌──────────────────────┐
           │   Orchestrator       │ ← 看用户 query, 决定:
           │   (main LLM, big)    │   - 拆 subtask
           │                      │   - 哪个 worker 接
           │                      │   - 怎么 aggregate result
           └──────────────────────┘
                ↓     ↓     ↓
         ┌──────┘     │     └──────┐
         ↓            ↓            ↓
    ┌────────┐  ┌────────┐  ┌────────┐
    │Worker A│  │Worker B│  │Worker C│
    │ (code) │  │ (web)  │  │ (math) │
    └────────┘  └────────┘  └────────┘
```

**Claude Code Agent / Cline / Aider** 都是这个模式：

- **Claude Code Agent**：main loop = orchestrator，TaskCreate 调出 subagent（不同 system prompt + 工具子集），完成后 subagent 把 final message 回传 orchestrator。
- **Cline**：plan mode = orchestrator 思考，act mode = worker 执行（同一 LLM 不同 prompt）。
- **Aider**：自动 split 大改成 small commit，每个 commit 是一个 sub-conversation。

> 💡 **orchestrator-worker 的成本结构** — orchestrator 用大模型（Claude 4.6 Sonnet 级别）决策，worker 用便宜模型（Haiku / Sonnet）执行——叫 **"cascading inference"**（Chen, Zaharia & Zou 2023 "FrugalGPT", arXiv 2305.05176）。1 次 orchestrator call + 10 次 worker call 比 11 次 orchestrator call 便宜 5-10×。

### 4.4　Sub-agent blame-shifting (常见 bug)

```
Orchestrator: "任务失败了。"
Worker A: "我已经写完代码了，是 Worker B 的 test 没写好。"
Worker B: "我 test 写的没问题，是 Worker A 的代码 bug。"
Orchestrator: 陷入 deadlock
```

**根因**：每个 worker 只看到自己的 conversation，没有 ground truth artifact。

**修复**：

1. **artifact-level verification**：让 worker 跑 unit test / linter，把 result 作为 objective evidence
2. **third-party judge**：单独跑一个 fresh-context judge LLM 看双方 conversation + artifact，做判断
3. **incremental commit**：每个 worker 的 output 立即落盘 + checksum，blame 时直接 diff 文件

## §5 long-horizon agent：memory architecture

### 5.1　memory 三类：sensory / short-term / long-term

借用认知心理学的 Atkinson-Shiffrin 三级模型：

| Memory | LLM 对应 | 容量 | 持续时间 |
|---|---|---|---|
| **Sensory** | 当前 input token | 几 K | 一次 forward |
| **Short-term / Working** | LLM context window | 几 K - 1M | 一次 session |
| **Long-term** | 外部 store (vector / KG / file) | unbounded | 永久 |

Long-term 又分两类（Tulving 1972）：

- **Episodic memory**：具体事件（"昨天我和 Alice 讨论 RLHF"）
- **Semantic memory**：抽象知识（"DPO 的 loss 是 ..."）

在 agent 系统里：

- episodic ≈ **trajectory log**（过去 action + observation 序列）
- semantic ≈ **knowledge base**（论文、文档、API 知识）

> 💡 **episodic 与 semantic 用不同 retrieval 策略** — episodic 按 **temporal + spatial proximity**（"过去 24h 在这个 file 改过的事"）；semantic 按 **semantic similarity**（"和当前问题最相关的知识"）。production 常用 **hybrid retrieval**：先按 time decay 过滤再按 embedding rank。

### 5.2　MemGPT：OS-style virtual memory

MemGPT (Packer et al., arXiv 2310.08560, 2023-10; engineered as **Letta** 2024+) 把 LLM context 类比 OS RAM，把外部 vector store 类比 disk：

```

  +─────────────────+         +─────────────────+
  │  Main Context   │         │  External Store │
  │  (RAM analog)   │  ←─→    │  (disk analog)  │
  │  - system info  │         │  - archival     │
  │  - recall snip  │         │    memory       │
  │  - chat history │         │  - recall mem   │
  └─────────────────┘         └─────────────────┘
        ↑   ↓
        │   │  page in / out via function call
        │   ↓
   ┌──────────────────┐
   │ LLM + functions: │
   │ - search_archi   │
   │ - insert_archi   │
   │ - search_recall  │
   │ - send_message   │
   │ - pause          │
   └──────────────────┘
```

**关键术语**：

- **Main Context** = 当前 LLM input（system prompt + working context + recall snippet + dialogue）
- **Recall Memory** = 过去对话历史（按时间存）
- **Archival Memory** = 任意 fact / document（按 semantic store）
- **Page Fault** = main context 不够，需要 retrieve archival/recall → 触发 function call

**page fault 对 latency 的影响**：

| 操作 | 延迟典型值（GPT-4o-class） |
|---|---|
| 一次 LLM forward（4K context） | 1-2 秒 |
| 一次 archival search（vector store） | 50-200 ms |
| 一次 page fault（search → LLM 再 generate） | **1-2 秒（搜索） + 1-2 秒（合成）** |
| 主上下文 hit（无 page fault） | 1-2 秒 |

→ **page fault 把单次 turn 延迟 ~翻倍**。生产系统常用 **prefetch**（预测要用什么 archival，提前 retrieve）来缓解。

### 5.3　MemoryBank：海马体启发

MemoryBank (Zhong et al., AAAI 2024, arXiv 2305.10250) 受 **Ebbinghaus forgetting curve** 启发，每条 memory 带 **strength** 和 **last_access_time**，用指数衰减：

$$S_t = S_0 \exp\left(-\frac{t - t_\text{last\_access}}{\tau}\right)$$

retrieval 时按 **similarity × current strength** 排序，**经常被访问的 memory 强度回升**（mimic 海马体 reactivation）。

> 💡 **MemoryBank vs MemGPT** — MemGPT 把 memory 当 disk（无优先级，按 query 检索），MemoryBank 给 memory 加 dynamic priority（按访问频率衰减/强化）。production agent 常**两者结合**：MemGPT 的 disk 抽象 + MemoryBank 的 forgetting curve 做 cache eviction。

### 5.4　GraphRAG：knowledge graph + community

vector RAG（最朴素 RAG）的核心问题：**multi-hop / global question 上崩**。

例：

- "总结整个 codebase 的设计哲学" → vector RAG 只能找几个 file，无法 global synthesize
- "Alice 的朋友的朋友是谁" → 三跳 query，vector retrieval 一次拿不到三跳路径

**GraphRAG** (Microsoft 2024, Edge et al., arXiv 2404.16130) 解决方案：

```
Stage 1 (offline, expensive):
  document → LLM extract (entity, relation, entity) 三元组
         → 构 KG → Leiden community detection → 多层社区摘要

Stage 2 (online, cheap):
  query → 判断 query 类型 (local entity? global theme?)
       ↓
  local: 找最相关 entity, 看邻居, 用 vector RAG 补
  global: 用社区摘要做 map-reduce
```

**Wins**：

| Benchmark | Vector RAG | GraphRAG | Δ |
|---|---|---|---|
| Multi-hop QA (HotpotQA-class) | ~50% | ~70% | +20pp |
| Global summarization | poor | strong | qualitative |

**Loss**：

- offline cost 高（一次 build KG 要扫全语料）
- KG schema 设计成败决定一切

> ⚠️ **GraphRAG 不是 silver bullet** — 单跳 factoid（"DPO 的 loss 公式"）vector RAG 已经够好；GraphRAG 只在 multi-hop / global 上有优势。**先 profile query 分布再决定要不要上 GraphRAG**——盲目堆 GraphRAG 是 2025-2026 见过最常见的 over-engineering。

### 5.5　代码：episodic + semantic 双轨 retrieval

```python
from dataclasses import dataclass
from typing import List
import time
import numpy as np

@dataclass
class MemoryItem:
    text: str
    embedding: np.ndarray
    created_at: float
    last_access: float
    access_count: int = 0
    kind: str = "semantic"   # "episodic" / "semantic"

class HybridMemory:
    """ 简化版 MemoryBank + episodic/semantic 分轨 """
    def __init__(self, decay_tau_episodic=86400.0, decay_tau_semantic=864000.0):
        self.items: List[MemoryItem] = []
        self.tau_e, self.tau_s = decay_tau_episodic, decay_tau_semantic

    def add(self, text, embedding, kind="semantic"):
        now = time.time()
        self.items.append(MemoryItem(text, embedding, now, now, 0, kind))

    def _strength(self, item: MemoryItem, now: float) -> float:
        tau = self.tau_e if item.kind == "episodic" else self.tau_s
        # access_count 增益: 每次 access 强度 +0.1 (Ebbinghaus spaced repetition)
        boost = 1.0 + 0.1 * item.access_count
        decay = np.exp(-(now - item.last_access) / tau)
        return boost * decay

    def retrieve(self, query_embedding, k=5, kinds=("episodic", "semantic")):
        now = time.time()
        scores = []
        for it in self.items:
            if it.kind not in kinds:
                continue
            # cosine similarity (假设已归一化)
            sim = float(np.dot(query_embedding, it.embedding))
            score = sim * self._strength(it, now)
            scores.append((score, it))
        scores.sort(reverse=True, key=lambda x: x[0])
        top = [it for _, it in scores[:k]]
        # 访问会强化
        for it in top:
            it.last_access = now
            it.access_count += 1
        return top
```

> 💡 **strength × similarity 的乘法 vs 加法** — 这里用乘法：strength 衰减到 0 时该 memory 完全消失（即使 similarity 高）；加法会让 strength 只是 bias。production 用乘法，但**对 episodic memory** 加 floor（避免新 episodic 立刻被淘汰）。

## §6 planning + tree search agents

### 6.1　Plan-then-execute 范式

HuggingGPT (Shen et al., NeurIPS 2023, arXiv 2303.17580, Microsoft + Zhejiang U) 是早期代表：

```
User Query
   │
   ↓
LLM (Plan)         ← 用 ChatGPT 把 task 拆成 sub-task DAG
   │
   ↓
HF Models (Execute) ← 调用 HuggingFace 上各种 vision / speech 模型
   │
   ↓
LLM (Aggregate)    ← 把各 sub-task 输出综合
```

**问题**：plan 一次定死，**执行失败无法 replan**。这就引出 ReAct / Reflexion / tree-search 系列。

### 6.2　ReAct：思考 + 行动交替

ReAct (Yao et al., ICLR 2023, arXiv 2210.03629, Princeton + Google) 形式：

```
Thought 1: I need to find Colorado mountain heights.
Action 1: search("Colorado eastern sector elevation range")
Observation 1: 1800 to 7000 ft.
Thought 2: Now I need to find the High Plains elevation.
Action 2: search("High Plains elevation")
...
```

每一步 thought 和 action 交替，**这种格式现在是所有 agent 的事实基础**——AutoGPT / LangChain Agent / Toolformer 都是 ReAct 的变种。

### 6.3　Tree of Thoughts (ToT)

ToT (Yao et al., NeurIPS 2023, arXiv 2305.10601) **第一次显式 search**：

```
                root (problem)
               /    |    \
            t1.a  t1.b  t1.c     ← LLM propose N thoughts
              |   |   |
          ┌────────────────┐
          │  LLM evaluator  │     ← 每个 thought 评分 (1-10 / "sure")
          └────────────────┘
              ↓ BFS / DFS expand
            t2.aa t2.ab ...
```

- **expand**：LLM 提议 K 个 thought
- **evaluate**：LLM 给每个 thought 打分 / 标 "sure / maybe / impossible"
- **search**：BFS（按 layer 推进 top-k）或 DFS（带 backtrack）

> ⚠️ **ToT 的 LLM-as-evaluator 是 self-referential 风险** — 同一个 LLM 既 propose 又 evaluate，可能**给自己的 thought 打高分**。production 用 ToT 时要么换 reward model 当 evaluator，要么用 cross-model evaluator。

### 6.4　RAP：MCTS + world model

RAP (Hao et al., EMNLP 2023, arXiv 2305.14992) 升级：把 ToT 的 LLM-as-evaluator 换成 **MCTS + LLM-as-world-model**。

```
MCTS step:
  1. select: 按 PUCT 公式选 leaf
  2. expand: LLM propose next action
  3. simulate: LLM rollout 到 terminal, 估 reward
  4. backup: 把 reward 沿 path 回溯更新 Q(s,a)

PUCT 公式 (UCT 的 AlphaGo 改良版):
  a* = argmax_a [ Q(s, a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a)) ]
```

公式各项：

$$\boxed{\;a^* = \arg\max_a \left[ Q(s, a) + c_\text{puct} \cdot P(s, a) \cdot \frac{\sqrt{N(s)}}{1 + N(s, a)} \right]\;}$$

- $Q(s, a)$：从 $(s,a)$ 出发 rollout 的 mean reward
- $P(s, a)$：prior probability（LLM 给的 action 概率）
- $N(s)$：状态 $s$ 的访问数
- $N(s, a)$：(s, a) 边的访问数
- $c_\text{puct}$：exploration constant（典型 1.0-3.0）

**直觉**：第一项 $Q$ exploit（选 reward 高的），第二项 explore（次数少 + prior 高的）。$\sqrt{N(s)} / (1 + N(s,a))$ 是 **UCB1 + AlphaGo prior** 混合。

### 6.5　LATS：MCTS + Reflexion + value

LATS (Zhou et al., ICML 2024, arXiv 2310.04406, Penn + UIUC) 把 RAP 进一步加 Reflexion：

```
For each expansion:
  1. action = LLM_policy(state, history)
  2. observation = env.step(action)
  3. value = LLM_value(state, history)   ← scalar 估计
  4. if failed: reflection = LLM_reflect(history)  ← 写入下次 prompt
  5. backup value 沿 path
```

LATS 在 HotpotQA / WebShop / HumanEval 上都比 ReAct + Reflexion 单独用强。

> 💡 **LATS = ToT + Reflexion + MCTS 的混合体** — 三个技术合一，工程复杂度高，但 sample efficiency 显著好。production 部署成本 + 工程难度让其至今不是主流，但**论文意义 / 思想完整性强**——面试可作为"我读过最 well-designed agent 论文之一"的例子。

### 6.6　Agent-Q：MCTS + DPO 离线训练

Agent-Q (Putta et al., 2024, arXiv 2408.07199, MultiOn + Stanford) 反思：MCTS at inference 太慢，能不能**用 MCTS 离线产生数据训 policy**？

```
Offline:
  for many episodes:
    trajectory = MCTS_search(env, llm)
    for state, good_action, bad_action in trajectory.preferences():
      DPO_dataset.add(state, good_action, bad_action)
  fine-tune LLM via DPO

Inference: 直接用 fine-tuned LLM (无 MCTS)，速度 ↑↑
```

**结果**：WebShop 任务上从 28% → 51% success rate，且 inference 速度比 in-loop MCTS 快 10×。

### 6.7　LATS 风格 tree-search 实现（核心 60 行）

```python
import math
from dataclasses import dataclass, field
from typing import Optional, List, Callable

@dataclass
class Node:
    state: str
    action: Optional[str] = None
    parent: Optional["Node"] = None
    children: List["Node"] = field(default_factory=list)
    visits: int = 0
    value_sum: float = 0.0
    prior: float = 1.0
    is_terminal: bool = False

    @property
    def Q(self):
        return self.value_sum / self.visits if self.visits > 0 else 0.0

def puct(child, parent, c=1.5):
    """ PUCT: Q + c * P * sqrt(N_parent) / (1 + N_child) """
    explore = c * child.prior * math.sqrt(parent.visits) / (1 + child.visits)
    return child.Q + explore

def select(node):
    while node.children and not node.is_terminal:
        node = max(node.children, key=lambda c: puct(c, node))
    return node

def expand(node, llm_propose, k=3):
    if node.is_terminal: return
    for action, prior in llm_propose(node.state, k=k):
        node.children.append(Node(state=None, action=action, parent=node, prior=prior))

def simulate(node, env_step, llm_propose_one, llm_value, max_depth=5):
    """ Rollout 从 (parent.state, node.action) 起跑 env 到 terminal 或深度上限.

    Args:
        llm_propose_one: policy — 给 state 返回单个 action（greedy/random rollout）
        llm_value:       value — 给 state 估剩余 cumulative reward
    """
    state, cur_action, cum_r = node.parent.state, node.action, 0.0
    for _ in range(max_depth):
        state, r, done = env_step(state, cur_action)
        cum_r += r
        if done:
            node.is_terminal, node.state = True, state
            return cum_r
        cur_action = llm_propose_one(state)   # policy rollout 选下一步 action
    node.state = state
    return cum_r + llm_value(state)            # value head 估剩余

def backup(node, value):
    while node:
        node.visits += 1
        node.value_sum += value
        node = node.parent

def lats_search(initial_state, llm_propose, llm_propose_one, llm_value, env_step,
                num_iter=50, c_puct=1.5, max_rollout=5):
    """LATS = MCTS + Reflexion + value 估计.
    
    llm_propose:       expand 时给 (state, k) → [(action, prior)]
    llm_propose_one:   rollout 中给 state → action（policy）
    llm_value:         value 估计 state → cumulative reward 剩余
    """
    root = Node(state=initial_state, visits=1)
    for _ in range(num_iter):
        leaf = select(root)
        if not leaf.is_terminal:
            expand(leaf, llm_propose, k=3)
            if leaf.children:
                leaf = leaf.children[0]
        value = simulate(leaf, env_step, llm_propose_one, llm_value, max_depth=max_rollout)
        backup(leaf, value)
    return max(root.children, key=lambda c: c.visits).action if root.children else None
```

> ⚠️ **LATS 实现陷阱** — (1) `simulate` 必须从 parent + action 推（不复用 select leaf 的 state）；(2) **policy（`llm_propose_one`）和 value（`llm_value`）必须分开传**——之前版本误把 value 当 policy 用，rollout 行为退化为只看 state value 不选 action；(3) `expand` 时 prior 用 `softmax(LLM_logp)` 归一化；(4) **terminal 判定要严格**，否则 rollout 无限循环；(5) PUCT 系数 c 典型 1.0-3.0；(6) 生产里加 wall-time budget 检查，避免 50-iter 跑超时。

## §7 long-horizon evaluation benchmarks

### 7.1　Benchmark 对比表（2024-2026）

| Benchmark | 论文 | 任务类型 | 步长 | 2026-05 SOTA |
|---|---|---|---|---|
| **AgentBench** | Liu et al., ICLR 2024, arXiv 2308.03688 | 8 envs (OS, DB, WebShop, HouseHold...) | 10-50 | GPT-4 ≈ 4.0/10 overall |
| **τ-bench (TAU-bench)** | Yao et al., 2024, arXiv 2406.12045 | customer service multi-turn (retail, airline) | 20-50 | Claude 3.5 Sonnet ≈ 45-55% (retail) |
| **OSWorld** | Xie et al., NeurIPS 2024, arXiv 2404.07972 | real OS GUI (Linux/macOS/Win, ≈ 369 任务) | 5-100 | Anthropic Claude (2026-05) ≈ 60% (subset) |
| **WebArena** | Zhou et al., ICLR 2024, arXiv 2307.13854 | self-hosted web (Reddit-clone, Gitea, etc.) | 5-30 | GPT-4 ≈ 14.4% (2024)，2026 头部 ~ 50% |
| **VisualWebArena** | Koh et al., ACL 2024, arXiv 2401.13649 | WebArena + 视觉理解 | 10-30 | GPT-4V ≈ 16.4% |
| **SWE-bench / Verified** | Jimenez et al., ICLR 2024, arXiv 2310.06770 | 2294 真实 GitHub issue (Python repos) | 多文件多 commit | Claude 4.6 Sonnet (2026-05) ~ 75% (Verified) |
| **MLE-bench** | Chan et al., 2024 (OpenAI), arXiv 2410.07095 | 75 Kaggle ML 比赛任务 | 24h compute budget | o1-preview + AIDE ≈ 16.9% medals |
| **SWE-Lancer** | OpenAI, 2025, arXiv 2502.12115 | 1488 真实 Upwork freelance task ($1M+ payout) | 持续 hours-days | GPT-4o ≈ 8% (managerial), 26% (IC) |
| **Adventure / TextWorld** | Yuan et al., AAAI 2019, arXiv 1806.11532 | text adventure game | 50-500 | RL-trained baseline + LLM > 80% on Coin Collector |

### 7.2　benchmark 选择决策表

| 想测什么 | 选哪个 |
|---|---|
| 通用 agent 能力 | AgentBench |
| 客服对话 + 工具调用准确率 | TAU-bench |
| 真实 GUI 操作（最难） | OSWorld |
| 编程 agent + 真实 repo 修复 | SWE-bench Verified |
| 长达数天的 ML 工作 | MLE-bench |
| 经济价值实测 | SWE-Lancer |
| Web navigation | WebArena / VisualWebArena |

### 7.3　benchmark 的 horizon 长度对失败模式的暗示

- **短 horizon (5-20 step)**：失败主要是 single-step error（误解 query / 调错 tool）
- **中 horizon (20-100 step)**：开始出现 context overflow / lost-in-the-middle / replan 失败
- **长 horizon (100+ step, multi-day)**：memory stale, sub-agent blame, cost explosion, decision paralysis

**长 horizon 上每多 10 step 成功率约衰减 1/2**（Anthropic 2025 内部观察，公开 blog 部分提及，不是严格 scaling law）——这是为什么 long-horizon agent 是 2026 最难也最有研究价值的方向。

## §8 long-horizon 特有失败模式

### 8.1　Context overflow + lost-in-the-middle

Liu et al. TACL 2024 (arXiv 2307.03172) 发现：**LLM 对 context 中间位置的信息利用率显著低于开头和结尾**——U-shape curve（开头>结尾>中间）。

```

Recall vs position in 25-doc QA (Liu 2024 fig 2):

  recall %
  100 ┤●                                 ●
      │  ●●●                          ●●●
   75 ┤      ●●●                  ●●●
      │          ●●●          ●●●
   50 ┤              ●●●  ●●●
      │                  ●●
      │
    0 └─────────────────────────────→ doc position
       1    5    10    15    20    25
```

**对 long-horizon agent 的暗示**：

1. **重要 fact 别放中间**——把 task instruction 放头部 + 当前 working memory 放尾部
2. **context window 不是越长越好**——超过 ~30K 时 utilization 显著下降
3. **periodic summarization**——周期把 context 中段总结成短 snippet 推到尾部

### 8.2　Error compound / drift

长任务里**单步错误率虽小，多步后必崩**。简化模型：

设单步 success rate $p$，独立假设下 $T$ step 总成功率：

$$P(\text{all success}) = p^T$$

$p=0.95$, $T=20$: $P = 0.358$
$p=0.95$, $T=50$: $P = 0.077$
$p=0.99$, $T=50$: $P = 0.605$

→ **single-step 95% 的 agent 在 50 step 任务上只有 7.7% 成功率**——必须把 single-step 提到 99%+ 才能撑住 50 step。

**缓解方法**：

- **rollback / replan**：每隔 N step checkpoint，失败时回滚（类似 video gen 的 keyframe）
- **hierarchical sub-task**：把 50 step 拆 5 个 10-step sub-task，每个 sub-task 内部独立失败可单独 retry
- **self-verify after each critical step**：每个关键 action 后让 verifier 检查

> 💡 **drift vs catastrophic failure** — drift 是渐进偏离目标（每步只错一点），catastrophic 是某一步彻底跑偏。**drift 比 catastrophic 更难发现**——agent 可能继续 happily 执行错误轨迹。production 防 drift 靠 periodic self-check（每 5 step 复核 task description vs current state）。

### 8.3　Decision paralysis / loop

agent 卡在某个状态反复执行相同 action：

```
Action: search("foo")
Observation: no results
Thought: I should search with different keywords
Action: search("foo")   ← 又来！
Observation: no results
...
```

**根因**：LLM 在该 prompt context 下被 greedy decode 到同一个 mode。

**修复**：

1. **action history hash**：检测最近 N step 是否重复（fingerprint），重复则强制 explore
2. **temperature ramp**：连续失败 K 次后调高 temperature
3. **explicit "give up" tool**：让 agent 知道 "I cannot solve this" 是合法 action（避免假装能解）

### 8.4　Stale memory

agent retrieve 一条**过时** memory，并基于此做决策。例：

```
Memory: "API endpoint = https://api.foo.com/v1/users"
Reality: 该 endpoint 已迁移到 /v2/users (memory 是 6 个月前的)
Agent: 调用旧 endpoint → 404 → 困惑
```

**比 memory 缺失更危险**：缺失会让 agent 主动 search 新信息，stale memory 让 agent **confidently wrong**。

**缓解**：

1. **TTL on memory**：每条 memory 带过期时间戳，过期后降权或删除
2. **memory consistency check**：weekly 跑一遍 memory vs ground truth 对比
3. **prefer recent over similar**：retrieval 公式从 `sim` 改成 `sim * exp(-age/τ)`

### 8.5　Sub-agent conflict / blame-shifting

见 §4.4。

### 8.6　Cost explosion

long-horizon agent 的 token consumption **不是线性**：

- 每 turn 都把 full history 喂进 LLM（KV cache 也救不了 prompt token 计费）
- $T$ turn 的累计 cost $\sim \sum_{t=1}^T O(t \cdot c) = O(T^2 c)$，其中 $c$ 是 per-turn 新增 token

实测：50-turn agent task 比 5-turn task token 消耗 **~ 50-200×**（不是 10×）。

> ⚠️ **cost explosion 是 production agent 头号杀手** — Devin / Claude Code 等都用 **history compaction**：把过去 30 turn 的 raw conversation 压缩成 ~ 1K token 的 summary。Anthropic 在 2025-05 加的 `/compact` 命令就是这个。

## §9 self-improvement + verification loops

### 9.1　Reflexion：从 trajectory log 反思

Reflexion (Shinn et al., NeurIPS 2023, arXiv 2303.11366) 算法：

```
For each episode:
  1. trajectory = run agent on task
  2. reward = evaluator(trajectory)
  3. if reward low:
       reflection = LLM(trajectory, reward) 
                    --> 自然语言总结失败原因
       memory.add(reflection)
  4. Next episode: prompt includes memory
```

**自然语言 reflection 比 scalar reward 信息密度高**——agent 不仅知道"失败"，还知道"为什么失败"。

### 9.2　test-time self-improvement (frame + video 双级)

A²RD 风格的 **hierarchical test-time self-improvement (HITS)** (Liu et al. 2026)：

- **Frame 级**：每生成一个 segment 后，用 frame-level verifier 检查（是否人脸变形、是否光照不一致）
- **Video 级**：每 K 个 segment 后做 video-level check（是否长程一致 / character 是否漂移）

**类比 long-horizon agent**：

- **Step 级**：每个 action 后用 unit-test / linter 检查
- **Sub-task 级**：每完成一个 sub-task 后做 integration check
- **Task 级**：最终用 acceptance test 验收

> 💡 **三级 self-correction 是 agent 工程的事实最佳实践** — Cursor agent / Claude Code 都是这个三级模式：immediate（lint），中间（partial test），最终（full test + user review）。

### 9.3　Cross-Time Replay (避免 over-specialization)

Ctx2Skill 风格的 **Cross-Time Replay**：

- 维护一个 **hard probe set**（历史上失败过的 hard cases）和 **easy probe set**
- 每次 agent skill update 后，在两个 set 上 re-evaluate
- 选 $\rho_\text{hard} \times \rho_\text{easy}$ 最大的版本（不是只看当前任务变好）

防止 agent 在新任务上"学会作弊"导致旧能力退化（catastrophic forgetting at the skill level）。

### 9.4　Dependency DAG for memory retrieval

A²RD 风格的 MVMem 用 **dependency DAG** 决定生成某 segment 时去 retrieve 哪些过去 segment：

```
seg_1 -- spatial overlap --> seg_5
seg_2 -- char A appearance --> seg_7
seg_3 -- camera continuation --> seg_4
```

→ 生成 seg_5 时只 retrieve {seg_1}，而不是 {seg_1..seg_4} 全部塞进 prompt。

**对 long-horizon agent 的意义**：把 episodic memory 之间的依赖**显式建图**（不是单一向 list），retrieval 时按图路径而不是全量 similarity。production 还在探索阶段（GraphRAG 是早期尝试）。

## §10 工程实践：subagent orchestration + budget tracking

```python
from dataclasses import dataclass
from typing import Callable, Dict
import time

@dataclass
class Budget:
    max_tokens: int = 200_000
    max_wall_seconds: float = 600.0
    max_subagent_calls: int = 20
    used_tokens: int = 0
    used_seconds: float = 0.0
    used_subagent_calls: int = 0

    def remaining(self):
        return dict(tokens=self.max_tokens - self.used_tokens,
                    seconds=self.max_wall_seconds - self.used_seconds,
                    subagents=self.max_subagent_calls - self.used_subagent_calls)

    def can_afford(self, est_tokens: int) -> bool:
        return (self.used_tokens + est_tokens <= self.max_tokens
                and self.used_subagent_calls + 1 <= self.max_subagent_calls)

class Orchestrator:
    """ Claude Code 风格的 orchestrator + N worker（带 budget tracking） """
    def __init__(self, llm_orch: Callable, llm_workers: Dict[str, Callable], budget: Budget):
        self.llm_orch, self.workers, self.budget, self.history = llm_orch, llm_workers, budget, []

    def call_subagent(self, name: str, prompt: str, est_tokens: int = 4000):
        if not self.budget.can_afford(est_tokens):
            return None
        t0 = time.time()
        out, used = self.workers[name](prompt)   # returns (output, tokens_used)
        self.budget.used_tokens += used
        self.budget.used_seconds += time.time() - t0
        self.budget.used_subagent_calls += 1
        return out

    def run(self, query: str, max_steps: int = 20):
        for step in range(max_steps):
            decision = self.llm_orch(query=query, history=self.history,
                                     budget=self.budget.remaining())
            if decision["action"] == "finalize":
                return decision["answer"]
            out = self.call_subagent(decision["subagent"], decision["prompt"],
                                     decision.get("est_tokens", 4000))
            if out is None:   # 预算不够 → forced finalize
                return self.llm_orch(query=query, history=self.history,
                                     budget=self.budget.remaining(),
                                     hint="budget_exhausted")["answer"]
            self.history.append({"step": step, "subagent": decision["subagent"], "result": out})
        # 步数耗尽: forced finalize
        return self.llm_orch(query=query, history=self.history,
                             budget=self.budget.remaining(),
                             hint="forced_finalize")["answer"]
```

> 💡 **budget tracking 是 production agent 的硬要求** — 没有 budget 的 agent 会**烧钱无上限**。Anthropic Claude Code / Cursor / OpenAI Agents SDK 都在 SDK 层加 budget hook——不只是 token，还有 wall time、sub-agent 调用次数、API rate-limit。生产环境**必须三轨同时控**。

## §11 复杂度分析

### 11.1　multi-agent 协作 cost 模型

设：

- $N$ = agent 数
- $R$ = debate / collaboration 轮数
- $C$ = 单个 LLM call 平均 token cost
- $L$ = 累计 message history 长度

**round-robin GroupChat**：
$$\text{cost} = N \times R \times C \times O(L)$$
其中 $L$ 单调增长（每个 agent 写一条 message，所有 agent 都看），所以总 cost $\approx O(N^2 R^2 C)$ 量级（前提 message 长度 ~ const, 即 $L \propto NR$）——**N 和 R 都是 quadratic**。

**MoA**（N 个 proposer + 1 个 aggregator）：
$$\text{cost} = (N + 1) \times C \times O(L_\text{prompt}) + 1 \times C \times O(NL_\text{response})$$
线性 in $N$，**实际上更便宜**。

**debate (Du 2023)**：
$$\text{cost} = N \times R \times C \times O(NL_\text{response})$$
（每个 agent 看其他 $N-1$ 个 response）→ $O(N^2 R C L_\text{response})$。

**对比**：

| 模式 | Cost 量级 |
|---|---|
| Single LLM | $O(C L)$ |
| MoA | $O(N C L)$ |
| Debate (Du) | $O(N^2 R C L)$ |
| GroupChat (round-robin) | $O(N R^2 C L)$ 或更糟 |

→ **GroupChat 是最贵的**（这就是为什么 production AutoGen 几乎都改用 hierarchical orchestrator）。

### 11.2　long-horizon agent memory cost

设 $T$ = total turn 数, $M$ = total memory size。

- **No retrieval**（all in context）: cost $\sim O(T^2)$（每 turn 看全 history）
- **Vector RAG**：每 turn retrieve $k$ 条 memory，cost $\sim O(T \cdot (k + \log M))$
- **GraphRAG**：offline build $O(M^2 / \text{community size})$, online query $O(T \cdot \log M)$ + occasional global synthesis
- **MemGPT**：cost $\sim O(T \cdot (k + L_\text{archival\_search}))$，**page fault 增加 ~ 2× per-turn 延迟**

### 11.3　PUCT-based tree search

LATS / Agent-Q 类的 tree search at inference：

- 单次 search: $O(I \cdot D \cdot C)$，其中 $I$ = MCTS iteration, $D$ = max depth, $C$ = per-LLM-call cost
- $I = 50, D = 5, C = $ 4K token → 单 query ~ 1M token

→ **tree search at inference 很贵**——这就是 Agent-Q 的核心动机（offline MCTS 训完 inference 直接用 fine-tuned policy）。

## §12 与相关方法对比

### 12.1　multi-agent vs single-agent + reflection

| 维度 | Multi-Agent (debate, MoA) | Single + Reflection (Reflexion, Self-Consistency) |
|---|---|---|
| 多样性 来源 | 多个 model (heterogeneous) | 同 model 不同 sampling |
| Cost | $O(NRC)$ 起 | $O(NC)$ |
| 失败模式 | sub-agent conflict, blame-shifting | echo-chamber (同一 bias 重复 N 次) |
| 何时选 | model 之间真的有不同 capability 互补 | 主要靠 sampling diversity |

**经验**：若 N 个 agent 是**同 base model 同 prompt**，multi-agent 与 self-consistency 收益几乎等同（同源 bias），**没必要上 multi-agent**。

### 12.2　Tree search vs long CoT (o1-style)

| 维度 | Tree search (ToT/LATS) | Long CoT (o1, R1) |
|---|---|---|
| 哪里 search | 外部显式 tree | LLM 内部 hidden CoT |
| 可观测性 | 高（tree 可视化） | 低（hidden trace） |
| Inference 延迟 | 高（多次 LLM call） | 中（单次 long generation） |
| 训练成本 | 几乎无（inference-only） | 高（RL on reasoning） |
| 何时选 | 需可解释 / 没法训 | 有足够 compute 训 |

### 12.3　memory architecture 选型

| 场景 | 推荐 |
|---|---|
| 小项目 / prototype | LangChain ConversationBuffer (in-memory list) |
| 中等规模 / 单用户 long session | MemGPT-style RAM/disk + recall |
| 多用户产品 + 知识库 | vector store (Chroma / Pinecone) + recall |
| 复杂 multi-hop QA / 文档密集 | GraphRAG |
| Agent 需要"忘记" | MemoryBank (Ebbinghaus decay) |
| 视频/科研等强结构化 | typed memory + dependency DAG (A²RD 风格) |

## §13 2025-2026 前沿系统

### 13.1　Anthropic Claude Code Agent / Computer Use

- **Claude Computer Use** (Oct 2024)：让 Claude 直接操作 desktop（截屏 → reason → mouse/keyboard action）。OSWorld 上 ~ 14.9% (2024)，到 2026-05 已 ~ 60% (Anthropic blog)。
- **Claude Code Agent** (Feb-2025 GA)：orchestrator + subagent 模式，subagent 可用 TaskCreate 调出（不同 system prompt + tool 子集）。CLI tool 形式开放给开发者。

### 13.2　OpenAI Operator / Agents SDK

- **Operator** (Jan 2025)：Anthropic Computer Use 的对标产品，但用 **CUA (Computer-Using Agent)** 训练专门 vision policy（不是通用 Claude），WebArena 准确率 ≈ 58.1%。
- **OpenAI Agents SDK** (Mar 2025)：Python framework，提供 handoff / guardrail / tracing 抽象。**handoff** = orchestrator 把任务"递交"给另一个 agent（不只是 tool call）。

### 13.3　Cognition Devin / SWE-Agent

- **Devin** (Cognition, Mar 2024 demo)：第一个商业化"AI 软件工程师" demo，能 plan + code + debug + browser 全 stack。**SWE-bench（原版）~ 13.9%** (Mar 2024；注：SWE-bench *Verified* 子集 2024-08 才推出)→ ~ 50% (2025-late, Devin 3.0)。
- **SWE-Agent** (Yang et al., NeurIPS 2024, arXiv 2405.15793, Princeton)：开源 SWE-bench agent，提出 **Agent-Computer Interface (ACI)** 概念——给 LLM 设计专门 file-edit / shell / search 接口（不是直接给 bash），SWE-bench Lite 上 18.0%。

### 13.4　Cursor / Cline / Aider / Continue

- **Cursor** (Anysphere)：commercial IDE-integrated agent，plan/act mode 切换，Composer 是 multi-file edit subagent。
- **Cline** (开源, originally Claude Dev)：VS Code extension，open-source orchestrator + worker，2025 GitHub stars 增至 30K+。
- **Aider**：CLI tool，repo-map + LLM auto-commit；**auto-split large change** 到 small commit 是 long-horizon 工程典范。
- **Continue**：自托管 alternative，可接 self-hosted LLM。

> 💡 **production agent 工具链汇总（面试可背）** —
> commercial: Cursor, Devin, GitHub Copilot Agent, Claude Code
> open-source CLI: Cline, Aider, Continue
> framework: AutoGen (MS), CrewAI, LangGraph (LangChain), LlamaIndex AgentWorkflow

## §14 25 高频面试题 — L1 必会 / L2 进阶 / L3 顶级 lab

### Level 1 — 必会（10 题）

<details>

<summary>Q1. multi-agent / long-horizon / agentic 三个词的区别？</summary>

- **multi-agent**：多个 LLM role-play 不同身份**协作 / 对抗** —— CAMEL, AutoGen, MetaGPT, debate
- **long-horizon**：同一 agent 跨**多 turn / 多天**保持目标 —— MemGPT, Voyager, OSWorld, SWE-Lancer
- **agentic**：LLM 具备**自主 plan + tool use + reflection** 能力（agent 数可以 ≥ 1） —— ReAct, Toolformer, AutoGPT

→ 真实系统通常三者都涉及，但**精确含义不同**，面试第一步先 disambiguate。

</details>

<details>

<summary>Q2. CAMEL 的 inception prompt 解决什么问题？</summary>

解决 **role flipping** 和 **task drift**：

- **role flipping**：assistant 不知不觉变 user（"我也想知道这个问题"）
- **task drift**：任务越聊越宽（"我们顺便来讨论一下数据库设计"）

inception prompt = **强约束的 system prompt**（"你只能给 instruction，等回答后再下一条" / "你只能 reply 不能主动追加 task"），加上 "Next request." 等格式锁。

</details>

<details>

<summary>Q3. AutoGen 的 GroupChat selector 有几种策略？production 怎么选？</summary>

三种：

1. **round_robin**：按顺序轮 —— 简单，但弱 agent 拖后腿
2. **random**：随机选 —— 缺乏控制
3. **llm_selector**：用一个 manager LLM 看 history 决定下一个 —— production 默认，但每轮多一次 LLM call

实际产品（如 Microsoft 自家产品）几乎都是 llm_selector，**因为 round_robin 在 N>3 时容易 chaos**。

</details>

<details>

<summary>Q4. MetaGPT 为什么强调 SOP？和 CAMEL 的本质差别？</summary>

CAMEL 是 **unstructured chat**：role-play + 自由对话 → 容易 hallucinate "我们已经完成了"，没有可 inspect 的 artifact。

MetaGPT 强制 **structured artifact pipeline**：

```
PM → PRD (markdown)
Architect → tech design (class diagram, API)
ProjectManager → task list (JSON)
Engineer → code (.py files)
QA → test cases
```

每个角色**只接收上一节点的 structured output**（不是自由 chat），强制可 inspect。

→ **structured vs unstructured 是 multi-agent 工程化的最大分水岭**。

</details>

<details>

<summary>Q5. ReAct = ?</summary>

**ReAct = Reasoning + Acting** (Yao et al., ICLR 2023, arXiv 2210.03629)：

```
Thought 1: I need to find X.
Action 1: tool_call("search X")
Observation 1: ...
Thought 2: With X, I can conclude...
Action 2: ...
```

**thought 和 action 交替**，目前是所有 agent framework 的事实标准（AutoGPT / LangChain / Toolformer 等都是 ReAct 的工程化）。

</details>

<details>

<summary>Q6. MoA 是什么？为什么用 N 个 proposer + 1 个 aggregator 而不是 best-of-N？</summary>

**MoA (Mixture-of-Agents, Wang et al. ICLR 2025 Spotlight, arXiv 2406.04692)**：N 个 LLM 各自给 response，把 N 个 response **concat 进 prompt**喂给 aggregator LLM 让它 synthesize。

为什么比 best-of-N 好：

- best-of-N：用 reward model 选**一个最高分**的 response —— 丢弃了其他 N-1 个的**互补信息**
- MoA：让 aggregator LLM 看到所有 N 个 perspective，在 prompt 空间做 **latent reasoning + synthesis**

**结果**：6× open-source model 的 MoA 在 AlpacaEval 2.0 LC win rate 65.1% > GPT-4 Omni 57.5%。

</details>

<details>

<summary>Q7. multi-agent debate 是怎么收敛的？</summary>

**Du et al. ICML 2024 / Liang et al. EMNLP 2024 风格**：

```
Round 0: 各 agent 独立答
Round 1: 各 agent 看其他 N-1 个答案 → 修订
Round 2: 再修订
...
最终: majority vote
```

**收敛性质**（细节见 §3.3）：averaging 算子（doubly-stochastic）通常**没有 Banach 唯一不动点**——consensus 方向特征值 = 1。正确说法是收敛到 **fixed-point set**（多个 consensus cluster），实际落到哪个由初始多数派 + agent 偏置决定。

**实践**：N=3 round=3 已吃到 ~ 80% 收益；N=5 仅 +1.2pp 但 cost +67%（Du 2023 fig 4）。

</details>

<details>

<summary>Q8. MemGPT 的 RAM 和 disk 对应什么？page fault 是什么？</summary>

- **RAM** = LLM main context（system prompt + working context + recall snippet + dialogue），容量 ~ 几 K - 1M token
- **Disk** = 外部 storage（recall memory 按时间 + archival memory 按 semantic）
- **Page fault** = main context 信息不够，模型 function-call `search_archival` 或 `search_recall` → 检索 → 把结果塞回 main context

**Page fault 对 latency**：一次 page fault ~ 一次额外 LLM call 数百 ms ~ 2 秒，把单 turn 延迟翻倍。production 用 **prefetch** 缓解。

</details>

<details>

<summary>Q9. Vector RAG 和 GraphRAG 的区别？什么场景 GraphRAG 才值？</summary>

| | Vector RAG | GraphRAG |
|---|---|---|
| 表示 | embedding chunks | entity-relation KG + community |
| 索引 | similarity search | graph traversal + community summary |
| Multi-hop | 弱 | 强 |
| Global query | 不能 | 行 |
| Offline cost | 低 | 高（要 LLM 抽 entity） |
| 单 hop factoid | OK | overkill |

→ **多跳 + global theme query 选 GraphRAG**；单跳 factoid 用 vector RAG 够了。盲目堆 GraphRAG 是 2025-2026 常见 over-engineering。

</details>

<details>

<summary>Q10. ToT、RAP、LATS、Agent-Q 的关系？</summary>

- **ToT** (Yao 2023, NeurIPS) = LLM-propose + LLM-evaluate + BFS/DFS
- **RAP** (Hao 2023, EMNLP) = ToT 升级到 MCTS + LLM-as-world-model（rollout）
- **LATS** (Zhou 2024, ICML) = RAP + Reflexion（自然语言反思） + value estimate
- **Agent-Q** (Putta 2024, arXiv 2408.07199) = MCTS at training-time + DPO 训 policy → inference 不再需要 MCTS（10× 速度）

**共性**：都用 **PUCT 公式** $a^* = \arg\max_a [Q + c P \sqrt{N} / (1 + N_a)]$ 平衡 exploit / explore。

**演化逻辑**：ToT 起步 → MCTS 显式化 → 加反思 → 训出来不用 search。

</details>

### Level 2 — 进阶（10 题）

<details>

<summary>Q11. 推 debate 收敛性的充分条件，并解释 N=3 vs N=5 的边际收益差。</summary>

**收敛性证明**：

把每个 agent 的更新看成 operator $T_i : \mathcal{A}^N \to \mathcal{A}$，把 N 个 peer 的当前答案映到自己新答案。联合 update $T = (T_1, \dots, T_N) : \mathcal{A}^N \to \mathcal{A}^N$。

若存在度量 $d$ 使 $d(T(x), T(y)) \le \beta \cdot d(x, y)$ ($\beta < 1$)，则按 **Banach 不动点定理**，迭代 $x_{k+1} = T(x_k)$ 从任意 $x_0$ 收敛到唯一不动点 $x^*$，收敛速度 $d(x_k, x^*) \le \beta^k d(x_0, x^*)$。

**$\beta < 1$ 的实现条件**：

- agent 倾向 majority pull（peer 一致时容易动摇 → 拉力强）
- temperature 适中（太高随机不收敛，太低固化）
- 同源 agent（heterogeneous 之间偏置不同会增加 contraction 难度）

**N=3 vs N=5（Du 2023 / Liang 2024 经验观察）**：

| N | 相对 baseline 提升（推理类任务，依 task 而变） | cost |
|---|---|---|
| 1 | baseline | 1× |
| 3 | +5-10pp | ~9× (N × R) |
| 5 | +1-2pp 在 N=3 之上 | ~15× |

**边际收益急剧下降**原因：

1. **agent 之间 correlation**：同模型 N agent 给出高度相关回答，ensemble gain $\sim \sigma^2 (1 + (N-1)\rho)$，$\rho$ 大则增益小
2. **diminishing return on independent voices**：投票理论的 Condorcet 定理告诉我们，多数对的概率随 N 增长但收敛速率递减

**production 选 N=3, round=2 已吃到 80% 收益**。

</details>

<details>

<summary>Q12. MoA 和 self-consistency (Wang 2022) 的区别在哪？为什么 MoA 看起来更强？</summary>

- **Self-consistency (Wang 2022)**：同模型 sample N 个 CoT，**majority vote**
- **MoA (Wang 2024)**：N 个**不同**模型 sample，**LLM aggregator** 合成

**关键差**：

1. **diversity 来源**：SC 靠 sampling temperature，MoA 靠**模型异质性**——后者更强（不同模型偏置不同 → independent error）
2. **聚合方式**：SC 是 hard vote（丢弃 reasoning trace），MoA 是 LLM-as-aggregator（保留 trace 信息，做 latent synthesis）

但 MoA 的强可能也部分来自 **aggregator 是大模型**（Qwen2-72B）——如果 aggregator 用小模型，gap 缩小。

**面试加分**：MoA 本质是 "self-consistency 把 majority vote 升级成 LLM-as-aggregator + 模型异质性"——两次升级的乘积。

</details>

<details>

<summary>Q13. lost-in-the-middle 现象是什么？对长 context agent 设计意味着什么？</summary>

**Liu et al. TACL 2024 (arXiv 2307.03172)**：把答案放在 25-doc context 不同位置，测 QA recall。发现 **U-shape**：头部和尾部 recall 高（80%+），中间 dip 到 50%。

**根因**：

- attention sink 现象（头部 token 被广泛 attended）
- recency bias（尾部 token 因 causal mask 影响后续 logit）
- 训练数据偏置（人写文章重要点在头尾）

**对长 context agent 设计的暗示**：

1. **重要 fact 别放中间**：task instruction → 头部；当前 working memory → 尾部
2. **context window 不是越长越好**：超过 30K 时 utilization 显著下降；先 retrieve 再 generate 比塞满 1M context 更稳
3. **periodic summarization**：把中间累积的 conversation 周期性压缩到尾部 snippet
4. **多 chunk re-rank** 把最相关的放尾部（不是头部）

</details>

<details>

<summary>Q14. error compound 对 long-horizon agent 的影响怎么量化？</summary>

简化模型：独立单步成功率 $p$，$T$ step 任务总成功率：

$$P(\text{all success}) = p^T$$

| $p$ | $T=10$ | $T=20$ | $T=50$ | $T=100$ |
|---|---|---|---|---|
| 0.90 | 0.349 | 0.122 | 0.005 | 2.7e-5 |
| 0.95 | 0.599 | 0.358 | 0.077 | 0.006 |
| 0.99 | 0.904 | 0.818 | 0.605 | 0.366 |
| 0.999 | 0.990 | 0.980 | 0.951 | 0.905 |

→ **single-step 99% 才能撑住 50 step**。

**实际中并非独立**——某些 step 的失败会 propagate（同 mode 错），所以 effective $p$ 更低；反之 step 间有 verify/rollback 时可"修复"，effective $p$ 提高。**真实长 horizon agent 的 effective $p$ ≈ 0.95-0.98**。

**缓解**：

1. **checkpoint / rollback**：每 5-10 step 一个 checkpoint
2. **hierarchical sub-task**：拆成短 horizon，每段独立 verify
3. **per-step verifier**：unit test / linter，把 $p$ 推高

</details>

<details>

<summary>Q15. 为什么 RAG 在 multi-hop 上崩？GraphRAG 如何修？</summary>

**vector RAG 在 multi-hop 上崩的根因**：

```
Q: "Alice 的朋友的朋友是谁?"
向量空间: "Alice", "Alice 的朋友", "Bob 的朋友" 在 embedding 上不一定接近
```

第一跳的 chunk（"Alice 朋友 = Bob"）和第二跳的 chunk（"Bob 朋友 = Carol"）在向量空间**不直接关联**，单次 retrieval 只能拿到第一跳，第二跳要 reasoning 出来再 retrieve（多步 retrieval-then-reason）—— production 实现复杂。

**GraphRAG 修法**：

```
Stage 1 (offline):
  doc → LLM extract triples (Alice, friend, Bob), (Bob, friend, Carol)
       → 构 KG → Leiden community detection → 多层社区摘要

Stage 2 (online):
  multi-hop query: Alice → friend → ?? → friend → ??
  KG traversal 直接拿到 Carol
  global query: 用社区摘要 map-reduce
```

**性价比**：multi-hop QA 上 ~ +20pp；single-hop factoid 没收益但 cost 高。

</details>

<details>

<summary>Q16. PUCT 公式各项的物理意义？c_puct 怎么调？</summary>

$$a^* = \arg\max_a \left[ Q(s, a) + c_\text{puct} \cdot P(s, a) \cdot \frac{\sqrt{N(s)}}{1 + N(s, a)} \right]$$

- $Q(s, a)$：**exploit** —— 该 action 历史平均 reward
- $P(s, a)$：**prior** —— LLM 给的 action 概率 (softmax over candidates)
- $\sqrt{N(s)}$：随父节点访问数增长 → 更激进 explore
- $1 / (1 + N(s,a))$：访问越多越不偏向 explore
- $c_\text{puct}$：**exploration 系数** —— 典型 1.0-3.0

**调 $c_\text{puct}$**：

- 任务有 noisy reward → $c_\text{puct}$ 大一点（多 explore 避免被 single noisy reward 误导）
- 任务 reward 干净 + branching factor 高 → $c_\text{puct}$ 中等
- 任务 prior 很准（LLM 提议质量高）→ $c_\text{puct}$ 小（trust prior）

AlphaZero 用 1.25-3.0，LATS/RAP 实现常用 1.0-2.0。

</details>

<details>

<summary>Q17. orchestrator + worker 模式比 GroupChat 强在哪？</summary>

| 维度 | GroupChat (flat) | Orchestrator + Worker (hierarchical) |
|---|---|---|
| Cost 量级 | $O(N R^2 C L)$ | $O(R \cdot (C_\text{orch} + N_\text{used} \cdot C_\text{worker}))$ |
| 控制流 | 隐式（selector LLM） | 显式（orchestrator 决定） |
| 失败追溯 | 难（多 agent 互相引用） | 易（orchestrator 是唯一决策者） |
| 模型分层 | 同 model | 可 cascade（orch 用大，worker 用小） |
| 适合任务 | 短任务 / 探索性 | 长任务 / 工程化 |

**经验**：production agent **几乎全是 orchestrator + worker**——Claude Code、Cursor、Devin、SWE-Agent 都是这个模式。GroupChat 主要在学术 demo 和 早期 prototype 用。

</details>

<details>

<summary>Q18. agent loop / decision paralysis 怎么检测和缓解？</summary>

**检测**：

1. **action history hash**：把最近 K step 的 (action, observation) 做 fingerprint，比较是否重复
2. **embedding-level similarity**：把每个 step 的 (thought, action) 嵌入向量，连续相似度高于阈值 → flag loop
3. **reward stagnation**：每 step reward 不变 → 可能 loop

**缓解**：

1. **强制 explore**：检测到 loop 后调高 temperature 或加 random prompt 扰动
2. **explicit "give up" action**：让 agent 知道 "I cannot solve this" 是合法 → 避免假装能解
3. **history compaction**：周期性把 history 压成 summary，强制 agent forget bad pattern
4. **outer-loop time budget**：单 step 超时 → orchestrator 强制干预

</details>

<details>

<summary>Q19. SWE-bench Verified / SWE-Lancer / MLE-bench 各测什么？怎么选？</summary>

| Benchmark | 测什么 | 任务长度 |
|---|---|---|
| **SWE-bench Verified** | 真实 GitHub Python bug 修复（人审过 500/2294 题） | 小时级 |
| **SWE-Lancer** | 真实 Upwork freelance task（含 IC + managerial） | 小时 ~ 天级 |
| **MLE-bench** | Kaggle ML 比赛（数据探索 + 模型训练） | 天级 (24h budget) |
| **OSWorld** | Real OS GUI 任务（截屏 + mouse/keyboard） | 分钟级 |
| **TAU-bench** | Customer service multi-turn | 分钟级 |

**选**：

- 想 demo coding agent → SWE-bench Verified（事实工业标准）
- 想 demo 经济价值 → SWE-Lancer（payment 信号）
- 想 demo ML R&D agent → MLE-bench
- 想 demo OS / desktop agent → OSWorld（最难）
- 想 demo 客服对话 + tool → TAU-bench

</details>

<details>

<summary>Q20. stale memory 比 missing memory 更危险？</summary>

是的：

- **missing memory**：agent retrieve 不到 → 主动 search 新信息 / 老实承认不知道
- **stale memory**：agent retrieve 到过时 fact → **confidently wrong**（不会 search verify）

**例**：

```
Memory (6 个月前): "OpenAI API endpoint = api.openai.com/v1"
Reality: 该 endpoint 已迁移到 /v2 with new auth scheme
Agent: 调用旧 endpoint → 403 forbidden → 困惑 → 反复 retry
```

**缓解**：

1. **TTL on memory**：每条 memory 带过期时间戳
2. **memory consistency check**：周期对照外部 source（如 API docs）
3. **prefer recent over similar**：retrieval score = sim × exp(-age/τ)
4. **explicit refresh signal**：如果 retrieve 出来的 fact 用了 → action 失败 → 标记该 memory stale → 降权 / 删除

</details>

### Level 3 — 顶级 lab（5 题）

<details>

<summary>Q21. 推 multi-agent debate 收敛性的充分条件（formal），并解释 Liang 2024 加 judge 的本质作用？</summary>

**Setup**：$N$ agent，每个在 round $k$ 给出 answer $x_i^{(k)} \in \mathcal{A}$。Round-update operator：

$$x_i^{(k+1)} = T_i\!\left(x_1^{(k)}, \dots, x_N^{(k)}\right)$$

Joint operator $T : \mathcal{A}^N \to \mathcal{A}^N, \; (T(x))_i = T_i(x)$。

**重要前置**：直接说 "Banach contraction → 唯一不动点" 是 **错的**。Debate 的 averaging 算子（majority + softmax）通常是 **doubly-stochastic** 形，对应 $A \mathbf{1} = \mathbf{1}$，特征值 1 总是存在 —— 严格 $\beta < 1$ 通常不成立。正确的收敛性 framework 是 **consensus dynamics**（多智能体一致性理论），结论是收敛到 **fixed-point set**（一致 cluster）而非唯一不动点。

**线性 averaging 情形（formal）**：

$$x_i^{(k+1)} = \sum_{j} A_{ij}\, x_j^{(k)},\quad A \in \mathbb{R}^{N \times N}\text{ doubly-stochastic}$$

由 Perron-Frobenius，$\lim_{k \to \infty} A^k = \mathbf{1} \pi^\top$（$\pi$ 是 stationary distribution），故 $x_i^{(k)} \to \pi^\top x^{(0)}$，**所有 agent 同意 stationary mean**。fixed-point set 是 $\{c \mathbf{1} : c \in \mathbb{R}\}$（参数化的 consensus 直线），起点不同 $c$ 不同——所以不是 Banach 唯一不动点。

**非线性 case (softmax-over-peers)**：

$$T_i(x) = \sum_{j \ne i} w_{ij}(x) \cdot x_j$$

若 $w_{ij}$ 是 majority-vote softmax（temperature $\tau$），收敛动力学非线性，但 fixed-point set 仍是若干离散 consensus cluster（每个对应一个候选答案的全员同意）；从不同 initial 收敛到不同 cluster。**真要唯一不动点需要 anchor agent**（固定 reference），但那已经不是 debate。

**Liang 2024 (affirmative vs negative) 不收敛**：

Liang 把 agent 分两组 (affirmative / negative)，affirmative 倾向 confirm 当前 majority，negative 倾向 否定 majority。两组互推 —— **fixed-point set 为空**或者 disjoint，没有公共 consensus。

加 **judge**：

$$x^{(k+1)} = J(x_\text{aff}^{(k)}, x_\text{neg}^{(k)})$$

$J$ 是 **外部 contractive operator**（不基于 affirmative/negative 内部 update，由独立 judge LLM 决策），把整个迭代映射到一个新的 contractive 系统。**Judge 引入了人工 contraction**——这是 Liang 2024 加 judge 的本质：用 design 引入 $\beta < 1$ 的外部信号。

**经验校准**：

Du 2023 实验观察到 GSM8K 类推理任务上：N=1 baseline → N=3 R=2-3 即可大幅提升 (+5-10pp)，N=5 在 N=3 之上仅边际提升。这与 $\beta \approx 0.5$ 的几何收敛速度 $\beta^3 \approx 0.125$ 匹配（即 ~12.5% 的剩余不一致后就达到饱和）。Liang 2024 的 affirmative/negative + judge 设计在翻译 / counter-intuitive 推理上有类似 +5-8pp 的提升。

</details>

<details>

<summary>Q22. MemGPT 的 RAM vs disk 类比里 page fault 对 latency 的影响怎么量化？怎么 amortize？</summary>

**单 turn 延迟模型**（无 page fault）：

$$L_\text{single} = L_\text{prefill} + L_\text{decode} \cdot T_\text{out}$$

其中 $L_\text{prefill}$ 与 prompt 长度 $L_\text{in}$ 大约成线性（attention $O(L_\text{in}^2)$ 但被 KV cache 摊平），$L_\text{decode}$ 是 per-token decoding latency。

**单 turn 延迟模型 (k 次 page fault)**：

$$L_\text{with\_pf} = L_\text{single} + k \cdot (L_\text{search} + L_\text{rerun})$$

- $L_\text{search}$ ~ 50-200 ms (vector store)
- $L_\text{rerun}$ ~ 1 个完整 LLM call（重新 prefill + decode）

典型数字（GPT-4o-class，4K context input, 500 token output）：

| 配置 | latency |
|---|---|
| 无 page fault | ~ 1.2 s |
| 1 page fault | ~ 2.5 s |
| 3 page fault | ~ 5-7 s |

**延迟 ~ 翻倍 per page fault**。

**Amortize 方法**：

1. **prefetch**：基于当前 conversation **predict 下一个 turn 需要什么 memory**，在 user 输入 / LLM 思考时**并行** retrieve
2. **batch fault**：把多个 small fault 合并成一次大 retrieval（cache locality）
3. **hierarchical paging**：常用 fact 缓存在 "L2 working memory"（不进 main context 但快速可取），冷数据在 archival
4. **speculative archival**：retrieve 时多拿 k+ 几条候选，可能用上（避免 next-turn re-fault）
5. **streaming response**：开始 search 时就开始 stream "I'm checking my notes..."，掩盖延迟
6. **session warmup**：session 开始时预先把高频 archival 拉到 working memory

**理论 amortized 复杂度**（OS page replacement 类比）：

若 working set 大小 $W$，main context 容量 $C$, $W \le C$ → 0 page fault（cache hit）。$W > C$ → fault rate $\sim (W - C) / W$ 。production 监控 fault rate < 30% 是健康。

</details>

<details>

<summary>Q23. 推导 RAG retrieval 的 similarity-based selection 概率，并解释为什么 cosine + temperature scaling 在长 retrieval 上崩？</summary>

**Setup**：query $q$, candidates $\{d_1, \dots, d_M\}$，embedding $e(\cdot)$，相似度 $s_i = e(q)^\top e(d_i) / (\|e(q)\| \cdot \|e(d_i)\|)$（cosine）。

**Softmax selection**：

$$P(d_i | q) = \frac{\exp(s_i / \tau)}{\sum_{j=1}^M \exp(s_j / \tau)}$$

- $\tau \to 0$：argmax（top-1）
- $\tau \to \infty$：uniform
- 实际 retrieval 用 top-k threshold 而非 softmax sample；但 softmax 视角对**理论分析**有用

**长 retrieval 上为什么崩**：

设 ground-truth doc $d^*$ 相似度 $s^* = e(q)^\top e(d^*)$, 其他 doc 相似度 $s_j$ i.i.d. ~ $\mathcal{N}(\mu, \sigma^2)$。

$$P(d^* = \arg\max_i s_i) = P(s^* > s_j, \forall j \ne d^*) = \prod_{j \ne d^*} P(s^* > s_j) \approx \Phi\!\left(\frac{s^* - \mu}{\sigma}\right)^{M-1}$$

当 $M$ 大：

$$\log P \approx (M - 1) \log \Phi\!\left(\frac{s^* - \mu}{\sigma}\right)$$

→ **$P$ 关于 $M$ 是指数衰减**：M 翻倍时 P 平方衰减。

**直观**：candidate pool 大 → noise distractor 多 → ground-truth 被淹没。

**实际 RAG 系统 4 个崩塌点**：

1. **distractor density** ↑：与 query 表面相关但语义无关的 doc 增多
2. **embedding rank collapse**：高维 embedding 在大语料上 collapse（Wang et al. 2022 anisotropy），所有 doc 都 ~ 等距
3. **chunk boundary 错位**：长 doc chunk 后 ground-truth 跨 chunk
4. **lexical-semantic 错位**：query 用 "RLHF" 但 doc 写 "human feedback fine-tuning"

**修法**：

- **re-ranker**（如 Cohere rerank, MS bge-reranker）：cross-encoder 重排 top-100 候选
- **hybrid search**：BM25 + vector ensemble (BM25 防 lexical 错位)
- **query rewriting**：用 LLM 把 query 改写多版本，每版本 retrieve top-k 取并集
- **chunk overlap + summary**：chunk 间留 overlap + chunk-level summary（structured retrieval）

</details>

<details>

<summary>Q24. 为什么 Agent-Q 的 DPO 训练后 inference 比 in-loop MCTS 快 10×？训练时 MCTS 数据怎么转 DPO preference？</summary>

**Inference 速度差异**：

- **In-loop MCTS**：每次 inference 跑 ~ 50 iteration × ~ 5 depth = ~ 250 LLM call，单 query ~ 1M token
- **DPO-trained policy**：单次 forward pass + greedy decode = ~ 1-3 LLM call（**100× 少**）

→ 实测 10× 是因为 trained policy 仍可能多次 sampling / re-plan，但比 MCTS 少 1-2 数量级。

**MCTS → DPO 数据生成**：

每次 MCTS rollout 产生一颗树 $\mathcal{T}$。在每个状态 $s$ 处有多个 child action $\{a_1, \dots, a_k\}$ 和各自的 visit count $N(s, a_j)$ + value $Q(s, a_j)$。

**构造 preference pair**：

对每个 state $s$：

- $a^+$ = argmax visit count (MCTS 探索后认为"好"的)
- $a^-$ = argmin visit count among visited (MCTS 探索后认为"差"的)

直接得 preference: $(s, a^+) \succ (s, a^-)$。

也可用 **value gap** 筛：只保留 $Q(s, a^+) - Q(s, a^-) > \delta$ 的对（高置信度 preference）。

**DPO loss**：

$$\mathcal{L}_\text{DPO} = -\log \sigma\!\left( \beta \log\frac{\pi_\theta(a^+ | s)}{\pi_\text{ref}(a^+ | s)} - \beta \log\frac{\pi_\theta(a^- | s)}{\pi_\text{ref}(a^- | s)} \right)$$

直接 fine-tune base LLM。

**Agent-Q 结果**（WebShop）：

| 方法 | success rate | inference cost |
|---|---|---|
| Base LLM | 28% | 1× |
| ReAct | 38% | 1× |
| In-loop MCTS | 48% | ~100× |
| Agent-Q (DPO from MCTS) | **51%** | 1× |

→ **MCTS at training, fast policy at inference** 是 2024-2026 一个关键 design pattern——类比 AlphaGo Zero 的 self-play + DPO 风格 distillation。

**面试加分**：这个 pattern 也出现在 **DeepSeek-R1 distill**（R1-Zero MCTS-like search 产数据 → 蒸馏小模型）、**rStar-Math**（Microsoft 2025 MCTS + PPM self-evolution）——共同主题是 **inference-time search 是产数据手段，不是部署目标**。

</details>

<details>

<summary>Q25. 如果让你设计下一代 long-horizon agent，应该往哪几个方向走？（open-ended 顶级 lab 面试题）</summary>

可信回答框架（不需面面俱到，挑 2-3 个深入展开）：

- **方向 1 — typed memory + dependency DAG**
  - 现状：memory 主要是 flat list（vector store）或 KG（GraphRAG）
  - 缺：**事件之间的因果 / 依赖关系**——为什么 retrieve 这条而不是那条？
  - Proposal：参考 A²RD 的 MVMem，把 episodic memory 之间建显式 dependency DAG（spatial/temporal/causal），retrieval 按图路径而不是单次 similarity
  - Expected failure：DAG 维护成本（事件多了图变密）；retrieval policy 选边算法（GNN？LLM-as-graph-traverser？）

- **方向 2 — self-evolving skill set + cross-time replay**
  - 现状：agent skill 是人工写的 prompt template（如 ReAct, ToT, Reflexion）
  - 缺：**agent 不会自动发现新 skill 也不会 deprecate 旧 skill**
  - Proposal：Ctx2Skill 风格 5 角色 self-play（challenger, reasoner, judge, proposer, generator），用 cross-time replay 防止 over-specialization
  - Expected failure：skill collapse（少数 skill 占满所有 task）；reward hacking（generator 学会写让 judge 满意但实际无用的 skill）

- **方向 3 — heterogeneous multi-agent + cross-family verification**
  - 现状：multi-agent 几乎全是同 base model
  - 缺：**没有 cross-family verification**——同 model 共享 hallucination
  - Proposal：Claude + Codex + Gemini 异质 co-evolution，跨模型 audit 防 self-confirmation bias
  - Expected failure：模型强弱 / 价格 / API 版本会混淆变量；reviewer 之间也可能共同 hallucinate

- **方向 4 — long-horizon cost model + adaptive budget**
  - 现状：token 计费线性，但长任务实际是 quadratic
  - 缺：**没有 cost-aware planner**
  - Proposal：agent 决策时把 expected cost 加进 utility（不仅 reward），动态调整 verbosity / search depth / reflection 频率
  - Expected failure：cost prediction 不准（LLM 不会准确估自己用多少 token）；过度节俭导致 quality drop

- **方向 5 — formal verification / safety guarantees**
  - 现状：reflection / debate / audit 都是 empirical
  - 缺：**没有 formal correctness guarantee**——多 agent 投票投错怎么办？
  - Proposal：借鉴 Bayesian truthful elicitation / peer prediction（如 Prelec score），给 audit 一个 incentive-compatible foundation
  - Expected failure：theoretical 桥到 production 难搭；agent 无法理解 truthful 激励

- **方向 6 — evaluation 标准**
  - 现状：benchmark fragmented（TAU / OSWorld / SWE-bench 各测一面）
  - 缺：**没有统一的 long-horizon 评测标准**
  - Proposal：定义 horizon-axis benchmark：5 / 20 / 100 / 1000 step 在同一任务族上分别测；track **per-step error compound rate**（不是 final accuracy）
  - Expected failure：1000-step task 的人工标注成本极高；replay/contamination 难控制

- **方向 7 — sim-to-real for agent**
  - 现状：训练 / 测试都在 simulator (WebArena)
  - 缺：**sim-to-real gap**——simulator 上的 agent 在真实 web 上崩
  - Proposal：把 robotics sim-to-real 工具搬到 agent（domain randomization, real-world fine-tune）
  - Expected failure：web/OS 比 robotics 更高维（视觉 + 文本 + workflow）；real-world 数据采集成本高

照搬现有方法 + 加一点 —— 不展现 research taste。**关键是展现你能列 3-5 个 concrete proposal + 每个有 expected failure mode**。

</details>

## §A 附录：核心 paper 时间线 + 一句话总结

按时间倒序（截止 2026-05）：

| 日期 | Paper | arXiv | 一句话贡献 |
|---|---|---|---|
| 2025-09 | Anthropic Claude Sonnet 4.5 + extended thinking | (no arXiv) | 长程 reasoning + budget-aware thinking |
| 2025-03 | OpenAI Agents SDK | (no arXiv) | handoff / guardrails / tracing 抽象 |
| 2025-01 | OpenAI Operator | (no arXiv) | CUA-trained vision-language agent for web |
| 2024-10 | Claude Computer Use | (no arXiv) | desktop GUI agent，OSWorld ~ 14.9% (2024) |
| 2024-08 | Agent-Q | 2408.07199 | MCTS + DPO，WebShop 28% → 51% |
| 2024-06 | MoA (Mixture of Agents) | 2406.04692 | N 个 proposer + aggregator 超 GPT-4 Omni |
| 2024-06 | TAU-bench | 2406.12045 | customer service multi-turn agent benchmark |
| 2024-05 | SWE-Agent | 2405.15793 | ACI for SWE-bench Lite 18.0% |
| 2024-04 | OSWorld | 2404.07972 | real OS GUI 369 任务 |
| 2024-04 | GraphRAG | 2404.16130 | LLM-extracted KG + community for global QA |
| 2024-01 | VisualWebArena | 2401.13649 | vision + web 任务 |
| 2023-12 | Math-Shepherd | 2312.08935 | MCTS rollout 自动标 PRM |
| 2023-10 | LATS | 2310.04406 | MCTS + Reflexion + value |
| 2023-10 | SWE-bench | 2310.06770 | 2294 真实 GitHub Python bug |
| 2023-10 | MemGPT | 2310.08560 | OS-style virtual memory for LLM |
| 2023-08 | AutoGen | 2308.08155 | conversable agent / GroupChat framework |
| 2023-08 | MetaGPT | 2308.00352 | SOP-driven 软件公司 multi-agent |
| 2023-08 | AgentBench | 2308.03688 | 8 envs 通用 agent benchmark |
| 2023-08 | AgentVerse | 2308.10848 | expert recruitment + collaborative decision |
| 2023-07 | WebArena | 2307.13854 | self-hosted real web benchmark |
| 2023-07 | ChatDev | 2307.07924 | chat chain for software dev |
| 2023-05 | RAP | 2305.14992 | MCTS + world model in LLM agent |
| 2023-05 | Tree of Thoughts | 2305.10601 | LLM propose + LLM evaluate + tree search |
| 2023-05 | Multi-Agent Debate (Du) | 2305.14325 | N agent debate → consensus |
| 2023-05 | Multi-Agent Debate (Liang) | 2305.19118 | affirmative / negative + judge |
| 2023-05 | MemoryBank | 2305.10250 | Ebbinghaus forgetting curve |
| 2023-04 | Generative Agents (Park) | 2304.03442 | 25-agent sandbox social emergence |
| 2023-03 | Reflexion | 2303.11366 | natural-language self-reflection from trajectory |
| 2023-03 | CAMEL | 2303.17760 | role-play inception prompting |
| 2023-03 | HuggingGPT | 2303.17580 | plan-then-execute with HF models |
| 2022-10 | ReAct | 2210.03629 | thought-action 交替 |

> 💡 **建议精读 5 篇** — 准备面试时间有限时，按优先级读：
>
> 1. **MemGPT (2310.08560)** —— virtual-memory 抽象，长程 memory 鼻祖
> 2. **MoA (2406.04692)** —— multi-agent 实证最强 baseline
> 3. **Multi-Agent Debate (Du, 2305.14325)** —— debate 范式起源
> 4. **LATS (2310.04406)** —— tree-search agent 工程典范
> 5. **MetaGPT (2308.00352)** —— structured multi-agent SOP

> ⚠️ **常考开放题准备** — 顶级 lab interview 经常问 open-ended 题（如 Q25），关键是展现 **research taste**：能列出 3-5 个具体方向（不是"我会做 multi-agent" 这种空话），每个方向能给一个 concrete proposal + 一个 expected failure mode。

> ✅ **2026 秋招重点 framework / 工具链汇总** — production agent 在 2026 的事实标准（面试可背）：
>
> - **commercial IDE-integrated**：Cursor (Anysphere), GitHub Copilot Agent, Continue
> - **commercial coding agent**：Devin (Cognition), Claude Code (Anthropic), OpenAI Codex CLI
> - **open-source CLI**：Cline, Aider, opencode, codex-cli
> - **multi-agent framework**：AutoGen (MS), CrewAI, LangGraph (LangChain), LlamaIndex AgentWorkflow, OpenAI Agents SDK
> - **protocol**：MCP (Anthropic, LLM↔tool), A2A (Google, agent↔agent)
> - **benchmark**：SWE-bench Verified（行业标准）+ TAU-bench + OSWorld + MLE-bench

读完 §10-§14 + 上述 5 篇 paper，multi-agent + long-horizon agent 面试题应能 80%+ 覆盖。
