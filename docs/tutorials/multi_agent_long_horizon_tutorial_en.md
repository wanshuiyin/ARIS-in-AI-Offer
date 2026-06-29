## §0 TL;DR Cheat Sheet

> 💡 **9 sentences to nail Multi-Agent & Long-Horizon Agent** — the second wave of LLM agents in 2024-2026, one page of interview essentials.

1. **Paradigm positioning**: a single LLM = "system 1, one forward pass"; an agent = "system 2, decomposing reasoning into perceive → plan → act → reflect". **multi-agent** = several LLMs role-playing distinct identities and collaborating; **long-horizon** = the same agent maintaining a goal across many turns / many days. The two axes are orthogonal but usually co-occur (e.g. ChatDev, MetaGPT).

2. **Three multi-agent archetypes**:
   - **role-play dialogue** (CAMEL, Li et al. NeurIPS 2023, arXiv 2303.17760): assistant-user inception prompting;
   - **SOP / pipeline** (MetaGPT, Hong et al. ICLR 2024, arXiv 2308.00352): decompose software development into seven fixed roles — PM / Architect / Engineer / QA, etc.;
   - **debate / aggregation** (Du et al. ICML 2024 arXiv 2305.14325 + Liang et al. EMNLP 2024 arXiv 2305.19118): N agents answer independently → cross-read → iterate over several rounds to consensus.

3. **MoA (Mixture-of-Agents, Wang et al. ICLR 2025 Spotlight, arXiv 2406.04692)**: N proposers each produce an answer; an aggregator concatenates the N answers into its prompt and synthesizes. An open-source 6.5B-70B mix **beats GPT-4 Omni** on AlpacaEval 2.0.

4. **Debate convergence — empirical observation + theoretical approximation**: treat each agent's update as a mixture operator over peer answer distributions. If the operator has the **doubly-stochastic averaging** form (majority-vote softmax + temperature β), then by consensus dynamics (linear averaging operators on a doubly-stochastic graph) the iterate converges to a **fixed-point set** (a consensus distribution or a discrete consensus cluster), **not a unique fixed point** (Banach contraction generally fails — averaging operators always have eigenvalue 1). Empirically, the GSM8K accuracy gap between N=3 and N=5 is ≈ 1-2pp (diminishing returns, see Du 2023 Fig 4 + Liang 2024 Table 3).

5. **MemGPT (Packer et al., arXiv 2310.08560, 2023-10)**: treat the LLM context as OS RAM and external storage as disk. A **page fault** = the model function-calls "search archival memory" to trigger retrieval. Latency cost: one page fault ≈ one extra LLM call (hundreds of ms to seconds), but avoids context overflow. Subsequently engineered under the name **Letta** (2024+).

6. **GraphRAG (Microsoft 2024, arXiv 2404.16130)**: vector RAG collapses on multi-hop; GraphRAG first uses an LLM to extract entity-relation triples, builds a knowledge graph + community detection, makes hierarchical summaries, and then performs retrieval-augmented generation. **Global queries** ("summarize the themes of the whole document") beat vector RAG by 70-80%.

7. **tree-search agents**:
   - **ToT (Yao et al., NeurIPS 2023, arXiv 2305.10601)** = BFS/DFS + LLM evaluator;
   - **RAP (Hao et al., EMNLP 2023, arXiv 2305.14992)** = MCTS + world-model rollout;
   - **LATS (Zhou et al., ICML 2024, arXiv 2310.04406)** = MCTS + self-reflection (Reflexion) + value estimation;
   - **Agent-Q (Putta et al., 2024, arXiv 2408.07199)** = MCTS + DPO offline policy training. Common ground: all use the **PUCT** formula $a^* = \arg\max_a Q(s,a) + c_\text{puct} P(s,a) \sqrt{N(s)} / (1+N(s,a))$ to balance exploit / explore.

8. **Three long-horizon benchmarks**: **TAU-bench** (Yao et al. 2024 arXiv 2406.12045, customer service multi-turn), **OSWorld** (Xie et al. NeurIPS 2024 arXiv 2404.07972, real OS GUI), **SWE-bench** (Jimenez et al., ICLR 2024, arXiv 2310.06770, fixing real GitHub issues). 2026-05 SOTA on SWE-bench Verified ≈ 75% (Claude 4.6 Sonnet / o3, etc.), but OSWorld still < 60%.

9. **Common pitfalls**: multi-agent is not better with more (cost ∝ N, accuracy gain ∝ log N); long-CoT context overflow cannot be solved simply by writing a longer context window (**lost-in-the-middle**, Liu et al. TACL 2024 arXiv 2307.03172); stale memory is more dangerous than missing memory (wrong retrieval produces a confidently wrong answer); sub-agent blame-shifting ("that's worker A's fault") is common in hierarchical orchestrator setups.

## §1 Intuition: from single LLM to agent system

### 1.1 Why a single LLM is not enough

Treating an LLM as a one-shot forward oracle $y = f_\theta(x)$ has three fundamental limitations:

- **Bounded state**: the context window is a hard physical limit (even 1M tokens are finite), so it cannot keep a goal in mind long-term
- **One-shot decisions**: a wrong sample is permanent — there is no "undo / replan" mechanism
- **Capability slicing**: a single prompt has to reason + act + verify, which causes **role drift** — earlier reasoning interferes with later verification

The essence of an **agent** is to externalize these three problems:

| Built into the LLM | Externalized by the agent |
|---|---|
| context window | persistent memory (vector store / KG / disk) |
| one-shot sampling | iterative loop: perceive → plan → act → reflect |
| single role | multi-agent role specialization |

> 💡 **mental model — agent = LLM × harness × memory × tools** — an agent is not an LLM; it is an LLM wrapped in an outer loop. The harness decides when the LLM is called, what to call it with, and how to feed the output back. Claude Code Agent / Cursor / Devin / SWE-Agent are all essentially **differences in harness design** — with the same LLM, a harness gap can swing task success rate by 30-50pp.

### 1.2 multi-agent vs long-horizon vs agentic

These three terms are often used interchangeably, but their **precise meanings differ**:

| Concept | Focus | Typical examples |
|---|---|---|
| **multi-agent** | multiple LLMs role-playing distinct identities to **collaborate / compete** | CAMEL, AutoGen, MetaGPT, debate |
| **long-horizon** | the same agent maintaining a goal across **many turns / long time** | MemGPT, Voyager, OSWorld, SWE-Lancer |
| **agentic** | an LLM equipped with **autonomous plan + tool use + reflection** (≥ 1 agent, ≥ 1 step) | Toolformer, ReAct, AutoGPT |

→ a real-world system usually **touches all three** (e.g. Devin = agentic (autonomous decisions) + long-horizon (debugging across hours) + multi-sub-agent (planner / executor / debugger)).

### 1.3 Two-thread framing (disambiguate every interview question first)

When the interviewer asks "how do you understand multi-agent systems?", the **first sentence should disambiguate**:

> "Multi-agent literally means multiple LLMs collaborating, but semantically it splits into two threads:
> (1) **cooperative** (CAMEL / AutoGen / MetaGPT), where agents with different roles complete a single goal together;
> (2) **adversarial / consensus-based** (debate), where agents answer independently and then reconcile.
> In engineering practice the first is more common; in theoretical analysis the second is."

This disambiguation also generalizes to RLHF / Diffusion / RAG and other topics — **a 30-second framing to win reviewer trust** scores more than reciting paper titles directly.

## §2 Core multi-agent collaboration paradigms

### 2.1 CAMEL: role-play inception prompting

CAMEL (Communicative Agents for "Mind" Exploration of Large scale Language model society, Li et al., NeurIPS 2023, arXiv 2303.17760) is **the first widely cited LLM multi-agent paper** (not the first multi-agent idea, but the first to make the prompt engineering systematic).

Core mechanism: **two frozen LLMs**, one playing the user (task initiator), one playing the assistant (task solver), with their roles locked in by an **inception prompt** to prevent drift:

```
[user prompt template]
You are <ROLE_USER>, working with <ROLE_ASSISTANT> to complete <TASK>.
Never give the answer directly; only issue an instruction, wait for the
assistant's reply, then give the next instruction.

[assistant prompt template]
You are <ROLE_ASSISTANT>, executing the instructions of <ROLE_USER>.
Never proactively ask a question or add a new task. After every reply,
say "Next request."
```

**Key contributions**: identifying two failure modes — **role flipping** (the assistant slipping into the user role) and **task drift** (task scope widening with conversation) — and mitigating them via prompt engineering.

> ⚠️ **CAMEL is not a cooperative game** — on the surface it is a user-assistant dialogue, but underneath both user and assistant are locked to the same task and **share no reward conflict** — it is not a multi-agent RL setup in the game-theoretic sense. A common misuse is to conflate CAMEL with self-play RL (such as AlphaGo).

### 2.2 AutoGen: GroupChat / a general multi-agent framework

AutoGen (Wu et al., arXiv 2308.08155, 2023-08, Microsoft Research; later published at **COLM 2024** + presented at **ICLR 2024 LLM Agents Workshop**) abstracts multi-agent into **conversable agents** — each agent is an object with three methods: `send / receive / generate_reply`. `GroupChat` is its most influential mode: N agents share a single message queue and take turns speaking, with a `GroupChatManager` deciding who speaks next.

Pseudocode (concept):

```python
class GroupChat:
    """ N agents share a message history and speak in turn per selector policy. """
    def __init__(self, agents, max_round=10):
        self.agents = agents
        self.messages = []
        self.max_round = max_round

    def select_speaker(self, last_speaker):
        # Three policies: round_robin / random / llm_selector (the LLM picks the next).
        # Production usually uses llm_selector: feed the manager LLM the history +
        # role descriptions, and let it choose the next speaker.
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

> 💡 **Selector design is the make-or-break of AutoGen** — round-robin is simple but lets weak agents drag the chat down; random lacks control; **llm_selector** has the manager LLM look at the history and pick the next speaker, which is the production default (and also the root of high cost — one extra manager call per turn).

### 2.3 MetaGPT: SOP-driven software-company simulation

MetaGPT (Hong et al., ICLR 2024 (oral), arXiv 2308.00352) is the standard-bearer of pushing multi-agent toward **structured workflow**. Core insight: free-form GroupChat easily descends into chaos — letting PM / Architect / Engineer chat freely makes the topic drift; instead, **hard-code an SOP** (standard operating procedure):

```
ProductManager  → PRD (Product Requirement Doc)
Architect       → Tech Design (class diagram, API)
ProjectManager  → Task List
Engineer        → Code (multiple .py files)
QAEngineer      → Test Cases
```

Each role **only receives the structured output of the previous node (no free-form chat)**, and its own output is also structured (a doc / code / test).

> 💡 **Structured vs unstructured is the biggest dividing line in multi-agent engineering** — CAMEL / early AutoGen are unstructured (free chat); MetaGPT / ChatDev / modern production agents are structured (pipeline of artifacts). Why: unstructured easily hallucinates "we're already done", whereas structured forces every step to produce an inspectable artifact.

### 2.4 ChatDev / AgentVerse / Generative Agents

- **ChatDev** (Qian et al., ACL 2024, arXiv 2307.07924): similar in spirit to MetaGPT, software-development multi-agent, with a **chat chain** linking design / coding / testing across three phases. The codebase is small (~7B tokens) but was open-sourced early and has been widely forked.
- **AgentVerse** (Chen et al., ICLR 2024, arXiv 2308.10848): proposes a four-stage framework — expert recruitment + collaborative decision-making + individual action + evaluation. Expert roles are **dynamically generated** by an LLM (not from a fixed pool), suiting diverse-task scenarios.
- **Generative Agents** (Park et al., UIST 2023, arXiv 2304.03442, Stanford + Google): 25 agents **live freely** in a sandbox town, each with a daily routine + memory stream + reflection. Birthday parties, dates, mayoral campaigns, and other social behaviors emerge — **the first time LLM agents have exhibited social emergence**.

> ⚠️ **Generative Agents' emergence ≠ real intelligence** — Park 2023 emphasizes this is "believable behavior" (something that feels human-like), not consciousness or agency. On Reddit / Twitter it is often hyped as an "AI village"; in interviews stay restrained: **this is a prompt-engineering case study taken to its limit, not an embryonic AGI**.

### 2.5 Mixture-of-Agents (MoA) — the catchiest work at NeurIPS 2024

The core idea of MoA (Wang et al., ICLR 2025 Spotlight, arXiv 2406.04692, Together AI) is **so simple no one would believe it**:

```
        ┌─ proposer_1 (Qwen2-72B)
prompt ─┼─ proposer_2 (LLaMA3-70B)  ──→ aggregator (Qwen2-72B):
        ├─ proposer_3 (WizardLM)         "synthesize these N responses
        └─ proposer_N (Mixtral)            into a final answer"
```

**Key observation** (Wang 2024 Fig 3): concatenating the responses of N proposers directly into the aggregator's prompt works better than best-of-N (picking the highest-reward one). Why: **the complementary signal from multiple imperfect answers carries more information than a single best answer**.

**MoA LC win rate on AlpacaEval 2.0**:

| System | LC Win Rate |
|---|---|
| GPT-4 Omni (May 2024) | 57.5% |
| MoA (6× open-source models) | **65.1%** |
| MoA-Lite (3× layer) | 59.3% |

> 💡 **A theoretical interpretation of MoA (for bonus points)** — this is not ensemble averaging (not a simple vote); it is **LLM-as-aggregator** doing latent reasoning in prompt space: treating the N differing-perspective responses as "expert evidence" and letting the aggregator perform a single round of in-context reasoning + synthesis. It is essentially **Wang et al. 2022 self-consistency promoted to use an LLM aggregator** (self-consistency uses majority vote; MoA uses an LLM).

```python
def moa_inference(prompt, proposers, aggregator, num_layers=3):
    """ Wang 2024 multi-layer MoA (paper default: 3 layers). """
    responses = [p(prompt) for p in proposers]
    for layer in range(num_layers - 1):
        # Each layer feeds the previous layer's N responses to the same
        # batch of proposers to regenerate.
        aggregate_prompt = build_moa_prompt(prompt, responses)
        responses = [p(aggregate_prompt) for p in proposers]
    # Final layer uses the aggregator instead of the proposers.
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

## §3 Multi-agent debate: from Society of Mind to modern implementations

### 3.1 Historical lineage

- **Minsky 1986 "Society of Mind"**: human intelligence is not a single process but multiple simple "agents" in a mind-society **criticizing and supplementing each other**. This is the **conceptual prototype** of multi-agent debate, but in 1986 there was no executable algorithm.
- **Du et al. ICML 2024 (arXiv 2305.14325, MIT)**: "Improving Factuality and Reasoning in Language Models through Multiagent Debate" — **the first** instantiation of Society of Mind with LLMs: N agents each give an answer, **revise after seeing each other's answers**, and converge in 2-3 rounds. +6-10pp on GSM8K.
- **Liang et al. EMNLP 2024 (arXiv 2305.19118, Tencent AI Lab)**: "Encouraging Divergent Thinking in LLMs through Multi-Agent Debate" — adds a **judge** role, with affirmative / negative agents in opposition and the judge arbitrating. +5-8pp on translation / counter-intuitive reasoning tasks.

### 3.2 Minimal runnable implementation

```python
import re
from collections import Counter

def extract_answer(response: str) -> str:
    """Extract the final answer from an LLM response. Production needs more robust parsing (\\boxed{}, LaTeX, units)."""
    m = re.search(r"(?:final answer|答案)[:：]\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # Fallback: take the last line.
    return response.strip().split("\n")[-1].strip()

def majority_vote(answers: list[str]) -> str:
    """Plain majority vote; on ties, return the first mode."""
    if not answers:
        return ""
    return Counter(answers).most_common(1)[0][0]

def multi_agent_debate(query, agents, num_rounds=3):
    """
    Du et al. 2023 / ICML 2024-style debate.

    Args:
        agents: List of N callable LLMs (same or different)
        num_rounds: 2-3 rounds usually suffice (diminishing returns)
    Returns:
        Final majority answer
    """
    # Round 0: each agent answers independently.
    responses = [a(query) for a in agents]

    for r in range(num_rounds):
        new_responses = []
        for i, a in enumerate(agents):
            # Show the *other* agents' answers to agent i.
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

    # Final: majority vote over extracted answers.
    return majority_vote([extract_answer(r) for r in responses])
```

### 3.3 Convergence: under what conditions does debate "converge"?

> ⚠️ **Common mistake: applying Banach contraction directly** — each agent's update $T_i$ maps the N peer answers to its new answer; the joint update $T = (T_1, \dots, T_N)$ on the "answer space" is typically of **doubly-stochastic averaging** style (softmax + majority vote). Such averaging operators on a doubly-stochastic graph **always have eigenvalue 1** (corresponding to the consensus direction), so **strict Banach contraction generally fails** — you cannot directly say "unique fixed point + geometric convergence".

> ✅ **Correct framing (consensus dynamics)** — say convergence to a **fixed-point set** (consensus cluster), not a unique fixed point:

1. **Linear averaging case**: if $T$ is a linear doubly-stochastic operator $A$ (i.e. $A \mathbf{1} = \mathbf{1}, A^\top \mathbf{1} = \mathbf{1}$), then by Perron-Frobenius the iterates $x_k = A^k x_0 \to \pi^\top x_0 \cdot \mathbf{1}$, i.e. they converge to a **consensus value** (all agents agree), but **different starting points yield different consensus values** — the fixed-point set is $\{c \cdot \mathbf{1} : c \in \mathbb{R}\}$
2. **Nonlinear case (softmax / majority vote)**: the fixed-point set is typically several discrete consensus clusters (each corresponding to unanimous agreement on a candidate answer); which cluster the iteration falls into depends on the initial majority
3. **For a true Banach fixed point**: you need to inject an external contraction (e.g. an anchor agent fixed to a reference answer), but this degenerates to a non-debate algorithm

**Practical implications for convergence**:

- **majority pull** is strong (an agent is easily swayed when peers agree) → the attraction basin of the majority cluster in the fixed-point set grows
- **low temperature** (agent leans to the majority answer) → fast convergence but easy to land in a local consensus (even a wrong answer)
- **heterogeneous agents** (different models / prompts) → fixed-point sets may not overlap, so no consensus emerges

**Liang 2024's affirmative-negative counterexample**: deliberately designed adversarial agents break the averaging structure — strictly speaking it is not "$\beta \ge 1$", but rather **no common fixed-point set** (two clusters push each other), which is why a judge is required as an **external arbiter**.

### 3.4 N=3 vs N=5: marginal-return analysis

**Empirical observation** (synthesizing the experiments in Du 2023 / Liang 2024):

| N | Improvement over baseline (on reasoning tasks like GSM8K / MMLU) | Cost (× LLM calls) |
|---|---|---|
| 1 (no debate) | baseline | 1× |
| 3 | +5-10pp (typical) | 3 × R rounds |
| 5 | +1-2pp on top of N=3 | 5 × R rounds |
| 7+ | <1pp marginal gain | 7 × R rounds |

→ **Marginal returns drop sharply from N=3 to N=5**; N=3, R=2-3 captures most of the gain. Exact numbers depend on the task (math vs trivia vs translation behave differently).

> ⚠️ **The cost-accuracy relation is not linear** — returns decrease as N grows; the fundamental reason is **rising correlation between agents** (N agents of the same model family give highly correlated answers, breaking the independent-ensemble assumption; per the Condorcet theorem, convergence speed depends on $\rho$, and a large $\rho$ means a small gain). A common interview question is "why not N=100?" — beyond cost, the answer is correlation.

## §4 Coordination protocols: A2A / blackboard architecture / hierarchical orchestrator

### 4.1 Communication-protocol stack

| Layer | Name | Typical example |
|---|---|---|
| Application | task semantics | "you fix the bug, I'll write the tests" |
| Coordination | message routing | AutoGen GroupChat, GroupChatManager |
| Protocol | message format | **A2A (Google 2025)**, **MCP (Anthropic 2024)**, OpenAI Agents SDK |
| Transport | RPC / WebSocket / stdio | HTTP, gRPC, stdio |

- **MCP (Model Context Protocol, Anthropic Nov 2024)**: protocol between **LLM ↔ tool**, not between agent ↔ agent. But many agent frameworks expose sub-agents as tools via MCP ("sub-agent as tool").
- **A2A (Agent-to-Agent, Google April 2025)**: explicitly targets **multi-agent interconnection**, defining four core objects — agent card / task / message / artifact — over HTTP+JSON-RPC. As of 2026-05 still in spec stage with limited industry adoption.
- **OpenAI Agents SDK** (2025) provides handoff / guardrails / tracing abstractions; in practice it is a proprietary OpenAI protocol.

### 4.2 Blackboard architecture

A classical AI architecture (HEARSAY-II 1980, DARPA project): **multi-agent share a globally readable / writable "blackboard"**, and each agent looks at the blackboard to decide whether to speak. The LLM-era variant looks like:

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
   (writes claim)    (verifies)       (refutes)
```

**Pros**: decoupling — agents don't need to know about each other, they only care about the blackboard state. **Cons**: you need a control unit to decide who speaks (else chaos). AutoGen GroupChat is essentially a blackboard + LLM-as-control-unit.

### 4.3 Hierarchical orchestrator + worker (the Claude Code pattern)

The de facto standard for production agents in 2024-2025: **1 orchestrator + N workers**.

```

           ┌──────────────────────┐
           │   Orchestrator       │ ← look at user query, decide:
           │   (main LLM, big)    │   - split into subtasks
           │                      │   - which worker handles which
           │                      │   - how to aggregate results
           └──────────────────────┘
                ↓     ↓     ↓
         ┌──────┘     │     └──────┐
         ↓            ↓            ↓
    ┌────────┐  ┌────────┐  ┌────────┐
    │Worker A│  │Worker B│  │Worker C│
    │ (code) │  │ (web)  │  │ (math) │
    └────────┘  └────────┘  └────────┘
```

**Claude Code Agent / Cline / Aider** all follow this pattern:

- **Claude Code Agent**: main loop = orchestrator; TaskCreate spins up a subagent (different system prompt + subset of tools), and on completion the subagent passes its final message back to the orchestrator.
- **Cline**: plan mode = orchestrator thinking, act mode = worker execution (same LLM, different prompt).
- **Aider**: automatically splits large changes into small commits; each commit is a sub-conversation.

> 💡 **Cost structure of orchestrator-worker** — orchestrators use big models (Claude 4.6 Sonnet class) for decisions; workers use cheap models (Haiku / Sonnet) for execution — known as **"cascading inference"** (Chen, Zaharia & Zou 2023 "FrugalGPT", arXiv 2305.05176). 1 orchestrator call + 10 worker calls is 5-10× cheaper than 11 orchestrator calls.

### 4.4 Sub-agent blame-shifting (a common bug)

```
Orchestrator: "Task failed."
Worker A: "I finished the code. Worker B's tests are broken."
Worker B: "My tests are fine. Worker A's code has bugs."
Orchestrator: stuck in deadlock
```

**Root cause**: each worker only sees its own conversation, with no ground-truth artifact.

**Fixes**:

1. **Artifact-level verification**: have workers run unit tests / linters and use the result as objective evidence
2. **Third-party judge**: spin up a separate fresh-context judge LLM that reads both conversations + artifacts and decides
3. **Incremental commit**: persist each worker's output to disk immediately + checksum; on blame, diff the files directly

## §5 Long-horizon agents: memory architecture

### 5.1 Three memory classes: sensory / short-term / long-term

Borrowing the Atkinson-Shiffrin three-stage model from cognitive psychology:

| Memory | LLM analog | Capacity | Duration |
|---|---|---|---|
| **Sensory** | current input tokens | a few K | one forward pass |
| **Short-term / Working** | LLM context window | a few K - 1M | one session |
| **Long-term** | external store (vector / KG / file) | unbounded | persistent |

Long-term further splits in two (Tulving 1972):

- **Episodic memory**: concrete events ("yesterday I discussed RLHF with Alice")
- **Semantic memory**: abstract knowledge ("the DPO loss is ...")

In an agent system:

- episodic ≈ **trajectory log** (past sequences of action + observation)
- semantic ≈ **knowledge base** (papers, docs, API knowledge)

> 💡 **Episodic and semantic use different retrieval strategies** — episodic is retrieved by **temporal + spatial proximity** ("things modified in this file in the past 24 hours"); semantic is retrieved by **semantic similarity** ("the knowledge most relevant to the current question"). Production often uses **hybrid retrieval**: filter by time decay first, then rank by embedding.

### 5.2 MemGPT: OS-style virtual memory

MemGPT (Packer et al., arXiv 2310.08560, 2023-10; engineered as **Letta** 2024+) treats LLM context as OS RAM and the external vector store as disk:

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

**Key terms**:

- **Main Context** = the current LLM input (system prompt + working context + recall snippets + dialogue)
- **Recall Memory** = past conversation history (stored chronologically)
- **Archival Memory** = arbitrary facts / documents (stored by semantics)
- **Page Fault** = main context is insufficient, so the LLM needs to retrieve archival/recall content → triggers a function call

**Impact of page faults on latency**:

| Operation | Typical latency (GPT-4o-class) |
|---|---|
| One LLM forward (4K context) | 1-2 s |
| One archival search (vector store) | 50-200 ms |
| One page fault (search → LLM regenerate) | **1-2 s (search) + 1-2 s (synthesis)** |
| Main-context hit (no page fault) | 1-2 s |

→ **A page fault roughly doubles per-turn latency**. Production systems often use **prefetch** (predicting which archival items will be needed and retrieving them in advance) to mitigate.

### 5.3 MemoryBank: hippocampus-inspired

MemoryBank (Zhong et al., AAAI 2024, arXiv 2305.10250) is inspired by the **Ebbinghaus forgetting curve**: each memory carries a **strength** and a **last_access_time**, with exponential decay:

$$S_t = S_0 \exp\left(-\frac{t - t_\text{last\_access}}{\tau}\right)$$

Retrieval ranks by **similarity × current strength**, and **frequently accessed memories recover their strength** (mimicking hippocampal reactivation).

> 💡 **MemoryBank vs MemGPT** — MemGPT treats memory as disk (no priority, retrieved by query); MemoryBank adds dynamic priority to memory (decay/strengthening by access frequency). Production agents often **combine the two**: MemGPT's disk abstraction + MemoryBank's forgetting curve for cache eviction.

### 5.4 GraphRAG: knowledge graph + community

The core problem of vector RAG (the most naive RAG): **it collapses on multi-hop / global questions**.

For example:

- "Summarize the design philosophy of the whole codebase" → vector RAG can only fetch a few files and cannot synthesize globally
- "Who is Alice's friend's friend" → a three-hop query; a single vector retrieval cannot fetch the three-hop path

**GraphRAG** (Microsoft 2024, Edge et al., arXiv 2404.16130) solution:

```
Stage 1 (offline, expensive):
  document → LLM extracts (entity, relation, entity) triples
         → build KG → Leiden community detection → multi-level community summaries

Stage 2 (online, cheap):
  query → classify (local entity? global theme?)
       ↓
  local: find the most relevant entity, look at neighbors, supplement with vector RAG
  global: map-reduce over community summaries
```

**Wins**:

| Benchmark | Vector RAG | GraphRAG | Δ |
|---|---|---|---|
| Multi-hop QA (HotpotQA-class) | ~50% | ~70% | +20pp |
| Global summarization | poor | strong | qualitative |

**Losses**:

- High offline cost (building the KG requires scanning the whole corpus once)
- KG schema design makes or breaks everything

> ⚠️ **GraphRAG is not a silver bullet** — single-hop factoids ("the DPO loss formula") are already well-served by vector RAG; GraphRAG only wins on multi-hop / global queries. **Profile the query distribution first, then decide whether to deploy GraphRAG** — blindly stacking GraphRAG is the most common over-engineering of 2025-2026.

### 5.5 Code: dual-track episodic + semantic retrieval

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
    """ Simplified MemoryBank + episodic / semantic dual-track. """
    def __init__(self, decay_tau_episodic=86400.0, decay_tau_semantic=864000.0):
        self.items: List[MemoryItem] = []
        self.tau_e, self.tau_s = decay_tau_episodic, decay_tau_semantic

    def add(self, text, embedding, kind="semantic"):
        now = time.time()
        self.items.append(MemoryItem(text, embedding, now, now, 0, kind))

    def _strength(self, item: MemoryItem, now: float) -> float:
        tau = self.tau_e if item.kind == "episodic" else self.tau_s
        # access_count boost: each access strengthens by +0.1 (Ebbinghaus spaced repetition)
        boost = 1.0 + 0.1 * item.access_count
        decay = np.exp(-(now - item.last_access) / tau)
        return boost * decay

    def retrieve(self, query_embedding, k=5, kinds=("episodic", "semantic")):
        now = time.time()
        scores = []
        for it in self.items:
            if it.kind not in kinds:
                continue
            # cosine similarity (assumed normalized)
            sim = float(np.dot(query_embedding, it.embedding))
            score = sim * self._strength(it, now)
            scores.append((score, it))
        scores.sort(reverse=True, key=lambda x: x[0])
        top = [it for _, it in scores[:k]]
        # Access strengthens.
        for it in top:
            it.last_access = now
            it.access_count += 1
        return top
```

> 💡 **Multiplicative vs additive strength × similarity** — multiplicative here: when strength decays to 0 the memory disappears entirely (even with high similarity); additive would make strength only a bias. Production uses multiplicative, but **applies a floor to episodic memory** (so new episodic items are not immediately evicted).

## §6 Planning + tree-search agents

### 6.1 Plan-then-execute paradigm

HuggingGPT (Shen et al., NeurIPS 2023, arXiv 2303.17580, Microsoft + Zhejiang U) is an early representative:

```
User Query
   │
   ↓
LLM (Plan)         ← use ChatGPT to decompose the task into a sub-task DAG
   │
   ↓
HF Models (Execute) ← invoke various vision / speech models on HuggingFace
   │
   ↓
LLM (Aggregate)    ← synthesize the outputs of all sub-tasks
```

**Problem**: the plan is fixed at the start and **cannot replan on execution failure**. This is the motivation for the ReAct / Reflexion / tree-search line of work.

### 6.2 ReAct: interleaving thinking + acting

ReAct (Yao et al., ICLR 2023, arXiv 2210.03629, Princeton + Google) format:

```
Thought 1: I need to find Colorado mountain heights.
Action 1: search("Colorado eastern sector elevation range")
Observation 1: 1800 to 7000 ft.
Thought 2: Now I need to find the High Plains elevation.
Action 2: search("High Plains elevation")
...
```

Thought and action interleave at every step, and **this format is now the de facto foundation of every agent** — AutoGPT / LangChain Agent / Toolformer are all ReAct variants.

### 6.3 Tree of Thoughts (ToT)

ToT (Yao et al., NeurIPS 2023, arXiv 2305.10601) introduced **explicit search** for the first time:

```
                root (problem)
               /    |    \
            t1.a  t1.b  t1.c     ← LLM proposes N thoughts
              |   |   |
          ┌────────────────┐
          │  LLM evaluator  │     ← each thought is scored (1-10 / "sure")
          └────────────────┘
              ↓ BFS / DFS expand
            t2.aa t2.ab ...
```

- **Expand**: the LLM proposes K thoughts
- **Evaluate**: the LLM scores each thought / labels it "sure / maybe / impossible"
- **Search**: BFS (advance layer-by-layer top-k) or DFS (with backtracking)

> ⚠️ **ToT's LLM-as-evaluator is self-referential risk** — the same LLM both proposes and evaluates, so it may **score its own thoughts too high**. In production ToT either swaps in a reward model as evaluator or uses a cross-model evaluator.

### 6.4 RAP: MCTS + world model

RAP (Hao et al., EMNLP 2023, arXiv 2305.14992) upgrades ToT: replace the LLM-as-evaluator with **MCTS + LLM-as-world-model**.

```
MCTS step:
  1. select: pick a leaf via the PUCT formula
  2. expand: LLM proposes the next action
  3. simulate: LLM rolls out to terminal, estimating reward
  4. backup: propagate reward up the path to update Q(s, a)

PUCT formula (AlphaGo's improved UCT):
  a* = argmax_a [ Q(s, a) + c_puct * P(s, a) * sqrt(N(s)) / (1 + N(s, a)) ]
```

The formula's terms:

$$\boxed{\;a^* = \arg\max_a \left[ Q(s, a) + c_\text{puct} \cdot P(s, a) \cdot \frac{\sqrt{N(s)}}{1 + N(s, a)} \right]\;}$$

- $Q(s, a)$: mean rollout reward from $(s, a)$
- $P(s, a)$: prior probability (action probability given by the LLM)
- $N(s)$: visit count of state $s$
- $N(s, a)$: visit count of edge $(s, a)$
- $c_\text{puct}$: exploration constant (typically 1.0-3.0)

**Intuition**: the first term $Q$ exploits (pick high-reward), the second term explores (pick low-visit + high-prior). $\sqrt{N(s)} / (1 + N(s,a))$ is a **UCB1 + AlphaGo prior** hybrid.

### 6.5 LATS: MCTS + Reflexion + value

LATS (Zhou et al., ICML 2024, arXiv 2310.04406, Penn + UIUC) further adds Reflexion to RAP:

```
For each expansion:
  1. action = LLM_policy(state, history)
  2. observation = env.step(action)
  3. value = LLM_value(state, history)   ← scalar estimate
  4. if failed: reflection = LLM_reflect(history)  ← injected into the next prompt
  5. back-propagate value along the path
```

LATS outperforms ReAct + Reflexion on HotpotQA / WebShop / HumanEval.

> 💡 **LATS = ToT + Reflexion + MCTS hybrid** — three techniques fused; engineering complexity is high, but sample efficiency is markedly better. Deployment cost + engineering difficulty have kept it from being mainstream production, but **its paper significance / conceptual completeness are strong** — in interviews you can cite it as "one of the most well-designed agent papers I have read".

### 6.6 Agent-Q: MCTS + DPO offline training

Agent-Q (Putta et al., 2024, arXiv 2408.07199, MultiOn + Stanford) reflects on this: MCTS at inference time is too slow — can we **use MCTS offline to generate data for training a policy**?

```
Offline:
  for many episodes:
    trajectory = MCTS_search(env, llm)
    for state, good_action, bad_action in trajectory.preferences():
      DPO_dataset.add(state, good_action, bad_action)
  fine-tune LLM via DPO

Inference: directly use the fine-tuned LLM (no MCTS), with much higher speed
```

**Result**: WebShop success rate goes from 28% → 51%, while inference speed is 10× faster than in-loop MCTS.

### 6.7 LATS-style tree search implementation (~60 lines core)

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
    """ Rollout from (parent.state, node.action) through env to terminal or depth cap.

    Args:
        llm_propose_one: policy — returns a single action for a given state (greedy/random rollout)
        llm_value:       value — estimates the remaining cumulative reward from a state
    """
    state, cur_action, cum_r = node.parent.state, node.action, 0.0
    for _ in range(max_depth):
        state, r, done = env_step(state, cur_action)
        cum_r += r
        if done:
            node.is_terminal, node.state = True, state
            return cum_r
        cur_action = llm_propose_one(state)   # policy rollout picks the next action
    node.state = state
    return cum_r + llm_value(state)            # value head estimates the remainder

def backup(node, value):
    while node:
        node.visits += 1
        node.value_sum += value
        node = node.parent

def lats_search(initial_state, llm_propose, llm_propose_one, llm_value, env_step,
                num_iter=50, c_puct=1.5, max_rollout=5):
    """LATS = MCTS + Reflexion + value estimation.

    llm_propose:       at expand, given (state, k) → [(action, prior)]
    llm_propose_one:   during rollout, given state → action (policy)
    llm_value:         value estimate, given state → remaining cumulative reward
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

> ⚠️ **LATS implementation pitfalls** — (1) `simulate` must roll from parent + action (not reuse the select-leaf state); (2) **policy (`llm_propose_one`) and value (`llm_value`) must be passed separately** — an earlier version mistakenly used value as policy, and the rollout behavior degenerated into looking at state value without picking an action; (3) prior in `expand` should be normalized via `softmax(LLM_logp)`; (4) **terminal detection must be strict**, otherwise rollout loops forever; (5) typical PUCT coefficient c is 1.0-3.0; (6) production should add wall-time budget checks to prevent a 50-iteration run from timing out.

## §7 Long-horizon evaluation benchmarks

### 7.1 Benchmark comparison table (2024-2026)

| Benchmark | Paper | Task type | Steps | 2026-05 SOTA |
|---|---|---|---|---|
| **AgentBench** | Liu et al., ICLR 2024, arXiv 2308.03688 | 8 envs (OS, DB, WebShop, HouseHold...) | 10-50 | GPT-4 ≈ 4.0/10 overall |
| **τ-bench (TAU-bench)** | Yao et al., 2024, arXiv 2406.12045 | customer service multi-turn (retail, airline) | 20-50 | Claude 3.5 Sonnet ≈ 45-55% (retail) |
| **OSWorld** | Xie et al., NeurIPS 2024, arXiv 2404.07972 | real OS GUI (Linux/macOS/Win, ≈ 369 tasks) | 5-100 | Anthropic Claude (2026-05) ≈ 60% (subset) |
| **WebArena** | Zhou et al., ICLR 2024, arXiv 2307.13854 | self-hosted web (Reddit-clone, Gitea, etc.) | 5-30 | GPT-4 ≈ 14.4% (2024); top in 2026 ~ 50% |
| **VisualWebArena** | Koh et al., ACL 2024, arXiv 2401.13649 | WebArena + visual understanding | 10-30 | GPT-4V ≈ 16.4% |
| **SWE-bench / Verified** | Jimenez et al., ICLR 2024, arXiv 2310.06770 | 2294 real GitHub issues (Python repos) | multi-file multi-commit | Claude 4.6 Sonnet (2026-05) ~ 75% (Verified) |
| **MLE-bench** | Chan et al., 2024 (OpenAI), arXiv 2410.07095 | 75 Kaggle ML competition tasks | 24h compute budget | o1-preview + AIDE ≈ 16.9% medals |
| **SWE-Lancer** | OpenAI, 2025, arXiv 2502.12115 | 1488 real Upwork freelance tasks ($1M+ payout) | hours-days | GPT-4o ≈ 8% (managerial), 26% (IC) |
| **Adventure / TextWorld** | Yuan et al., AAAI 2019, arXiv 1806.11532 | text adventure game | 50-500 | RL-trained baseline + LLM > 80% on Coin Collector |

### 7.2 Benchmark selection decision table

| What you want to measure | Pick |
|---|---|
| General agent capability | AgentBench |
| Customer-service dialogue + tool-call accuracy | TAU-bench |
| Real GUI operation (hardest) | OSWorld |
| Coding agent + real-repo fix | SWE-bench Verified |
| Multi-day ML work | MLE-bench |
| Economic value (measured) | SWE-Lancer |
| Web navigation | WebArena / VisualWebArena |

### 7.3 What benchmark horizon length implies about failure modes

- **Short horizon (5-20 steps)**: failures are mainly single-step errors (misreading the query / calling the wrong tool)
- **Medium horizon (20-100 steps)**: context overflow / lost-in-the-middle / replan failures start to appear
- **Long horizon (100+ steps, multi-day)**: memory stale, sub-agent blame, cost explosion, decision paralysis

**At long horizon, every additional 10 steps roughly halves success rate** (an internal Anthropic 2025 observation, partially mentioned in public blogs — not a strict scaling law) — which is why long-horizon agents are the most difficult and most research-valuable direction of 2026.

## §8 Failure modes unique to long-horizon

### 8.1 Context overflow + lost-in-the-middle

Liu et al. TACL 2024 (arXiv 2307.03172) found: **the LLM's utilization of information in the middle of its context is significantly lower than at the beginning or end** — a U-shape curve (head > tail > middle).

```

Recall vs position in 25-doc QA (Liu 2024 Fig 2):

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

**Implications for long-horizon agent design**:

1. **Don't put important facts in the middle** — put the task instruction at the head + the current working memory at the tail
2. **The context window is not always better when longer** — utilization drops significantly above ~30K
3. **Periodic summarization** — periodically summarize the middle context into short snippets pushed to the tail

### 8.2 Error compound / drift

In long tasks **even a small per-step error rate makes the whole thing fail eventually**. A simplified model:

Let single-step success rate be $p$; under independence assumption, total success over $T$ steps is:

$$P(\text{all success}) = p^T$$

$p=0.95$, $T=20$: $P = 0.358$
$p=0.95$, $T=50$: $P = 0.077$
$p=0.99$, $T=50$: $P = 0.605$

→ **An agent with single-step 95% has only 7.7% success on a 50-step task** — single-step must be pushed to 99%+ for 50 steps to hold up.

**Mitigations**:

- **Rollback / replan**: checkpoint every N steps, roll back on failure (analogous to keyframes in video gen)
- **Hierarchical sub-tasks**: split a 50-step task into 5 ten-step sub-tasks, each of which can fail and retry independently
- **Self-verify after each critical step**: have a verifier check after every key action

> 💡 **Drift vs catastrophic failure** — drift is a gradual deviation from the goal (each step is only slightly off); catastrophic is one step going completely off course. **Drift is harder to detect than catastrophic** — the agent may happily keep executing the wrong trajectory. Production defenses against drift rely on periodic self-checks (every 5 steps re-check the task description against the current state).

### 8.3 Decision paralysis / loop

The agent gets stuck in a state and repeats the same action:

```
Action: search("foo")
Observation: no results
Thought: I should search with different keywords
Action: search("foo")   ← again!
Observation: no results
...
```

**Root cause**: the LLM, given that prompt context, is greedy-decoded into the same mode.

**Fixes**:

1. **action history hash**: detect whether the last N steps are repeating (fingerprint), and force exploration if so
2. **temperature ramp**: bump the temperature after K consecutive failures
3. **explicit "give up" tool**: let the agent know "I cannot solve this" is a legal action (to avoid pretending it can solve)

### 8.4 Stale memory

The agent retrieves a **stale** memory and makes a decision based on it. Example:

```
Memory: "API endpoint = https://api.foo.com/v1/users"
Reality: the endpoint has migrated to /v2/users (memory is 6 months old)
Agent: calls the old endpoint → 404 → confused
```

**More dangerous than missing memory**: missing memory makes the agent actively search for new info, but stale memory makes the agent **confidently wrong**.

**Mitigations**:

1. **TTL on memory**: each memory carries an expiration timestamp; once expired, demote or remove
2. **Memory consistency check**: weekly job to diff memory against ground truth
3. **Prefer recent over similar**: change retrieval from `sim` to `sim * exp(-age/τ)`

### 8.5 Sub-agent conflict / blame-shifting

See §4.4.

### 8.6 Cost explosion

A long-horizon agent's token consumption **is not linear**:

- Every turn feeds the full history into the LLM (the KV cache cannot save prompt-token billing)
- Cumulative cost over $T$ turns $\sim \sum_{t=1}^T O(t \cdot c) = O(T^2 c)$, where $c$ is the per-turn new-token count

Measurement: a 50-turn agent task consumes **~ 50-200×** the tokens of a 5-turn task (not 10×).

> ⚠️ **Cost explosion is the #1 killer of production agents** — Devin / Claude Code all use **history compaction**: compress 30 turns of raw conversation into a ~ 1K-token summary. The `/compact` command Anthropic added in 2025-05 is exactly this.

## §9 Self-improvement + verification loops

### 9.1 Reflexion: reflecting from the trajectory log

Reflexion (Shinn et al., NeurIPS 2023, arXiv 2303.11366) algorithm:

```
For each episode:
  1. trajectory = run agent on task
  2. reward = evaluator(trajectory)
  3. if reward low:
       reflection = LLM(trajectory, reward)
                    --> natural-language summary of the failure cause
       memory.add(reflection)
  4. Next episode: prompt includes memory
```

**Natural-language reflections have higher information density than scalar rewards** — the agent knows not only that it "failed" but **why**.

### 9.2 Test-time self-improvement (frame + video two-level)

A²RD-style **hierarchical test-time self-improvement (HITS)** (Liu et al. 2026):

- **Frame-level**: after generating each segment, check with a frame-level verifier (face deformation? lighting inconsistencies?)
- **Video-level**: every K segments, do a video-level check (long-range consistency? character drift?)

**Analogy for long-horizon agents**:

- **Step-level**: check with unit tests / linters after each action
- **Sub-task-level**: integration check after each sub-task
- **Task-level**: final acceptance test

> 💡 **Three-level self-correction is the de facto best practice in agent engineering** — Cursor agent / Claude Code both use this three-level pattern: immediate (lint), intermediate (partial test), final (full test + user review).

### 9.3 Cross-Time Replay (avoid over-specialization)

Ctx2Skill-style **Cross-Time Replay**:

- Maintain a **hard probe set** (historical hard cases that failed) and an **easy probe set**
- After each agent skill update, re-evaluate on both sets
- Pick the version that maximizes $\rho_\text{hard} \times \rho_\text{easy}$ (not just whichever does better on the current task)

This prevents the agent from "learning to cheat" on the new task and regressing on old capabilities (catastrophic forgetting at the skill level).

### 9.4 Dependency DAG for memory retrieval

A²RD-style MVMem uses a **dependency DAG** to decide which past segments to retrieve when generating a given segment:

```
seg_1 -- spatial overlap --> seg_5
seg_2 -- char A appearance --> seg_7
seg_3 -- camera continuation --> seg_4
```

→ When generating seg_5, only retrieve {seg_1}, instead of stuffing all of {seg_1..seg_4} into the prompt.

**Implication for long-horizon agents**: **explicitly model the dependencies** between episodic memories as a graph (not a flat list), and traverse by graph paths instead of brute-force similarity at retrieval time. Production is still exploring (GraphRAG is an early attempt).

## §10 Engineering practice: subagent orchestration + budget tracking

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
    """ Claude Code-style orchestrator + N workers (with budget tracking). """
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
            if out is None:   # budget exhausted → forced finalize
                return self.llm_orch(query=query, history=self.history,
                                     budget=self.budget.remaining(),
                                     hint="budget_exhausted")["answer"]
            self.history.append({"step": step, "subagent": decision["subagent"], "result": out})
        # Steps exhausted: forced finalize
        return self.llm_orch(query=query, history=self.history,
                             budget=self.budget.remaining(),
                             hint="forced_finalize")["answer"]
```

> 💡 **Budget tracking is a hard requirement for production agents** — an agent without a budget will **burn money indefinitely**. Anthropic Claude Code / Cursor / OpenAI Agents SDK all add budget hooks at the SDK layer — not just tokens but also wall time, sub-agent call counts, API rate limits. In production **all three rails must be controlled simultaneously**.

## §11 Complexity analysis

### 11.1 Cost model for multi-agent collaboration

Let:

- $N$ = number of agents
- $R$ = debate / collaboration rounds
- $C$ = average token cost per LLM call
- $L$ = accumulated message history length

**Round-robin GroupChat**:
$$\text{cost} = N \times R \times C \times O(L)$$
where $L$ grows monotonically (each agent writes a message and all agents read), so the total cost $\approx O(N^2 R^2 C)$ (assuming each message length is roughly constant, so $L \propto NR$) — **both N and R are quadratic**.

**MoA** (N proposers + 1 aggregator):
$$\text{cost} = (N + 1) \times C \times O(L_\text{prompt}) + 1 \times C \times O(NL_\text{response})$$
Linear in $N$, **so actually cheaper**.

**Debate (Du 2023)**:
$$\text{cost} = N \times R \times C \times O(NL_\text{response})$$
(each agent reads the other $N-1$ responses) → $O(N^2 R C L_\text{response})$.

**Comparison**:

| Mode | Cost order |
|---|---|
| Single LLM | $O(C L)$ |
| MoA | $O(N C L)$ |
| Debate (Du) | $O(N^2 R C L)$ |
| GroupChat (round-robin) | $O(N R^2 C L)$ or worse |

→ **GroupChat is the most expensive** (which is why production AutoGen almost universally switched to hierarchical orchestrator).

### 11.2 Long-horizon agent memory cost

Let $T$ = total number of turns and $M$ = total memory size.

- **No retrieval** (all in context): cost $\sim O(T^2)$ (each turn looks at the full history)
- **Vector RAG**: each turn retrieves $k$ memories, cost $\sim O(T \cdot (k + \log M))$
- **GraphRAG**: offline build $O(M^2 / \text{community size})$, online query $O(T \cdot \log M)$ + occasional global synthesis
- **MemGPT**: cost $\sim O(T \cdot (k + L_\text{archival\_search}))$; **page faults add ~ 2× per-turn latency**

### 11.3 PUCT-based tree search

LATS / Agent-Q-class tree search at inference:

- One search: $O(I \cdot D \cdot C)$, where $I$ = MCTS iterations, $D$ = max depth, $C$ = per-LLM-call cost
- $I = 50, D = 5, C = $ 4K tokens → ~ 1M tokens per query

→ **Tree search at inference is expensive** — this is exactly the core motivation of Agent-Q (use offline MCTS to train, then deploy the fine-tuned policy directly).

## §12 Comparison with related methods

### 12.1 Multi-agent vs single-agent + reflection

| Dimension | Multi-Agent (debate, MoA) | Single + Reflection (Reflexion, Self-Consistency) |
|---|---|---|
| Source of diversity | multiple models (heterogeneous) | same model, different sampling |
| Cost | $O(NRC)$ and up | $O(NC)$ |
| Failure mode | sub-agent conflict, blame-shifting | echo-chamber (the same bias repeated N times) |
| When to choose | the models actually have complementary capabilities | diversity is mainly from sampling |

**Empirical**: if N agents use the **same base model + same prompt**, multi-agent and self-consistency yield nearly the same gain (shared bias), and **there is no point doing multi-agent**.

### 12.2 Tree search vs long CoT (o1-style)

| Dimension | Tree search (ToT/LATS) | Long CoT (o1, R1) |
|---|---|---|
| Where the search happens | external explicit tree | inside the LLM's hidden CoT |
| Observability | high (tree is visualizable) | low (hidden trace) |
| Inference latency | high (many LLM calls) | medium (single long generation) |
| Training cost | essentially none (inference-only) | high (RL on reasoning) |
| When to choose | need interpretability / can't train | have enough compute to train |

### 12.3 Memory architecture selection

| Scenario | Recommended |
|---|---|
| Small project / prototype | LangChain ConversationBuffer (in-memory list) |
| Medium-size / single-user long session | MemGPT-style RAM/disk + recall |
| Multi-user product + knowledge base | vector store (Chroma / Pinecone) + recall |
| Complex multi-hop QA / document-heavy | GraphRAG |
| Agents that need to "forget" | MemoryBank (Ebbinghaus decay) |
| Video / research / strongly structured | typed memory + dependency DAG (A²RD style) |

## §13 2025-2026 frontier systems

### 13.1 Anthropic Claude Code Agent / Computer Use

- **Claude Computer Use** (Oct 2024): lets Claude operate the desktop directly (screenshot → reason → mouse/keyboard action). ~ 14.9% on OSWorld (2024), reaching ~ 60% by 2026-05 (Anthropic blog).
- **Claude Code Agent** (Feb-2025 GA): orchestrator + subagent pattern; subagents are spawned via TaskCreate (different system prompt + tool subset). Opened to developers as a CLI tool.

### 13.2 OpenAI Operator / Agents SDK

- **Operator** (Jan 2025): the counterpart to Anthropic Computer Use, but with a **CUA (Computer-Using Agent)** trained specifically as a vision policy (not general Claude); WebArena accuracy ≈ 58.1%.
- **OpenAI Agents SDK** (Mar 2025): a Python framework providing handoff / guardrail / tracing abstractions. **Handoff** = the orchestrator "handing off" the task to another agent (more than a tool call).

### 13.3 Cognition Devin / SWE-Agent

- **Devin** (Cognition, Mar 2024 demo): the first commercial "AI software engineer" demo, capable of plan + code + debug + browser end-to-end. **SWE-bench (original) ~ 13.9%** (Mar 2024; note: the *Verified* subset only launched 2024-08) → ~ 50% (late 2025, Devin 3.0).
- **SWE-Agent** (Yang et al., NeurIPS 2024, arXiv 2405.15793, Princeton): an open-source SWE-bench agent that proposed the **Agent-Computer Interface (ACI)** concept — designing a dedicated file-edit / shell / search interface for the LLM (not giving it bash directly), reaching 18.0% on SWE-bench Lite.

### 13.4 Cursor / Cline / Aider / Continue

- **Cursor** (Anysphere): commercial IDE-integrated agent, with plan/act mode switching; Composer is a multi-file-edit subagent.
- **Cline** (open-source, originally Claude Dev): VS Code extension, open-source orchestrator + worker, GitHub stars grew to 30K+ in 2025.
- **Aider**: CLI tool; repo-map + LLM auto-commit; **automatically splitting large changes into small commits** is a benchmark of long-horizon engineering.
- **Continue**: a self-hosted alternative, can be wired to a self-hosted LLM.

> 💡 **Production agent toolchain summary (memorize for interviews)** —
> commercial: Cursor, Devin, GitHub Copilot Agent, Claude Code
> open-source CLI: Cline, Aider, Continue
> framework: AutoGen (MS), CrewAI, LangGraph (LangChain), LlamaIndex AgentWorkflow

## §14 25 frequently-asked interview questions — L1 must-know / L2 advanced / L3 top-lab

### Level 1 — must-know (10 questions)

<details>

<summary>Q1. Difference between multi-agent / long-horizon / agentic?</summary>

- **multi-agent**: multiple LLMs role-playing distinct identities to **collaborate / compete** — CAMEL, AutoGen, MetaGPT, debate
- **long-horizon**: the same agent maintaining a goal across **many turns / many days** — MemGPT, Voyager, OSWorld, SWE-Lancer
- **agentic**: the LLM is equipped with **autonomous plan + tool use + reflection** (agent count can be ≥ 1) — ReAct, Toolformer, AutoGPT

→ Real systems usually involve all three, but the **precise meanings differ** — disambiguate first in interviews.

</details>

<details>

<summary>Q2. What does CAMEL's inception prompt solve?</summary>

It solves **role flipping** and **task drift**:

- **role flipping**: the assistant inadvertently turns into the user ("I'd also like to know this question")
- **task drift**: the task scope widens with chat ("by the way let's also discuss database design")

The inception prompt = a **strongly constrained system prompt** ("you can only issue an instruction, wait for a reply, then issue the next" / "you can only reply, never add a new task") plus format locks like "Next request."

</details>

<details>

<summary>Q3. How many selector strategies does AutoGen's GroupChat have? Which does production use?</summary>

Three:

1. **round_robin**: take turns in order — simple but weak agents drag the chat
2. **random**: pick at random — lacks control
3. **llm_selector**: a manager LLM looks at the history and picks the next speaker — production default, but adds one LLM call per round

In real products (e.g. Microsoft's own offerings) it is almost always llm_selector, **because round_robin tends toward chaos for N > 3**.

</details>

<details>

<summary>Q4. Why does MetaGPT emphasize SOP? What is the essential difference from CAMEL?</summary>

CAMEL is **unstructured chat**: role-play + free-form dialogue → easily hallucinates "we're done" without inspectable artifacts.

MetaGPT enforces a **structured artifact pipeline**:

```
PM → PRD (markdown)
Architect → tech design (class diagram, API)
ProjectManager → task list (JSON)
Engineer → code (.py files)
QA → test cases
```

Each role **only receives the structured output of the previous node** (no free-form chat), forcing inspectability.

→ **Structured vs unstructured is the biggest dividing line in multi-agent engineering**.

</details>

<details>

<summary>Q5. ReAct = ?</summary>

**ReAct = Reasoning + Acting** (Yao et al., ICLR 2023, arXiv 2210.03629):

```
Thought 1: I need to find X.
Action 1: tool_call("search X")
Observation 1: ...
Thought 2: With X, I can conclude...
Action 2: ...
```

**Thought and action interleave**. It is now the de facto standard of every agent framework (AutoGPT / LangChain / Toolformer are all ReAct engineered for production).

</details>

<details>

<summary>Q6. What is MoA? Why N proposers + 1 aggregator rather than best-of-N?</summary>

**MoA (Mixture-of-Agents, Wang et al. ICLR 2025 Spotlight, arXiv 2406.04692)**: N LLMs each give a response, the N responses are **concatenated into the prompt** fed to an aggregator LLM that synthesizes.

Why it beats best-of-N:

- best-of-N: a reward model picks **the single highest-scoring** response — discards the **complementary information** in the other N-1
- MoA: lets the aggregator LLM see all N perspectives and do **latent reasoning + synthesis** in prompt space

**Result**: 6× open-source models in MoA achieve 65.1% LC win rate on AlpacaEval 2.0 > 57.5% for GPT-4 Omni.

</details>

<details>

<summary>Q7. How does multi-agent debate converge?</summary>

**Du et al. ICML 2024 / Liang et al. EMNLP 2024 style**:

```
Round 0: each agent answers independently
Round 1: each agent sees the other N-1 answers → revises
Round 2: revise again
...
Final: majority vote
```

**Convergence properties** (see §3.3 for details): averaging operators (doubly-stochastic) generally **have no unique Banach fixed point** — the eigenvalue in the consensus direction = 1. The correct statement is convergence to a **fixed-point set** (multiple consensus clusters); which cluster the iteration settles into is determined by initial majority + agent bias.

**Practice**: N=3, round=3 already captures ~ 80% of the gain; N=5 adds only +1.2pp but +67% cost (Du 2023 Fig 4).

</details>

<details>

<summary>Q8. What do MemGPT's RAM and disk correspond to? What is a page fault?</summary>

- **RAM** = LLM main context (system prompt + working context + recall snippets + dialogue), capacity ~ a few K - 1M tokens
- **Disk** = external storage (recall memory by time + archival memory by semantics)
- **Page fault** = main context lacks information, so the model function-calls `search_archival` or `search_recall` → retrieves → stuffs results back into the main context

**Page fault impact on latency**: one page fault ~ one extra LLM call, several hundred ms ~ 2 s, doubling per-turn latency. Production uses **prefetch** to mitigate.

</details>

<details>

<summary>Q9. Vector RAG vs GraphRAG? When is GraphRAG worth it?</summary>

| | Vector RAG | GraphRAG |
|---|---|---|
| Representation | embedding chunks | entity-relation KG + community |
| Indexing | similarity search | graph traversal + community summary |
| Multi-hop | weak | strong |
| Global query | can't | yes |
| Offline cost | low | high (LLM-extracted entities) |
| Single-hop factoid | OK | overkill |

→ **For multi-hop + global theme queries, pick GraphRAG**; for single-hop factoids, vector RAG is enough. Blindly stacking GraphRAG is a common 2025-2026 over-engineering pitfall.

</details>

<details>

<summary>Q10. Relationship between ToT, RAP, LATS, and Agent-Q?</summary>

- **ToT** (Yao 2023, NeurIPS) = LLM-propose + LLM-evaluate + BFS/DFS
- **RAP** (Hao 2023, EMNLP) = ToT upgraded to MCTS + LLM-as-world-model (rollout)
- **LATS** (Zhou 2024, ICML) = RAP + Reflexion (natural-language reflection) + value estimation
- **Agent-Q** (Putta 2024, arXiv 2408.07199) = MCTS at training-time + DPO-trained policy → no MCTS at inference (10× speed)

**Common ground**: all use the **PUCT formula** $a^* = \arg\max_a [Q + c P \sqrt{N} / (1 + N_a)]$ to balance exploit / explore.

**Evolution logic**: ToT starts → MCTS makes search explicit → add reflection → train and skip search.

</details>

### Level 2 — advanced (10 questions)

<details>

<summary>Q11. Derive sufficient conditions for debate convergence, and explain the marginal-return gap between N=3 and N=5.</summary>

**Convergence proof**:

View each agent's update as an operator $T_i : \mathcal{A}^N \to \mathcal{A}$ mapping the current answers of N peers to its new answer. The joint update is $T = (T_1, \dots, T_N) : \mathcal{A}^N \to \mathcal{A}^N$.

If there exists a metric $d$ such that $d(T(x), T(y)) \le \beta \cdot d(x, y)$ ($\beta < 1$), then by the **Banach fixed-point theorem**, the iteration $x_{k+1} = T(x_k)$ converges from any $x_0$ to the unique fixed point $x^*$, with rate $d(x_k, x^*) \le \beta^k d(x_0, x^*)$.

**Conditions for $\beta < 1$ to hold**:

- The agent leans toward majority pull (when peers agree, it is easily swayed → strong pull)
- Moderate temperature (too high is random and doesn't converge; too low locks in)
- Same-source agents (heterogeneous agents make contraction harder because biases differ)

**N=3 vs N=5 (empirical observation from Du 2023 / Liang 2024)**:

| N | Improvement over baseline (reasoning tasks, task-dependent) | cost |
|---|---|---|
| 1 | baseline | 1× |
| 3 | +5-10pp | ~9× (N × R) |
| 5 | +1-2pp on top of N=3 | ~15× |

**Reasons for sharp diminishing returns**:

1. **Correlation between agents**: same-model N agents give highly correlated answers; ensemble gain $\sim \sigma^2 (1 + (N-1)\rho)$, larger $\rho$ ⇒ smaller gain
2. **Diminishing return on independent voices**: by the Condorcet theorem in voting theory, the probability that the majority is correct grows with N but at a decreasing rate

**In production N=3, round=2 already captures 80% of the gain**.

</details>

<details>

<summary>Q12. Where is MoA different from self-consistency (Wang 2022)? Why does MoA look stronger?</summary>

- **Self-consistency (Wang 2022)**: same model samples N CoTs, **majority vote**
- **MoA (Wang 2024)**: N **different** models sample, **LLM aggregator** synthesizes

**Key differences**:

1. **Source of diversity**: SC relies on sampling temperature; MoA relies on **model heterogeneity** — the latter is stronger (different model biases ⇒ independent errors)
2. **Aggregation**: SC is hard voting (discards reasoning traces); MoA is LLM-as-aggregator (keeps trace information for latent synthesis)

But MoA's strength may partly come from **a big aggregator** (Qwen2-72B) — if the aggregator is small, the gap shrinks.

**Interview bonus**: MoA is essentially "self-consistency upgraded by replacing majority vote with LLM-as-aggregator + model heterogeneity" — the product of two upgrades.

</details>

<details>

<summary>Q13. What is the lost-in-the-middle phenomenon? What does it imply for long-context agent design?</summary>

**Liu et al. TACL 2024 (arXiv 2307.03172)**: placing the answer at different positions in a 25-doc context and measuring QA recall, they observed a **U-shape**: head and tail recall are high (80%+), with a middle dip to 50%.

**Root causes**:

- attention sink (head tokens are heavily attended)
- recency bias (tail tokens influence downstream logits through the causal mask)
- training-data bias (humans put important points at head and tail)

**Implications for long-context agent design**:

1. **Don't put important facts in the middle**: task instruction → head; current working memory → tail
2. **Longer context windows are not always better**: above ~30K utilization drops significantly; retrieve-then-generate is more stable than stuffing 1M tokens of context
3. **Periodic summarization**: periodically compress the middle of the conversation into a tail snippet
4. **Multi-chunk re-rank** puts the most relevant chunks at the tail (not the head)

</details>

<details>

<summary>Q14. How do you quantify error compound for long-horizon agents?</summary>

A simplified model: with independent single-step success rate $p$, the total success over $T$ steps:

$$P(\text{all success}) = p^T$$

| $p$ | $T=10$ | $T=20$ | $T=50$ | $T=100$ |
|---|---|---|---|---|
| 0.90 | 0.349 | 0.122 | 0.005 | 2.7e-5 |
| 0.95 | 0.599 | 0.358 | 0.077 | 0.006 |
| 0.99 | 0.904 | 0.818 | 0.605 | 0.366 |
| 0.999 | 0.990 | 0.980 | 0.951 | 0.905 |

→ **You need single-step 99% to survive 50 steps**.

**In reality steps are not independent** — some failures propagate (mode-collapse repeating the same error), so effective $p$ is lower; conversely, with verify/rollback between steps you can "repair" things and effective $p$ rises. **Real long-horizon agents have effective $p$ ≈ 0.95-0.98**.

**Mitigations**:

1. **Checkpoint / rollback**: one checkpoint every 5-10 steps
2. **Hierarchical sub-tasks**: split into short horizons, verify each segment independently
3. **Per-step verifier**: unit tests / linters push $p$ higher

</details>

<details>

<summary>Q15. Why does RAG break on multi-hop? How does GraphRAG fix it?</summary>

**Root cause of vector RAG failing on multi-hop**:

```
Q: "Who is Alice's friend's friend?"
Vector space: "Alice", "Alice's friend", "Bob's friend" are not necessarily close in embedding
```

The first-hop chunk ("Alice's friend = Bob") and second-hop chunk ("Bob's friend = Carol") are **not directly connected** in vector space; one retrieval only gets the first hop, and the second hop requires reasoning before retrieving (multi-step retrieval-then-reason) — complex to implement in production.

**GraphRAG fix**:

```
Stage 1 (offline):
  doc → LLM extracts triples (Alice, friend, Bob), (Bob, friend, Carol)
       → build KG → Leiden community detection → multi-level community summaries

Stage 2 (online):
  multi-hop query: Alice → friend → ?? → friend → ??
  KG traversal directly yields Carol
  global query: map-reduce over community summaries
```

**Cost-effectiveness**: ~ +20pp on multi-hop QA; no gain on single-hop factoids but cost is high.

</details>

<details>

<summary>Q16. Physical meaning of each term in the PUCT formula? How to tune c_puct?</summary>

$$a^* = \arg\max_a \left[ Q(s, a) + c_\text{puct} \cdot P(s, a) \cdot \frac{\sqrt{N(s)}}{1 + N(s, a)} \right]$$

- $Q(s, a)$: **exploit** — historical mean reward of that action
- $P(s, a)$: **prior** — LLM-given action probability (softmax over candidates)
- $\sqrt{N(s)}$: grows with parent visit count → more aggressive exploration
- $1 / (1 + N(s,a))$: more visits ⇒ less exploration bias
- $c_\text{puct}$: **exploration coefficient** — typically 1.0-3.0

**Tuning $c_\text{puct}$**:

- Noisy reward → larger $c_\text{puct}$ (more exploration to avoid being misled by a single noisy reward)
- Clean reward + high branching factor → moderate $c_\text{puct}$
- High-quality prior (LLM proposals are accurate) → smaller $c_\text{puct}$ (trust the prior)

AlphaZero uses 1.25-3.0; LATS/RAP implementations often use 1.0-2.0.

</details>

<details>

<summary>Q17. How is orchestrator + worker mode stronger than GroupChat?</summary>

| Dimension | GroupChat (flat) | Orchestrator + Worker (hierarchical) |
|---|---|---|
| Cost order | $O(N R^2 C L)$ | $O(R \cdot (C_\text{orch} + N_\text{used} \cdot C_\text{worker}))$ |
| Control flow | implicit (selector LLM) | explicit (orchestrator decides) |
| Failure traceability | hard (agents cite each other) | easy (orchestrator is the sole decision-maker) |
| Model stratification | same model | can cascade (orchestrator big, workers small) |
| Suitable tasks | short / exploratory | long / engineered |

**Empirical**: production agents are **almost universally orchestrator + worker** — Claude Code, Cursor, Devin, SWE-Agent are all this pattern. GroupChat lives mainly in academic demos and early prototypes.

</details>

<details>

<summary>Q18. How to detect and mitigate agent loops / decision paralysis?</summary>

**Detection**:

1. **Action history hash**: fingerprint the (action, observation) of the past K steps and compare for repeats
2. **Embedding-level similarity**: embed each step's (thought, action) and flag a loop if consecutive cosine similarity exceeds a threshold
3. **Reward stagnation**: reward not changing per step ⇒ possible loop

**Mitigations**:

1. **Force exploration**: after detecting a loop, bump temperature or perturb the prompt randomly
2. **Explicit "give up" action**: let the agent know "I cannot solve this" is legal → avoid pretending it can
3. **History compaction**: periodically compress history into a summary, forcing the agent to forget the bad pattern
4. **Outer-loop time budget**: single-step timeout → orchestrator intervenes forcibly

</details>

<details>

<summary>Q19. What do SWE-bench Verified / SWE-Lancer / MLE-bench measure? How to pick?</summary>

| Benchmark | What it measures | Task length |
|---|---|---|
| **SWE-bench Verified** | real GitHub Python bug fixes (500/2294 reviewed by humans) | hours |
| **SWE-Lancer** | real Upwork freelance tasks (IC + managerial) | hours ~ days |
| **MLE-bench** | Kaggle ML competitions (data exploration + model training) | days (24h budget) |
| **OSWorld** | real OS GUI tasks (screenshot + mouse/keyboard) | minutes |
| **TAU-bench** | customer service multi-turn | minutes |

**Pick**:

- Demo coding agent → SWE-bench Verified (de facto industry standard)
- Demo economic value → SWE-Lancer (payment signal)
- Demo ML R&D agent → MLE-bench
- Demo OS / desktop agent → OSWorld (hardest)
- Demo customer service + tool → TAU-bench

</details>

<details>

<summary>Q20. Is stale memory really more dangerous than missing memory?</summary>

Yes:

- **Missing memory**: agent can't retrieve → actively searches for new info / honestly admits ignorance
- **Stale memory**: agent retrieves an outdated fact → **confidently wrong** (doesn't search to verify)

**Example**:

```
Memory (6 months ago): "OpenAI API endpoint = api.openai.com/v1"
Reality: this endpoint has migrated to /v2 with a new auth scheme
Agent: calls old endpoint → 403 forbidden → confused → keeps retrying
```

**Mitigations**:

1. **TTL on memory**: each memory has an expiry timestamp
2. **Memory consistency check**: periodically diff against external sources (e.g. API docs)
3. **Prefer recent over similar**: retrieval score = sim × exp(-age/τ)
4. **Explicit refresh signal**: if a retrieved fact was used → action failed → mark the memory stale → demote / delete

</details>

### Level 3 — top-lab (5 questions)

<details>

<summary>Q21. Derive (formal) sufficient conditions for multi-agent debate convergence, and explain the essential role of Liang 2024 adding a judge.</summary>

**Setup**: $N$ agents; at round $k$, each gives an answer $x_i^{(k)} \in \mathcal{A}$. Round-update operator:

$$x_i^{(k+1)} = T_i\!\left(x_1^{(k)}, \dots, x_N^{(k)}\right)$$

Joint operator $T : \mathcal{A}^N \to \mathcal{A}^N, \; (T(x))_i = T_i(x)$.

**Important caveat**: saying "Banach contraction → unique fixed point" directly is **wrong**. Debate's averaging operators (majority + softmax) are typically **doubly-stochastic**, corresponding to $A \mathbf{1} = \mathbf{1}$ — eigenvalue 1 always exists — strict $\beta < 1$ generally fails. The correct convergence framework is **consensus dynamics** (multi-agent consensus theory); the conclusion is convergence to a **fixed-point set** (consensus cluster) rather than a unique fixed point.

**Linear averaging case (formal)**:

$$x_i^{(k+1)} = \sum_{j} A_{ij}\, x_j^{(k)},\quad A \in \mathbb{R}^{N \times N}\text{ doubly-stochastic}$$

By Perron-Frobenius, $\lim_{k \to \infty} A^k = \mathbf{1} \pi^\top$ ($\pi$ is the stationary distribution), so $x_i^{(k)} \to \pi^\top x^{(0)}$, i.e. **all agents agree on the stationary mean**. The fixed-point set is $\{c \mathbf{1} : c \in \mathbb{R}\}$ (a consensus line parameterized by $c$); different starts give different $c$ — so it is not a Banach unique fixed point.

**Nonlinear case (softmax-over-peers)**:

$$T_i(x) = \sum_{j \ne i} w_{ij}(x) \cdot x_j$$

If $w_{ij}$ is a majority-vote softmax (temperature $\tau$), the dynamics are nonlinear, but the fixed-point set is still a set of discrete consensus clusters (each corresponding to unanimous agreement on a candidate); different inits converge to different clusters. **A true unique fixed point requires an anchor agent** (fixed reference), but that is no longer debate.

**Liang 2024 (affirmative vs negative) does not converge**:

Liang splits agents into two groups (affirmative / negative): affirmative tends to confirm the current majority; negative tends to negate it. They push each other — **the fixed-point set is empty** or disjoint, so there is no common consensus.

Adding a **judge**:

$$x^{(k+1)} = J(x_\text{aff}^{(k)}, x_\text{neg}^{(k)})$$

$J$ is an **external contractive operator** (not based on internal affirmative/negative updates, but on an independent judge LLM), mapping the whole iteration into a new contractive system. **The judge introduces artificial contraction** — this is the essence of Liang 2024 adding the judge: design-injected external signal that yields $\beta < 1$.

**Empirical calibration**:

Du 2023 experiments on GSM8K-class reasoning tasks observed: N=1 baseline → N=3 R=2-3 already yields large gains (+5-10pp), while N=5 gives only a marginal lift over N=3. This is consistent with $\beta \approx 0.5$, yielding a geometric rate $\beta^3 \approx 0.125$ (i.e. saturation at ~ 12.5% residual disagreement). Liang 2024's affirmative/negative + judge design gives a comparable +5-8pp on translation / counter-intuitive reasoning.

</details>

<details>

<summary>Q22. How do you quantify the page-fault latency impact in MemGPT's RAM vs disk analogy? How to amortize it?</summary>

**Single-turn latency model** (no page fault):

$$L_\text{single} = L_\text{prefill} + L_\text{decode} \cdot T_\text{out}$$

where $L_\text{prefill}$ scales roughly linearly with prompt length $L_\text{in}$ (attention is $O(L_\text{in}^2)$ but amortized by the KV cache), and $L_\text{decode}$ is per-token decoding latency.

**Single-turn latency model (k page faults)**:

$$L_\text{with\_pf} = L_\text{single} + k \cdot (L_\text{search} + L_\text{rerun})$$

- $L_\text{search}$ ~ 50-200 ms (vector store)
- $L_\text{rerun}$ ~ one full LLM call (re-prefill + decode)

Typical numbers (GPT-4o-class, 4K input context, 500-token output):

| Setup | latency |
|---|---|
| no page fault | ~ 1.2 s |
| 1 page fault | ~ 2.5 s |
| 3 page faults | ~ 5-7 s |

**Latency ~ doubles per page fault**.

**Amortization methods**:

1. **Prefetch**: from the current conversation, **predict what memory the next turn will need** and **retrieve in parallel** while the user is typing / the LLM is thinking
2. **Batch faults**: merge several small faults into one big retrieval (cache locality)
3. **Hierarchical paging**: cache hot facts in an "L2 working memory" (not in main context but quickly accessible); cold data in archival
4. **Speculative archival**: retrieve a few extra candidates that may be useful (avoiding a next-turn re-fault)
5. **Streaming response**: start streaming "I'm checking my notes..." when the search starts, hiding latency
6. **Session warmup**: at session start, pre-pull high-frequency archival items into working memory

**Theoretical amortized complexity** (OS page-replacement analogy):

Let working-set size be $W$ and main-context capacity be $C$. $W \le C$ → 0 page faults (cache hit). $W > C$ → fault rate $\sim (W - C) / W$. In production, monitoring fault rate < 30% is healthy.

</details>

<details>

<summary>Q23. Derive the similarity-based selection probability in RAG retrieval, and explain why cosine + temperature scaling breaks down on long retrievals.</summary>

**Setup**: query $q$, candidates $\{d_1, \dots, d_M\}$, embeddings $e(\cdot)$, similarity $s_i = e(q)^\top e(d_i) / (\|e(q)\| \cdot \|e(d_i)\|)$ (cosine).

**Softmax selection**:

$$P(d_i | q) = \frac{\exp(s_i / \tau)}{\sum_{j=1}^M \exp(s_j / \tau)}$$

- $\tau \to 0$: argmax (top-1)
- $\tau \to \infty$: uniform
- In practice retrieval uses top-k threshold rather than softmax sampling; but the softmax view is useful for **theoretical analysis**

**Why long retrieval breaks down**:

Let ground-truth doc $d^*$ have similarity $s^* = e(q)^\top e(d^*)$, and other docs have similarity $s_j$ i.i.d. ~ $\mathcal{N}(\mu, \sigma^2)$.

$$P(d^* = \arg\max_i s_i) = P(s^* > s_j, \forall j \ne d^*) = \prod_{j \ne d^*} P(s^* > s_j) \approx \Phi\!\left(\frac{s^* - \mu}{\sigma}\right)^{M-1}$$

When $M$ is large:

$$\log P \approx (M - 1) \log \Phi\!\left(\frac{s^* - \mu}{\sigma}\right)$$

→ **$P$ decays exponentially in $M$**: doubling M means squaring P.

**Intuition**: a large candidate pool ⇒ many noise distractors ⇒ ground truth gets drowned out.

**Four failure points of real RAG systems**:

1. **Distractor density** ↑: docs that are surface-relevant but semantically irrelevant grow in number
2. **Embedding rank collapse**: high-dim embeddings collapse on large corpora (Wang et al. 2022 anisotropy); all docs become roughly equidistant
3. **Chunk boundary mismatch**: after long-doc chunking, ground truth spans across chunks
4. **Lexical-semantic mismatch**: query uses "RLHF" but doc says "human feedback fine-tuning"

**Fixes**:

- **Re-ranker** (e.g. Cohere rerank, MS bge-reranker): cross-encoder rerank of top-100 candidates
- **Hybrid search**: BM25 + vector ensemble (BM25 guards against lexical mismatch)
- **Query rewriting**: have an LLM rewrite the query into multiple versions and union the top-k of each
- **Chunk overlap + summary**: overlap chunks + chunk-level summaries (structured retrieval)

</details>

<details>

<summary>Q24. Why is Agent-Q's DPO-trained policy 10× faster than in-loop MCTS at inference? How is MCTS data converted into DPO preferences during training?</summary>

**Inference-speed gap**:

- **In-loop MCTS**: each inference runs ~ 50 iterations × ~ 5 depth = ~ 250 LLM calls, ~ 1M tokens per query
- **DPO-trained policy**: a single forward pass + greedy decode = ~ 1-3 LLM calls (**100× fewer**)

→ The measured 10× is because the trained policy may still need a bit of sampling / re-planning, but it is still 1-2 orders of magnitude less than MCTS.

**MCTS → DPO data generation**:

Each MCTS rollout produces a tree $\mathcal{T}$. At each state $s$ there are multiple child actions $\{a_1, \dots, a_k\}$ with their visit counts $N(s, a_j)$ + values $Q(s, a_j)$.

**Construct preference pairs**:

For each state $s$:

- $a^+$ = argmax visit count (the action MCTS explored to be "good")
- $a^-$ = argmin visit count among visited (the action MCTS explored to be "bad")

Yielding the preference: $(s, a^+) \succ (s, a^-)$.

You can also filter by **value gap**: keep only pairs with $Q(s, a^+) - Q(s, a^-) > \delta$ (high-confidence preferences).

**DPO loss**:

$$\mathcal{L}_\text{DPO} = -\log \sigma\!\left( \beta \log\frac{\pi_\theta(a^+ | s)}{\pi_\text{ref}(a^+ | s)} - \beta \log\frac{\pi_\theta(a^- | s)}{\pi_\text{ref}(a^- | s)} \right)$$

Fine-tunes the base LLM directly.

**Agent-Q results** (WebShop):

| Method | success rate | inference cost |
|---|---|---|
| Base LLM | 28% | 1× |
| ReAct | 38% | 1× |
| In-loop MCTS | 48% | ~100× |
| Agent-Q (DPO from MCTS) | **51%** | 1× |

→ **MCTS at training, fast policy at inference** is a key design pattern in 2024-2026 — analogous to AlphaGo Zero's self-play + DPO-style distillation.

**Interview bonus**: this pattern also appears in **DeepSeek-R1 distill** (R1-Zero MCTS-like search produces data → distill to small models) and **rStar-Math** (Microsoft 2025 MCTS + PPM self-evolution); the shared theme is **inference-time search is a data-production tool, not a deployment target**.

</details>

<details>

<summary>Q25. If you were designing the next generation of long-horizon agent, in which directions would you push? (open-ended top-lab question)</summary>

A credible answer framework (no need to cover all; pick 2-3 to dive into):

- **Direction 1 — typed memory + dependency DAG**
  - Status quo: memory is mostly a flat list (vector store) or a KG (GraphRAG)
  - Missing: **causal / dependency relations between events** — why retrieve this rather than that?
  - Proposal: borrow A²RD's MVMem, build an explicit dependency DAG (spatial/temporal/causal) over episodic memory, and retrieve along graph paths rather than one-shot similarity
  - Expected failure: DAG maintenance cost (the graph densifies as events grow); choosing the retrieval policy / edge algorithm (GNN? LLM-as-graph-traverser?)

- **Direction 2 — self-evolving skill set + cross-time replay**
  - Status quo: agent skills are hand-written prompt templates (e.g. ReAct, ToT, Reflexion)
  - Missing: **the agent does not automatically discover new skills or deprecate old ones**
  - Proposal: Ctx2Skill-style 5-role self-play (challenger, reasoner, judge, proposer, generator) + cross-time replay to prevent over-specialization
  - Expected failure: skill collapse (a few skills dominate all tasks); reward hacking (the generator learns to write skills that please the judge but are actually useless)

- **Direction 3 — heterogeneous multi-agent + cross-family verification**
  - Status quo: multi-agent is almost universally same-base-model
  - Missing: **no cross-family verification** — same models share hallucinations
  - Proposal: Claude + Codex + Gemini heterogeneous co-evolution, cross-model audit to prevent self-confirmation bias
  - Expected failure: model strength / price / API versions confound variables; reviewers may also hallucinate together

- **Direction 4 — long-horizon cost model + adaptive budget**
  - Status quo: token billing is linear, but long-task cost is actually quadratic
  - Missing: **no cost-aware planner**
  - Proposal: add expected cost into utility at decision time (not just reward), dynamically adjust verbosity / search depth / reflection frequency
  - Expected failure: cost prediction is inaccurate (LLMs can't accurately estimate their own token use); over-thrift drops quality

- **Direction 5 — formal verification / safety guarantees**
  - Status quo: reflection / debate / audit are all empirical
  - Missing: **no formal correctness guarantee** — what if multi-agent voting is wrong?
  - Proposal: borrow Bayesian truthful elicitation / peer prediction (e.g. Prelec score) to give audit an incentive-compatible foundation
  - Expected failure: hard to bridge theory to production; agents may not understand truthful incentives

- **Direction 6 — evaluation standards**
  - Status quo: benchmarks are fragmented (TAU / OSWorld / SWE-bench each cover one facet)
  - Missing: **no unified long-horizon evaluation standard**
  - Proposal: define a horizon-axis benchmark: test 5 / 20 / 100 / 1000 steps on the same task family; track **per-step error compound rate** (not just final accuracy)
  - Expected failure: human-annotation cost for 1000-step tasks is prohibitive; replay / contamination is hard to control

- **Direction 7 — sim-to-real for agents**
  - Status quo: training / testing both in simulators (WebArena)
  - Missing: **sim-to-real gap** — simulator-trained agents break on the real web
  - Proposal: port robotics sim-to-real tools to agents (domain randomization, real-world fine-tune)
  - Expected failure: web/OS is higher-dimensional than robotics (vision + text + workflow); real-world data collection is expensive

Copy-and-paste existing methods + small tweaks — does not exhibit research taste. **The key is to demonstrate the ability to list 3-5 concrete proposals + each with an expected failure mode**.

</details>

## §A Appendix: core paper timeline + one-sentence summaries

In reverse chronological order (as of 2026-05):

| Date | Paper | arXiv | One-sentence contribution |
|---|---|---|---|
| 2025-09 | Anthropic Claude Sonnet 4.5 + extended thinking | (no arXiv) | long-horizon reasoning + budget-aware thinking |
| 2025-03 | OpenAI Agents SDK | (no arXiv) | handoff / guardrails / tracing abstractions |
| 2025-01 | OpenAI Operator | (no arXiv) | CUA-trained vision-language agent for the web |
| 2024-10 | Claude Computer Use | (no arXiv) | desktop GUI agent, OSWorld ~ 14.9% (2024) |
| 2024-08 | Agent-Q | 2408.07199 | MCTS + DPO, WebShop 28% → 51% |
| 2024-06 | MoA (Mixture of Agents) | 2406.04692 | N proposers + aggregator beats GPT-4 Omni |
| 2024-06 | TAU-bench | 2406.12045 | customer service multi-turn agent benchmark |
| 2024-05 | SWE-Agent | 2405.15793 | ACI for SWE-bench Lite 18.0% |
| 2024-04 | OSWorld | 2404.07972 | real OS GUI, 369 tasks |
| 2024-04 | GraphRAG | 2404.16130 | LLM-extracted KG + community for global QA |
| 2024-01 | VisualWebArena | 2401.13649 | vision + web tasks |
| 2023-12 | Math-Shepherd | 2312.08935 | MCTS rollout automatically labels PRM |
| 2023-10 | LATS | 2310.04406 | MCTS + Reflexion + value |
| 2023-10 | SWE-bench | 2310.06770 | 2294 real GitHub Python bugs |
| 2023-10 | MemGPT | 2310.08560 | OS-style virtual memory for LLMs |
| 2023-08 | AutoGen | 2308.08155 | conversable agent / GroupChat framework |
| 2023-08 | MetaGPT | 2308.00352 | SOP-driven software-company multi-agent |
| 2023-08 | AgentBench | 2308.03688 | 8 envs general agent benchmark |
| 2023-08 | AgentVerse | 2308.10848 | expert recruitment + collaborative decision |
| 2023-07 | WebArena | 2307.13854 | self-hosted real web benchmark |
| 2023-07 | ChatDev | 2307.07924 | chat chain for software dev |
| 2023-05 | RAP | 2305.14992 | MCTS + world model in LLM agents |
| 2023-05 | Tree of Thoughts | 2305.10601 | LLM propose + LLM evaluate + tree search |
| 2023-05 | Multi-Agent Debate (Du) | 2305.14325 | N-agent debate → consensus |
| 2023-05 | Multi-Agent Debate (Liang) | 2305.19118 | affirmative / negative + judge |
| 2023-05 | MemoryBank | 2305.10250 | Ebbinghaus forgetting curve |
| 2023-04 | Generative Agents (Park) | 2304.03442 | 25-agent sandbox social emergence |
| 2023-03 | Reflexion | 2303.11366 | natural-language self-reflection from trajectory |
| 2023-03 | CAMEL | 2303.17760 | role-play inception prompting |
| 2023-03 | HuggingGPT | 2303.17580 | plan-then-execute with HF models |
| 2022-10 | ReAct | 2210.03629 | interleaved thought-action |

> 💡 **Suggested 5 papers to read closely** — when interview-prep time is limited, prioritize:
>
> 1. **MemGPT (2310.08560)** — virtual-memory abstraction, ancestor of long-horizon memory
> 2. **MoA (2406.04692)** — the strongest empirical multi-agent baseline
> 3. **Multi-Agent Debate (Du, 2305.14325)** — origin of the debate paradigm
> 4. **LATS (2310.04406)** — the engineering exemplar for tree-search agents
> 5. **MetaGPT (2308.00352)** — structured multi-agent SOP

> ⚠️ **Preparing for the common open-ended question** — top-lab interviews often ask open-ended questions (such as Q25); the key is to demonstrate **research taste**: list 3-5 concrete directions (not vacuous "I'd do multi-agent" lines), with each direction backed by one concrete proposal + one expected failure mode.

> ✅ **2026 fall recruitment: framework / toolchain summary** — the de facto standards for production agents in 2026 (memorize for interviews):
>
> - **Commercial IDE-integrated**: Cursor (Anysphere), GitHub Copilot Agent, Continue
> - **Commercial coding agent**: Devin (Cognition), Claude Code (Anthropic), OpenAI Codex CLI
> - **Open-source CLI**: Cline, Aider, opencode, codex-cli
> - **Multi-agent framework**: AutoGen (MS), CrewAI, LangGraph (LangChain), LlamaIndex AgentWorkflow, OpenAI Agents SDK
> - **Protocol**: MCP (Anthropic, LLM↔tool), A2A (Google, agent↔agent)
> - **Benchmark**: SWE-bench Verified (industry standard) + TAU-bench + OSWorld + MLE-bench

After reading §10-§14 + the 5 papers above, you should cover 80%+ of multi-agent + long-horizon agent interview questions.
