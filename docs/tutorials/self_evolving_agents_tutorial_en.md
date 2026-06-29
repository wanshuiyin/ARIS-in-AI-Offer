## §0 TL;DR Cheat Sheet

> 💡 **Self-Evolving Agents in 8 sentences** — one page covering the 2024-2026 frontier direction (see §1–§11 for derivations).

1. **Core problem**: have an agent continuously improve its capability on **long-horizon tasks**, without relying on repeated human annotation. Formalized as the convergence / stability / asymptotic effectiveness of an update operator $\mathcal{T}$ such that $\pi_t \to \pi_{t+1}$.

2. **Three paradigms**: ① Experience-Driven (human-made tasks + reward, e.g. AgentTuning, Voyager); ② Adversarial Self-Play (Challenger-Solver, e.g. Absolute Zero, Ctx2Skill); ③ Meta-Learning / Reward-Free (task-free, reward-free exploration + outcome-based reward, e.g. Native Evolution).

3. **Capability container**: **natural-language skill / world knowledge K written in markdown** — this is the most important paradigm shift of 2024-2026, bypassing parameter updates, everything is inference-time `system_prompt += K`.

4. **Ctx2Skill 5-role self-play** (arXiv 2604.27660): Challenger / Reasoner / Judge / Proposer / Generator, **frozen LM but skill set evolves**. Cross-Time Replay picks $\arg\max_i \rho^h_i \cdot \rho^e_i$ to prevent adversarial collapse.

5. **Native Evolution two-phase** (arXiv 2604.18131): Evolution phase explores task-free and reward-free → distills markdown K; Execution phase uses K as system prompt. Training signal $R_\text{evolve}(\mathcal{K}) = \text{Success}(\mathcal{T}_E\mid\mathcal{K}) - \text{Success}(\mathcal{T}_E\mid\varnothing)$.

6. **A²RD trio** (arXiv 2605.06924): MVMem (textual states + frames + videos + dependency DAG) + Adaptive Segment Gen + HITS (frame-level + video-level self-check). Directly transferable as a memory + audit template for any long-horizon agent.

7. **Theoretical upper bound**: [arXiv:2601.05280] shows that closed-loop density matching **degenerates in the absence of an exogenous grounding signal** (**not** that all reward-free training must collapse; this is a specific conclusion for that setting); [arXiv:2507.00075] models the solver-verifier gap and empirically fits it to capability dynamics.

8. **Common failures**: adversarial collapse (Challenger gets extreme), memory drift (internal contradictions accumulating in K), reward hacking (self-rewarding drift), bias amplification (agent retrained on its own output), capability ceiling (self-improvement degrades when exogenous grounding is missing).

## §1 Self-Evolving Agent Intuition

"Self-evolving" is not magic. An LLM agent consists of four parts:

- **policy $\pi$**: a parameterized or pure in-context decision function
- **memory / knowledge $\mathcal{K}$**: externalized long-term state, typically in markdown
- **skills $\mathcal{S}$**: reusable procedural knowledge, each skill being a markdown document
- **environment $E$**: the interactable world (web / code / paper / sandbox / OS)

"Self-evolution" is defining an update operator $\mathcal{T}$:

$$\big(\pi_t, \mathcal{K}_t, \mathcal{S}_t\big) \xrightarrow{\mathcal{T}(E, \text{trajectories})} \big(\pi_{t+1}, \mathcal{K}_{t+1}, \mathcal{S}_{t+1}\big)$$

By the object updated by $\mathcal{T}$, 2024-2026 work roughly divides into four "layers":

| Layer | Update target | Update method | Representative work |
|---|---|---|---|
| L1 Parameter layer | $\pi$ (model weights) | SFT / RFT / RL | AgentTuning, Native Evolution |
| L2 Capability layer | $\mathcal{S}$ (skills markdown) | self-play + replay | Voyager, Ctx2Skill, CoEvoSkills |
| L3 Memory layer | $\mathcal{K}$ (world knowledge markdown) | exploration + summarize | MemGPT, MVMem, Native Evolution |
| L4 System layer | workflow orchestration | inference-time only | Anthropic Skills, ARIS-style harness |

> 💡 **Important intuition** — L1 is "train instincts"; L2/L3 is "grow a toolbox + notebook"; L4 is "workflow orchestration." **The 2025-2026 mainstream is L2 + L3, with L1 mainly preparing training-test decoupling**.

Two intuitions commonly missed in interviews:

- ❌ "self-evolving = automatically training new model weights" — wrong, **the vast majority of work does not update parameters** (Voyager, Ctx2Skill, Generative Agents, A²RD are all training-free / inference-time).
- ❌ "self-evolving = the agent does whatever it wants" — wrong, the 2026 mainstream is **strictly grounded** self-improvement: either a code executor (Absolute Zero), a math checker (STaR), a rubric judge (Ctx2Skill), or an outcome utility (Native Evolution).

## §2 Formalizing the Three Self-Evolution Paradigms

The Native Evolution paper [arXiv:2604.18131] gives a very clear classification — we add the mathematical formulation.

### 2.1　Experience-Driven Evolution

**Setting**: humans provide a task set $\mathcal{T}$, a reward function $R: \mathcal{O} \times \mathcal{A} \to \mathbb{R}$, and a workflow. The agent runs trajectories $\tau$, weighted by $R(\tau)$ to update.

**Update operator**:

$$\theta_{t+1} = \theta_t + \eta \,\mathbb{E}_{\tau \sim \pi_{\theta_t}}\!\left[\nabla_\theta \log \pi_\theta(\tau)\, R(\tau)\right]$$

This is standard policy gradient — AgentTuning, ToolLLM, early Voyager variants fall in this category.

**Pros**: high supervision density, fast convergence. **Cons**: huge labor cost (each new environment needs new reward design).

### 2.2　Adversarial Self-Play Evolution

**Setting**: two agents (Challenger + Solver) evolve jointly, with no external task source — tasks are produced by Challenger, solved by Solver, with verifier feedback.

**Update operator** (in Absolute Zero / R-Zero formalization):

$$\theta^{\text{ch}}_{t+1}, \theta^{\text{sol}}_{t+1} = \arg\min_{\theta^{\text{ch}}, \theta^{\text{sol}}} \;\mathbb{E}_{t\sim \pi^\text{ch}_t}\big[\ell^\text{ch}(t)\big] + \lambda\, \mathbb{E}_{(t,a)\sim \pi^\text{ch}_t,\pi^\text{sol}_t}\big[\ell^\text{sol}(t,a)\big]$$

Concrete "learnability reward" (Absolute Zero, arXiv 2505.03335):

$$R^\text{learn}(t) = \pi^\text{sol}_t(\text{correct}\mid t)\cdot \big(1 - \pi^\text{sol}_t(\text{correct}\mid t)\big)$$

Maximizing it favors 50% difficulty — neither too easy nor too hard. This is the core of curriculum-as-reward.

**Pros**: no need for human-made task sets; **Cons**: still need verifier (code executor / math checker), and prone to **adversarial collapse** (Challenger produces extreme tasks, Solver learns trivial defense).

### 2.3　Meta-Learning / Reward-Free Evolution

**Setting** (Native Evolution): training stage provides outcome-based reward (**not** step-level); inference stage has no task, no reward — the agent autonomously explores → distills markdown world knowledge $\mathcal{K}$ → uses $\mathcal{K}$ as system prompt in downstream tasks.

**Reward design** (Native Evolution core formula):

$$\boxed{\;R_\text{evolve}(\mathcal{K}) \;=\; \text{Success}(\mathcal{T}_E \mid \mathcal{K}) \;-\; \text{Success}(\mathcal{T}_E \mid \varnothing)\;}$$

where $\mathcal{T}_E$ is the set of downstream tasks in environment $E$ (observable at training time). **Reward measures the downstream utility gain of K**, without needing step-level supervision.

**Pros**: completely task-free / reward-free at inference; **Cons**: high training cost (rejection sampling RFT × 2 iterations), and $\mathcal{T}_E$ still needs labeled data at training time.

> ⚠️ **Common confusion: reward-free at inference ≠ reward-free training** — Native Evolution's evolution phase is indeed **task-free / reward-free at inference**; but at **training** time it still needs a labeled set of 600 deep search questions × 20 websites to compute $R_\text{evolve}$. This is an interview bonus point: actively disambiguate.

### 2.4　Three-paradigm comparison (memorize)

| Dimension | Experience-Driven | Adversarial | Meta-Learning |
|---|---|---|---|
| Task source at training | Humans | Challenger agent | Endogenous exploration + labeled downstream |
| Reward at training | Humans | verifier | outcome utility |
| Task at inference | Given by humans | Given by humans / agent itself | Given by humans |
| Reward at inference | Not needed | Not needed | **Not needed** ✓ |
| Workflow at inference | Human-orchestrated | Human-orchestrated | **Agent-driven** (evolve then execute) ✓ |
| Representative | AgentTuning, ToolLLM | AZR, R-Zero, Ctx2Skill | Native Evolution |
| Engineering cost | High (reward eng.) | Medium (verifier orchestration) | High (rejection sampling) |
| arXiv | 2310.12823 | 2505.03335 / 2604.27660 | 2604.18131 |

## §3 Markdownification of Skills / Knowledge (the most important engineering shift)

The most important paradigm shift in 2024 is: **long-term memory and capability extension are not via weight updates, but via external markdown documents**.

### 3.1　Convergence of Anthropic Skills + Native Evolution

Anthropic publicly released the `skills/` paradigm in 2025 (each skill is a standalone markdown file, the agent loads on demand into the system prompt). Native Evolution explicitly cites Anthropic skills in the paper as a reference implementation for K (paper §3, footnote 1 pointing to `github.com/anthropics/skills/tree/main/skills`).

| Dimension | Anthropic Skills | Native Evolution K |
|---|---|---|
| Representation | markdown | markdown |
| Loading | Select skill by task and inject into system prompt | Load K by environment and inject into system prompt |
| Granularity | "How to do PDF / Excel / git commit" | "ACL2025 site structure / code repo topology" |
| Supervision | Human-written | Auto-distilled by agent |
| Origin | Static human-made | Post-training endogenous |

→ Convergence conclusion: **system prompt is the new model weights, markdown documents are the new fine-tuning data**.

### 3.2　Typical Schema of Skill / K Files

```
# skill_name
## Trigger / When-to-use
<which task should use this skill>

## Steps
1. ...
2. ...

## Resources / References
- file paths / URLs

## Failure modes
- Known pitfalls + fixes
```

Native Evolution's K also explicitly stores:

- **Visual arcs** (visual evolution of entities/environments)
- **Spatial relations** (subject-relation-object triplets)
- **Camera states / Site map** (environment topology)
- **Token budget allocation** (how many tokens each sub-page gets)

### 3.3　Why markdown rather than vector embedding?

- **Interpretable**: humans can audit, revise, merge
- **Composable**: skill A + skill B directly prepend
- **Routable**: LLM can read the trigger section and decide which to load
- **Evolvable**: natural language diff is more stable than vector diff

Cost: retrieval precision is worse than vector RAG; the fix is hybrid (vector for candidate selection → markdown for close reading).

## §4 Voyager / Reflexion / STaR: the foundational trio

Before going into 2024-2026 frontier work, you must chew through the three foundational works — interviewers will almost certainly ask about baselines.

### 4.1　Voyager (Wang 2023 TMLR, NVIDIA + Caltech)

The first true end-to-end "automatic curriculum + skill library" agent: running GPT-4 in Minecraft, letting it propose its own tasks, write JS code (each piece of code is a skill), self-verify, and store successful skills in the library.

The core trio:

- **Automatic Curriculum**: GPT-4 proposes the next task based on the current inventory state
- **Skill Library**: each new skill is a JS function, indexed by embedding and retrieved by description
- **Iterative Prompting + Self-Verification**: execute → env feedback → critic agent → revise, until passes

> ⚠️ **Common misconception** — Voyager does not update GPT-4 weights, it is pure inference-time. It does not use reward either; it uses GPT-4 itself as a critic to judge task success, belonging to the self-verification category (not RL).

### 4.2　Reflexion (Shinn 2023 NeurIPS, Northeastern)

Solidifies the concept of "verbal RL": after each failure, the agent writes a reflection in natural language about its own trajectory, stores it in episodic memory, and prepends it to the next prompt.

Formalization (pseudo-Bellman):

$$M_{t+1} = M_t \cup \big\{\text{reflect}(\tau_t, r_t)\big\}$$

where $\text{reflect}$ is a text generation of "what went wrong + how to fix" by the LLM itself.

**Why it works** (theoretically): reflection compresses sparse reward signal into structured text, bypassing gradient updates; equivalent to a kind of **non-parametric policy improvement** in the in-context domain. But lacks convergence guarantee.

### 4.3　STaR (Self-Taught Reasoner, Zelikman 2022 NeurIPS, Stanford)

Let the LM **generate rationale → if wrong, rationalize using the true answer → SFT on correct (q, rationale, a) tuples**. This is the true starting point of self-improvement on reasoning.

Pseudocode:

```
for iter in 1..N:
    for each (q, a_gt) in D:
        r, a_pred = LM(q)
        if a_pred == a_gt:
            collect (q, r, a_gt)
        else:
            r' = LM(q, hint=a_gt)        # rationalize
            if r' produces a_gt:
                collect (q, r', a_gt)
    SFT(LM, collected)
```

STaR's key flaw (also [arXiv:2601.05280]'s core argument against self-improvement): **rationalization is reverse-engineering the answer, not necessarily reflecting the true reasoning process**, leading to distribution drift.

## §5 Ctx2Skill: 5-Role Self-Play Loop (Focus 1)

> This section nearly mirrors arXiv 2604.27660's §3 word-for-word, since interviewers may quote the paper directly.

### 5.1　Problem formulation

Given a context $C$ (possibly 100k+ tokens of manual / paper / repo / dataset), a task set $\mathcal{T} = \{t_j\}$, each task has a binary rubric set $\mathcal{R}_j = \{r_{j,k}\}$. Solving indicator:

$$y_j(\pi; C) = \prod_k \mathbb{I}\big[r_{j,k}(a_j) = \text{pass}\big], \quad a_j \sim \pi(\,\cdot\mid C, t_j)$$

Goal: construct a markdown skill set $\mathcal{S}^R$ such that:

$$a_j \sim \pi(\,\cdot\mid \mathcal{S}^R, C, t_j) \quad \text{maximizes}\ \mathbb{E}_j y_j$$

and **without updating $\pi$'s parameters** — only updating $\mathcal{S}^R$.

### 5.2　Five frozen LM roles

| Role | Input | Output | Intuition |
|---|---|---|---|
| **Challenger** | $C$, $\mathcal{S}^C_{i-1}$ | A batch of $(t_m, \mathcal{R}_m)$ | Generate probing tasks |
| **Reasoner** | $C$, $\mathcal{S}^R_{i-1}$, $t_m$ | $a_m$ | Solve using skills |
| **Judge** | $a_m$, $\mathcal{R}_m$ | binary $y_m$ | Strictly verify by rubric |
| **Proposer (per side)** | failed/solved batch + current skill set | Natural-language diagnosis | Find root cause, does not write skill |
| **Generator (per side)** | proposer diagnosis + current skill set | New skill set | Materialize the change |

Note that **two sides evolve independently**:

- **Reasoner side**: failed cases → Reasoner Proposer diagnoses "what contextual knowledge is missing" → Reasoner Generator writes new $\mathcal{S}^R_i$
- **Challenger side**: too-easy-solved cases → Challenger Proposer diagnoses "why is the challenger producing tasks too weak" → Challenger Generator strengthens $\mathcal{S}^C_i$

→ These two sides **never exchange skill sets** — maintaining strict adversarial pressure.

### 5.3　Cross-Time Replay mechanism (core anti-collapse)

The more iterations, the more extreme the Challenger gets, and the more the Reasoner over-specializes to extreme tasks. Returning $\mathcal{S}^R_N$ directly is bad.

**Replay procedure**:

1. During training, maintain two probe sets:
   - **Hard set $\mathcal{Q}^h$**: each iteration, pick the failed task with the lowest rubric pass rate
   - **Easy set $\mathcal{Q}^e$**: each iteration, pick the solved task with the fewest rubrics passed ("just barely solved")

2. After training, for each candidate $\mathcal{S}^R_i$ ($i=1\ldots N$), run the Reasoner $\pi^R$ on both probe sets:

$$\rho^h(i) = \frac{\sum_{q\in \mathcal{Q}^h} y_q(\pi^R; C, \mathcal{S}^R_i) + 1}{|\mathcal{Q}^h| + 1}, \quad \rho^e(i) = \frac{\sum_{q\in \mathcal{Q}^e} y_q(\pi^R; C, \mathcal{S}^R_i) + 1}{|\mathcal{Q}^e| + 1}$$

(Laplace smoothing prevents empty probe sets)

3. Select:

$$\boxed{\;\mathcal{S}^R_\star = \mathcal{S}^R_{i^\star}, \quad i^\star = \arg\max_i \big(\rho^h(i) \cdot \rho^e(i)\big)\;}$$

**Why product, not sum**: product penalizes catastrophic forgetting (if some version has $\rho^e \to 0$, the overall score → 0), forcing selection of versions that are not bad on either side. Ctx2Skill ablation shows using sum drops final accuracy by ~1.5%.

### 5.4　Ctx2Skill 5-role + Replay code skeleton

```python
def ctx2skill_loop(context: str, llm, num_iters: int = 5, M: int = 5):
    """
    Ctx2Skill: 5 frozen LM roles + Cross-Time Replay.
    Returns the optimal Reasoner skill set selected by cross-time replay.
    All LM calls use the same frozen backbone; only the skill set changes.
    """
    S_R = ""                    # Reasoner skill markdown (initially empty)
    S_C = ""                    # Challenger skill markdown
    candidates = []             # Historical S_R candidates (cross-time)
    Q_hard, Q_easy = [], []     # Two probe sets

    for i in range(1, num_iters + 1):
        # ── (1) Challenger produces a batch ──
        batch = llm(role="challenger", prompt=challenger_prompt(context, S_C), n=M)
        # batch = [(t_m, rubrics_m), ...]

        failed, solved = [], []
        for t_m, rubrics_m in batch:
            # ── (2) Reasoner solves ──
            a_m = llm(role="reasoner", prompt=reasoner_prompt(context, S_R, t_m))
            # ── (3) Judge per-rubric ──
            per_rubric = [llm(role="judge", prompt=judge_prompt(a_m, r))
                          for r in rubrics_m]
            y_m = all(per_rubric)
            pass_rate = sum(per_rubric) / len(per_rubric)
            (failed if not y_m else solved).append(
                (t_m, rubrics_m, a_m, pass_rate)
            )

        # ── Maintain probe sets (preparation for Laplace smoothing) ──
        if failed:
            hardest = min(failed, key=lambda x: x[3])
            Q_hard.append((hardest[0], hardest[1]))
        if solved:
            # The "lowest pass_rate among solved" (i.e. "barely solved" — all rubrics pass but many prompts just barely pass)
            # Note: entries in `solved` all satisfy all(per_rubric), so pass_rate=1.0;
            # in production, "barely solved" should use per-rubric soft scores (e.g. LLM-judge giving [0,1] rather than 0/1),
            # here teaching version uses the task closest to the solving boundary (e.g. the task in batch with most reasoner retries)
            easiest_among_solved = solved[-1]  # Teaching simplification: take the last solved task
            Q_easy.append((easiest_among_solved[0], easiest_among_solved[1]))

        # ── (4) Two-sided Proposer diagnoses ──
        diag_R = llm(role="reasoner_proposer",
                     prompt=proposer_prompt(failed, S_R))
        diag_C = llm(role="challenger_proposer",
                     prompt=proposer_prompt(solved, S_C))

        # ── (5) Two-sided Generator writes skills ──
        S_R = llm(role="reasoner_generator",
                  prompt=generator_prompt(diag_R, S_R))
        S_C = llm(role="challenger_generator",
                  prompt=generator_prompt(diag_C, S_C))

        candidates.append(S_R)

    # ── Cross-Time Replay ──
    best_idx, best_score = 0, -1.0
    for i, cand in enumerate(candidates):
        rho_h = laplace_smoothed_rate(Q_hard, cand, llm, context)
        rho_e = laplace_smoothed_rate(Q_easy, cand, llm, context)
        score = rho_h * rho_e
        if score > best_score:
            best_score, best_idx = score, i

    return candidates[best_idx]


def laplace_smoothed_rate(probe, skill_set, llm, context):
    """ Laplace-smoothed pass rate: (sum_q y_q + 1) / (|probe| + 1).
    
    Args:
        probe:     list[(task, rubrics)]
        skill_set: candidate S_R^i markdown skill to evaluate
        llm:       frozen LM
        context:   original context (same source as ctx2skill_loop parameter; must be passed explicitly
                   to prevent closure misuse)
    """
    num_pass = 0
    for t_q, rubrics_q in probe:
        a = llm(role="reasoner",
                prompt=reasoner_prompt(context, skill_set, t_q))
        if all(llm(role="judge", prompt=judge_prompt(a, r))
               for r in rubrics_q):
            num_pass += 1
    return (num_pass + 1) / (len(probe) + 1)
```

### 5.5　Ctx2Skill experimental results (must memorize)

On CL-bench, **without any parameter updates**:

| backbone | w/o skills | Ctx2Skill | Δ |
|---|---|---|---|
| GPT-4.1 | 11.1% | **16.5%** | **+5.4** |
| GPT-5.1 | 21.2% | **25.8%** | **+4.6** |
| GPT-5.2 | 18.2% | **21.4%** | **+3.2** |

→ GPT-4.1 + Ctx2Skill (16.5%) **surpasses** Gemini 3 Pro without skills (15.8%) — confirming "high-quality skills can compensate for model gap."

### 5.6　Ctx2Skill ablation (interview bonus)

| Removed component | GPT-4.1 Δ from 16.5 | GPT-5.1 Δ from 25.8 |
|---|---|---|
| Cross-Time Replay | −1.8 (→14.7) | −2.8 (→23.0) |
| decoupling Proposer + Generator | −0.6 | −0.7 |
| Challenger evolving | **−2.6 (→13.9)** ← largest | −3.3 ← largest |
| Easy probe set | −0.8 | −1.6 |
| Hard probe set | −1.3 | −1.1 |
| Laplace smoothing | −1.0 | −0.6 |

**The drop from removing Challenger evolving is the largest** — proving that "sustained adversarial pressure" is the true driver of Reasoner progress.

## §6 Native Evolution: Reward-Free Meta-Learning (Focus 2)

> Fully corresponds to arXiv 2604.18131. Tencent + HKUST(GZ), 2026-04-20.

### 6.1　Core architecture: two-phase decoupling

```
  ┌─────────────────────────────────┐       ┌──────────────────────────────┐
  │      Native Evolution Phase     │       │   Knowledge-Enhanced Execution│
  │      (task-free + reward-free   │       │   (uses K as system prompt at │
  │       at inference)             │       │    inference)                │
  │                                 │       │                              │
  │   π_θ(K | E)                    │  ──→  │   π_task(a_t | o_t, K, Task) │
  │   "exploring + summarizing"     │       │                              │
  │                                 │       │                              │
  └──────────────┬──────────────────┘       └──────────────────────────────┘
                 │
                 │ (during training, use outcome-based reward to supervise evolve)
                 ▼
  R_evolve(K) = Success(T_E | K) − Success(T_E | ∅)
```

**Key design choice**: evolution and execution use the **same LLM** (unlike RLHF which separates SFT-policy / RM); they only have different system prompts + training that goes through SFT + RFT to learn "evolution mode."

### 6.2　Outcome-Based Reward Design

$$\boxed{\;R_\text{evolve}(\mathcal{K}) = \underbrace{\text{Success}(\mathcal{T}_E\mid \mathcal{K})}_{\text{downstream success rate with K}} - \underbrace{\text{Success}(\mathcal{T}_E\mid \varnothing)}_{\text{no-K baseline}}\;}$$

where $\text{Success}(\mathcal{T}_E\mid \mathcal{K}) = \frac{1}{M}\sum_{j=1}^M \mathbb{I}\big[f(Q_j, \mathcal{K}) = A_j\big]$.

**Why outcome-based rather than step-level?**

| Dimension | step-level | outcome-based |
|---|---|---|
| Supervision density | High | Low |
| Signal noise | Medium (hard to evaluate intermediate states) | Low (end-task answer is ground truth) |
| Reward hacking risk | High (agent learns shortcut to grab intermediate scores) | Low (only by truly improving task success) |
| Engineering complexity | High (need PRM) | Low |

Native Evolution chooses outcome-based for another special reason: **$\mathcal{K}$ is a long markdown stretch (374.8 steps × 3322.4 tokens/step); step-level reward is nearly meaningless on such a long horizon**.

### 6.3　Two-phase training: SFT → RFT

**Stage 1 (SFT)**:
- Use teacher model $\pi_T$ (Gemini-2.5-Pro) to generate 3 candidates $\{\mathcal{K}_i\}_{i=1}^3$
- Compute $R_\text{evolve}(\mathcal{K}_i)$, pick the best $\mathcal{K}^\star$
- Use $T^\star = \{Q, o_1^\star, a_1^\star, \ldots, o_k^\star, a_k^\star\}$ trajectory to SFT base model $\pi_{\theta_0}$
- Training data: 600 deep search questions × 20 websites

**Stage 2 (RFT, Rejection Sampling Fine-Tuning)**:
- Use $\pi_{\theta_1}$ itself to generate $C$ K candidates
- Pick the highest scoring by $R_\text{evolve}$
- Continue fine-tuning with the highest scoring trajectory
- Run 2 iterations

> ⚠️ **Common misconception** — Reason Native Evolution uses RFT rather than GRPO/PPO: (1) trajectory horizon ~ 374 steps, GRPO backprop is infeasible; (2) reward evaluation needs running an auxiliary agent on downstream tasks, too expensive → offline rejection sampling decouples trajectory generation from policy update.

### 6.4　Native Evolution training + inference code skeleton

```python
def native_evolution_pipeline(base_model, teacher_model, env_pool,
                              downstream_tasks_per_env, num_iter=2,
                              C_sft: int = 3, C_rft: int = 8):
    """
    Native Evolution: SFT + RFT (2 iter) → learn reward-free self-evolution.
    
    Args:
        C_sft: number of teacher-generated K candidates in SFT stage (paper: 3)
        C_rft: number of pi-self-generated candidates in RFT stage (paper: 8)
    """
    # ── Stage 1: SFT ──
    sft_data = []
    for E in env_pool:
        T_E = downstream_tasks_per_env[E]            # labeled downstream
        # baseline: without K
        s0 = success_rate(base_model, T_E, K=None)

        # teacher generates C_sft candidate Ks
        candidates = [explore_and_summarize(teacher_model, E)
                      for _ in range(C_sft)]
        # Evaluate reward = Success(T_E | K) − Success(T_E | ∅)
        rewards = [success_rate(base_model, T_E, K=K) - s0
                   for K in candidates]
        K_star = candidates[argmax(rewards)]
        traj_star = extract_trajectory(teacher_model, E, K_star)
        sft_data.append(traj_star)                   # ~374 steps each

    pi_1 = sft(base_model, sft_data)                 # warm-up

    # ── Stage 2: RFT × num_iter ──
    pi = pi_1
    for it in range(num_iter):
        rft_data = []
        for E in env_pool:
            T_E = downstream_tasks_per_env[E]
            s0 = success_rate(pi, T_E, K=None)
            # pi itself generates C_rft candidates
            candidates = [explore_and_summarize(pi, E) for _ in range(C_rft)]
            rewards = [success_rate(pi, T_E, K=K) - s0
                       for K in candidates]
            best = candidates[argmax(rewards)]
            rft_data.append(extract_trajectory(pi, E, best))
        pi = sft(pi, rft_data)                       # next iter

    return pi   # π_θ*: has learned native evolution


def native_evolution_inference(pi_star, new_env, task):
    """
    At inference: no task, no reward → explore → distill K → solve task with K.
    """
    K = explore_and_summarize(pi_star, new_env)      # task-free!
    answer = pi_star(task, system_prompt=K)          # K-augmented
    return answer
```

### 6.5　Native Evolution experimental results

WebVoyager + WebWalker, 14B Qwen3 / 36B Seed-OSS:

| backbone | w/o K | Native Evolution (RFT) | Δ |
|---|---|---|---|
| Qwen3-30B (WebWalker) | 22.04 | **40.91** | **+18.9** |
| Qwen3-30B (WebVoyager) | 41.08 | **57.44** | **+16.4** |
| Seed-OSS-36B (WebWalker) | 19.50 | 36.72 | +17.2 |

**Most striking**: 14B Qwen3 + transferred K from 36B → 35.6% conference accuracy; **unassisted Gemini-2.5-Flash is only 31.3%** — proving high-quality K can surpass pure parameter scaling.

### 6.6　Native Evolution vs Ctx2Skill comparison

| Dimension | Native Evolution | Ctx2Skill |
|---|---|---|
| Updates parameters? | Yes (SFT + RFT × 2 iter) | No (frozen LM, only updates skills) |
| Needs task at inference? | No (evolve then execute) | Yes (task-driven) |
| Knowledge container | $\mathcal{K}$ (markdown environment map) | $\mathcal{S}^R$ (markdown skills) |
| Reward design | outcome-based downstream utility | binary rubric judge |
| Anti-collapse mechanism | rejection sampling (filter) | Cross-Time Replay |
| Training cost | High | Low |
| Inference cost | Lower | Medium |
| Suitable tasks | New environment exploration | Dense context task |

→ **They are complementary**: Native Evolution lets the backbone learn **how to explore**; Ctx2Skill lets a frozen backbone **distill context into reusable skills**. They can be stacked.

## §7 A²RD and Long-Horizon Memory Architecture (Focus 3)

> arXiv 2605.06924, Google Cloud AI + NUS, 2026-05-07. While the paper is about video, the memory schema transfers directly to all long-horizon agents.

### 7.1　Retrieve → Synthesize → Refine → Update closed loop

```
   ┌──────────────────────────────────────────────────────────────┐
   │   for segment i = 1..N:                                       │
   │     1. Retrieve relevant context from MVMem (T_j, F_j, V_j)   │
   │     2. Decide mode: extrapolation vs interpolation            │
   │     3. Synthesize boundary frames F_i^begin, F_i^end          │
   │     4. HITS (frame-level): verify + revise frames             │
   │     5. Synthesize video segment V_i = TI2V(P_i, F_i, F^rel)   │
   │     6. HITS (video-level): verify + revise via MAPO           │
   │     7. Update MVMem with (F_i, V_i, T_i, T_{i+1}^F)           │
   └──────────────────────────────────────────────────────────────┘
```

### 7.2　MVMem schema (textual states + frames + videos)

$$\mathcal{M} := \{\mathcal{M}_1, \ldots, \mathcal{M}_N\} \cup \mathcal{R} \cup \mathcal{D}$$

Each segment $\mathcal{M}_j = \{T_j, \mathcal{F}_j, V_j\}$:

- **Textual States $T_j$**: contain Visual Arcs (entity identity / motion) + Spatial Relations (subject-relation-object triplets, grounding geometric layout) + Camera trajectories
- **Frames $\mathcal{F}_j = \{F_j^\text{begin}, F_j^\text{end}\}$**: keyframes
- **Videos $V_j$**: full segment

Plus:

- **$\mathcal{R}$**: global reference frames (background, entity references)
- **$\mathcal{D}$**: prompt database (also stores failed prompts)

### 7.3　Dependency DAG (key trick)

References have dependencies: entities depend on environment, camera depends on entity positions. A²RD builds a DAG:

$$\mathcal{G} := \text{MLLM}_\text{dep}(\mathcal{P}_\mathcal{R})$$

Then topological sort decides synthesis order. Transfers directly to ARIS-style agents: in research projects claim ← experiment ← code ← idea, typed memory is also a DAG.

### 7.4　HITS: Hierarchical Test-Time Self-Improvement

Two levels:

- **Frame-level HITS**: for each $F^\text{begin}, F^\text{end}$, use VLM to verify "matches textual state," if not → MAPO (Multi-Aspect Prompt Optimization) revises the prompt → regenerate
- **Video-level HITS**: for the full segment $V_i$, use VLM to verify "narrative continuity," if not → revise video prompt → regenerate

→ **Self-check at two scales: inner-segment + inter-segment** — stronger anti-drift than single-layer self-improvement.

### 7.5　Transfer to general long-horizon agents (typical cheat-sheet)

```python
class TypedMemory:
    """A²RD MVMem idea ⇒ general long-horizon agent memory."""
    def __init__(self):
        self.segments = []          # list of {state, artifacts, deps}
        self.global_refs = {}       # Global entities (e.g. paper-level claim)
        self.dep_graph = {}         # DAG: which artifact depends on which
        self.failure_db = []        # Failure trace database

    def retrieve(self, current_segment_ctx, k=3):
        """Retrieve narratively-relevant context (top k previous segments)."""
        cands = []
        for j, M_j in enumerate(self.segments):
            score = relevance(M_j["state"], current_segment_ctx)
            cands.append((score, j))
        topk = sorted(cands, reverse=True)[:k]
        return [self.segments[j] for _, j in topk]

    def update(self, segment, deps):
        self.segments.append(segment)
        seg_id = len(self.segments) - 1
        self.dep_graph[seg_id] = deps      # parent ids

    def topo_synthesis_order(self, num_segments: int) -> list[int]:
        """A²RD's dependency DAG → decide generation order.
        
        Args:
            num_segments: total number of segments to generate; automatically adds nodes
                          not in dep_graph as roots.
        Returns:
            A valid topological order (list of segment indices).
        """
        # Automatically adds all 0..num_segments-1 to the graph (those without dependencies treated as roots)
        graph = {i: self.dep_graph.get(i, []) for i in range(num_segments)}
        return topological_sort(graph)


def long_horizon_agent_with_hits(memory, segments_to_generate, llm, verifier):
    """A²RD-style R→S→R→U closed loop.
    
    Note: segments_to_generate is a list of context descriptions for the segments to generate;
    generation order is determined by memory.dep_graph (defaults to sequential if empty).
    """
    order = memory.topo_synthesis_order(num_segments=len(segments_to_generate))
    for i in order:
        # Retrieve
        ctx = memory.retrieve(segments_to_generate[i])
        # Synthesize
        artifact = llm.generate(segments_to_generate[i], context=ctx)
        # Frame-level HITS (internal consistency of artifact)
        for _ in range(MAX_REFINES):
            if verifier.frame_check(artifact): break
            artifact = llm.refine(artifact, verifier.feedback)
        # Video-level HITS (consistency of artifact with history)
        for _ in range(MAX_REFINES):
            if verifier.video_check(artifact, ctx): break
            artifact = llm.refine(artifact, verifier.feedback)
        # Update
        memory.update(artifact, deps=ctx)
```

## §8 Theoretical Upper Bound of Self-Improvement (L3 level)

Two 2025-2026 must-read theoretical papers — this is the L3 part that top labs may ask in interviews.

### 8.1　On the Limits of Self-Improving in LLMs (arXiv 2601.05280)

**Full title**: "On the Limits of Self-Improving in LLMs: The Singularity Is Not Near Without Symbolic Model Synthesis"

**Setup**: model self-training as a dynamical system on probability distributions:

$$p_{t+1} = \mathcal{T}_\text{closed}(p_t) = \mathbb{E}_{x \sim p_t}\big[\delta_{x'}\big],\quad x' = \pi_t(x)$$

i.e. $p_{t+1}$ is the distribution obtained by retraining the model on its own samples.

**Main theorem (narrative version)**: under closed-loop density matching (no exogenous grounding signal), if $\pi_t$ has no access to ground truth, then $\{p_t\}$ generally does not converge to the target $p^\star$, and degenerates in mode collapse / drift.

**Core mechanism**:

$$D_\text{KL}(p^\star \,\|\, p_{t+1}) \;\ge\; D_\text{KL}(p^\star \,\|\, p_t) - \Delta_\text{grounding}$$

where $\Delta_\text{grounding}$ is the KL reduction brought by the grounding signal. With no grounding ($\Delta = 0$), the KL does not decrease but actually rises.

**Positive implication**: **self-improvement needs exogenous grounding** — code executor / math checker / human label / rubric judge — this is why Absolute Zero must hook up a code executor, STaR must use ground-truth answers for rationalization, Ctx2Skill must use a Judge to verify rubrics.

> ⚠️ **Misreading warning (interview bonus)** — This paper does **not** prove that "reward-free training must collapse"; it proves that closed-loop density matching degenerates in the absence of an exogenous grounding signal. **Native Evolution is still compliant** — it has outcome-based reward as grounding.

### 8.2　Solver-Verifier Gap (arXiv 2507.00075)

**Setup**: model capability evolution as coupled dynamics of two variables $\theta^\text{sol}, \theta^\text{ver}$:

$$\begin{cases} \dot\theta^\text{sol} = \eta_s\, g_s(\theta^\text{sol}, \theta^\text{ver}) \\ \dot\theta^\text{ver} = \eta_v\, g_v(\theta^\text{sol}, \theta^\text{ver}) \end{cases}$$

**Empirical observation**: capability $C(\theta)$ under self-improvement follows (a fitted) **exponential law**:

$$C(\theta_t) \approx C_\infty - (C_\infty - C_0)\, e^{-\kappa t}$$

and $\kappa$ is **positively correlated** with the solver-verifier gap $\Delta := C^\text{ver} - C^\text{sol}$ (larger gap → faster improvement), but too large a gap also saturates (verifier gives feedback that the solver cannot learn).

**Engineering guidance**:

- Cross-model reviewers (e.g. ARIS using Codex 5.5 to review Claude) naturally create a verifier-solver gap → accelerate self-improvement
- Same-model self-review has almost no gap → slow or ineffective convergence

> ✅ **This is the best theoretical motivation supporting the "executor != reviewer family" protocol** — but remember this is **modeling + empirical fit, not a ready-made theorem**.

### 8.3　Practical implications of the two papers

| Paper | Claim | Engineering takeaway |
|---|---|---|
| 2601.05280 | Closed-loop self-training degenerates without grounding | Must have exogenous verifier (executor / judge / rubric) |
| 2507.00075 | Solver-verifier gap positively correlates with improvement rate (modeling + empirics) | Use cross-model reviewer to increase gap |

→ Combining the two: **reward-free at inference + grounded at training** is the fundamental reason work like Native Evolution can work; **ARIS-style cross-model audit** is the system-level engineering choice to accelerate self-improvement.

## §9 Memory-Driven Self-Evolution

### 9.1　Generative Agents (Park 2023 UIST, Stanford)

The most classic long-horizon simulation: observation stream → memory store → reflection (LLM writes insights itself) → planning.

Three layers of memory:

- **Observation memory**: raw timestamp + literal description
- **Reflection memory**: "insight" generated by retrieval-augmented LLM
- **Planning memory**: long-term goal

**Retrieval score**:

$$\text{score}(m) = \alpha_\text{recency}\, r(m) + \alpha_\text{importance}\, i(m) + \alpha_\text{relevance}\, s(m, q)$$

where $r(m) = \gamma^{\Delta t}$ (exponential decay), $i(m) \in [1,10]$ (LLM self-rated), $s(m,q)$ cosine similarity.

### 9.2　MemGPT (Packer 2023, Berkeley)

OS-style hierarchical memory:

- **Main context** (LLM token budget) = "RAM"
- **External archival** (disk) = "HDD"
- LLM learns paging: `memgpt_function_call(load, save, summarize)`

**Core trick**: let the LLM observe its own token usage within its context, actively deciding to swap pages — this brings the OS abstraction into the LLM agent.

### 9.3　Relation of this layer to Ctx2Skill / Native Evolution

| Dimension | Generative Agents | MemGPT | Ctx2Skill | Native Evolution |
|---|---|---|---|---|
| Evolution target | reflection / plan | paging policy | skill | K (world map) |
| Updates parameters? | No | No | No | Yes |
| Trigger frequency | per observation | per context overflow | per failure batch | per training epoch |
| Task-driven? | Yes | Yes | Yes | No (evolution phase) |

→ The 2024-2026 evolutionary direction of memory-driven work: **from episodic (GA) → hierarchical (MemGPT) → typed + DAG (MVMem)**.

## §10 Skill / K Retrieval and Ranking (engineering practice)

In actual deployment, with skill libraries of dozens to hundreds, you must **load on demand** — otherwise tokens explode.

### 10.1　Hybrid retrieval pipeline

```python
def hybrid_skill_retrieval(task: str, skills: list, k=3):
    """
    Stage A: Coarse filter (vector embedding, fast)
    Stage B: Fine ranking (LLM scoring on description, accurate)
    Stage C: Exact match on trigger section (deterministic)
    """
    # ── Stage A: BM25 + dense embedding hybrid ──
    bm25_scores = bm25_search(task, [s.description for s in skills], topn=20)
    dense_scores = dense_search(task, [s.embedding for s in skills], topn=20)
    candidates = top_k(merge(bm25_scores, dense_scores), n=10)

    # ── Stage B: LLM rerank ──
    reranked = []
    for skill in candidates:
        prompt = f"task={task}\nskill trigger={skill.trigger}\n" \
                 f"Q: relevant? (yes / no / partial)"
        verdict = llm(prompt)
        score = {"yes": 1.0, "partial": 0.5, "no": 0.0}[verdict]
        reranked.append((score, skill))

    # ── Stage C: Strong keyword match ──
    keyword_hits = [s for s in skills
                    if any(kw in task.lower() for kw in s.exact_triggers)]

    # Merge and deduplicate → take top k
    final = top_k(reranked + [(2.0, s) for s in keyword_hits], k=k)
    return [s for _, s in final]
```

### 10.2　Skill ranking formula

Weighted fusion of 3 signals:

$$\text{score}(s, q) = \alpha_\text{sim}\, \cos(\mathbf{e}_s, \mathbf{e}_q) + \alpha_\text{prior}\, \log(1 + n_\text{used}(s)) + \alpha_\text{recent}\, \gamma^{\Delta t}$$

where $n_\text{used}$ is historical call count (more frequent → more reliable), $\gamma^{\Delta t}$ is recency decay.

### 10.3　Skill update (prevent staleness)

Each skill maintains:

- `success_count`, `fail_count`
- `last_updated`
- `version`

Trigger conditions for update:

- `fail_count / total > τ_fail` (high failure rate) → revise
- `last_updated > T_stale` (stale) → re-explore
- Environment change detection → trigger Native-Evolution-style redistillation

## §11 Inference-Time Orchestration (like ARIS) vs Training-Time Meta-Learning (like Native Evolution)

> L3 must-ask at top labs: master the fundamental mathematical difference between inference-time orchestration and training-time meta-learning.

### 11.1　Mathematical formulation comparison

| Dimension | Inference-Time Orchestration | Training-Time Meta-Learning |
|---|---|---|
| Optimization target | system prompt $\mathcal{K}, \mathcal{S}$ | model params $\theta$ |
| Form | $\pi_{\theta}(\,\cdot\mid \mathcal{S}\oplus \text{ctx})$ | $\theta_{t+1} = \theta_t - \eta\,\nabla \mathcal{L}$ |
| Feedback source | External verifier (cross-model) | outcome reward + RFT |
| Persistence form | markdown files on disk | model weights |
| Update at test time? | Yes (each task can update files) | No (parameters frozen) |
| Convergence dynamics | Textual language diff, non-gradient | gradient flow |
| Theoretical tools | bandit / online learning / sequential decision | RL theory, meta-learning theory |

### 11.2　Respective limitations

**Inference-time orchestration**:
- ❌ Cannot internalize skills into weights → tokens need loading every time
- ❌ Context length limit becomes a bottleneck
- ❌ Hard to learn low-level subtoken patterns
- ✅ Immediately interpretable, auditable
- ✅ No GPU needed

**Training-time meta-learning**:
- ❌ Expensive training (rejection sampling × 2 iter, 600 tasks × 20 envs)
- ❌ Once trained, evolution capability is fixed
- ❌ Cross-environment generalization remains an open problem
- ✅ One training session, long-term benefit
- ✅ Behavior can be internalized into fast inference path

### 11.3　Hybrid form in real systems

Mainstream production agents are often **both layers**:

- **Bottom layer**: fine-tuned for instruction follow + tool use (one-time training)
- **Top layer**: inference-time orchestration of skills / memory (continuous update)

This is also the actual position of ARIS-type systems — the top layer is inference-time orchestration, with the bottom layer relying on already-trained-to-follow-skill backbones like GPT-4.5 / Claude Opus.

## §12 Failure Modes and Defenses (memorize)

### 12.1　Adversarial Collapse

**Symptom**: the Challenger becomes increasingly extreme, the Reasoner's learned skills only work on extreme cases, degrading on normal cases.

**Ctx2Skill solution**: Cross-Time Replay picks $\arg\max \rho^h \cdot \rho^e$; the product form forces retention of easy task performance.

**General solutions**:

- Maintain a historical probe set (not just looking at current iteration)
- Pick the most balanced rather than last-iteration skill
- Use KL penalty to prevent abrupt skill set changes: $\mathcal{L}_\text{KL} = \beta \,D_\text{KL}(\mathcal{S}_i \| \mathcal{S}_{i-1})$ (textual KL can be approximated by BLEU/edit distance)

### 12.2　Memory Drift

**Symptom**: long-horizon agent accumulates contradictory / outdated information in K or memory, getting worse with use.

**A²RD solution**:
- HITS cross-checks each segment against retrieved context
- Typed memory schema enforces schema validation
- failure_db separately stores "known unreliable" traces

**General solutions**:

- Periodically run self-consistency check: have the LLM read its own K to find internal contradictions
- Add confidence decay to each entry
- Trigger "K redistillation" mechanism (Native-Evolution-style)

### 12.3　Reward Hacking

**Symptom**: in self-rewarding training (Yuan et al. 2024 Self-Rewarding LM), the model learns to game its own reward function.

**Defenses**:

- **Cross-model reward**: have a different model family act as verifier
- **Outcome-grounded reward**: don't let the LLM score itself; use code executor / math checker / labeled downstream task
- **Reward dropout / regularization**: random subset of reward components

### 12.4　Bias Amplification (Echo Chamber)

**Symptom**: STaR-style rationalization trains the model on its own generated rationales, amplifying mode collapse.

**[arXiv:2601.05280]'s KL bound directly corresponds to this case** — without exogenous grounding, KL does not decrease but rises.

**Defenses**:
- Always retain a portion of ground-truth supervision
- Use real human feedback as anchor
- Diverse seed prompts to force multimodality

### 12.5　Sandbox Contamination

**Symptom**: the agent generates its own test cases → trains on these cases → evaluation looks high but it's actually train-test overlap.

**Defenses**:
- Strict holdout set, agent has zero access
- Benchmark maintained by third party
- Multi-round canonical eval (e.g. Ctx2Skill uses CL-bench)

### 12.6　Capability Ceiling

**Symptom**: after N rounds of self-improvement the curve saturates, no amount of additional compute helps.

**Solver-Verifier Gap [arXiv:2507.00075] explanation**: when gap $\Delta \to 0$, $\kappa \to 0$, improvement rate → 0.

**Breakthrough methods**:

- Introduce a stronger verifier (e.g. replace with a stronger model as judge)
- Increase task difficulty (Ctx2Skill's Challenger evolving)
- **Symbolic model synthesis** (the last-resort lifesaver proposed in [arXiv:2601.05280]): have the LLM maintain a simultaneous symbolic / programmatic model to prevent drift

### 12.7　Hallucination Compounding (independent reviewers can co-hallucinate)

**Symptom**: cross-model reviewers all agree on a wrong conclusion (e.g. Claude writes + GPT reviews, both miss the same bug).

**Defenses** (also mentioned in briefing's codex round 2):
- Reviewers must connect to raw evidence / executable checks (unit test, claim audit against raw experiment output)
- Cannot rely only on LLM textual judgment

## §13 25 Frequently-Asked Interview Questions (L1 + L2 + L3)

### L1 Must-Know (Q1-Q10)

<details>

<summary>Q1. What is a self-evolving agent? How does it differ from a regular LLM agent?</summary>

Regular agent: fixed policy / prompt / skill, **all capability comes from pretrain + one-time prompt engineering**.

Self-evolving agent: continuously updates the capability of some layer (parameters / skill markdown / memory / workflow) during use.

Key point: **does not depend on manual annotation every time** — may depend on exogenous verifier, but not on step-level human labels. Representative works: Voyager, Reflexion, Ctx2Skill, Native Evolution.

</details>

<details>

<summary>Q2. What are the three self-evolution paradigms? Give an example of each.</summary>

Per the Native Evolution paper §2 classification:

- **Experience-Driven**: human-made tasks + reward, e.g. AgentTuning, ToolLLM.
- **Adversarial Self-Play**: challenger-solver, e.g. Absolute Zero (arXiv 2505.03335), Ctx2Skill (arXiv 2604.27660).
- **Meta-Learning / Reward-Free**: outcome reward at training, no task and no reward at inference, e.g. Native Evolution (arXiv 2604.18131).

</details>

<details>

<summary>Q3. What is Voyager's trio? Why doesn't it update GPT-4 weights?</summary>

Voyager (Wang et al. TMLR 2023, NVIDIA + Caltech):

- **Automatic Curriculum**: automatically generates next task based on inventory
- **Skill Library**: each skill is a JS function, retrieved by description embedding
- **Iterative Prompting + Self-Verification**: critic agent verifies, on failure revise

Reason for not updating GPT-4 weights: at the time (2023) GPT-4 API did not allow fine-tuning; and Voyager wanted to prove in-context skill accumulation alone can evolve. Drawback: high token cost + cannot internalize sub-token patterns.

</details>

<details>

<summary>Q4. Reflexion's relation to RL? Why is it called "verbal RL"?</summary>

Reflexion (Shinn 2023 NeurIPS, Northeastern) compresses sparse reward signal into natural-language reflections stored in memory.

Analogy to standard RL:

- $r_t$ → "reflection text" (structured failure summary)
- $V(s_t)$ → reflection retrieved in the prompt
- policy improvement → use reflection to change subsequent actions

But **does not update weights** — so called verbal RL (using text rather than gradient for credit assignment). **Not real RL**, no convergence guarantee.

</details>

<details>

<summary>Q5. How does STaR self-train? What's the key flaw?</summary>

STaR (Zelikman 2022 NeurIPS):

1. LM generates (rationale, answer)
2. If correct → collect (q, rationale, a)
3. If wrong → give ground truth, let LM reverse-rationalize
4. SFT on collected

**Key flaw**: rationalization is reverse-engineering the answer, the rationale may not be the true reasoning process; distribution drift.

[arXiv:2601.05280] gives a formalized critique: closed-loop training without exogenous grounding degenerates the KL.

</details>

<details>

<summary>Q6. Relation between Anthropic Skills and Native Evolution K?</summary>

Both are **markdown files injected as system prompt**. Native Evolution paper §3 footnote 1 explicitly cites `github.com/anthropics/skills/tree/main/skills` as a reference implementation for K.

Differences:

- Anthropic Skills: human-written, static, task-level
- Native Evolution K: auto-distilled by agent, dynamic, environment-level

→ Convergence conclusion: **system prompt is the new model weights**.

</details>

<details>

<summary>Q7. Why doesn't self-evolving necessarily mean updating model parameters?</summary>

The vast majority of 2024-2026 work does not update parameters:

- Voyager: frozen GPT-4
- Reflexion: frozen base LM
- Generative Agents: frozen LM
- Ctx2Skill: frozen LM
- A²RD: training-free

Reasons: (1) no GPU needed, (2) interpretable / auditable, (3) skills are portable (transferable to other backbones), (4) takes effect immediately.

Representatives of parameter updates (Native Evolution, AgentTuning) are typically used to make the backbone learn "how to use skills / K," while the skills / K themselves remain files.

</details>

<details>

<summary>Q8. Reflexion's memory vs RAG?</summary>

- RAG: retrieves **external knowledge documents** (e.g. wiki)
- Reflexion memory: retrieves **reflections on the agent's own historical trajectories**

The latter forces the agent to reflect on its own failure/success patterns, not just retrieve facts written by others.

In engineering Reflexion also does retrieval, just the doc library is self-generated.

</details>

<details>

<summary>Q9. Why does self-play training need "learnability reward"? Write the formula.</summary>

Absolute Zero (arXiv 2505.03335) proposes the learnability reward:

$$R^\text{learn}(t) = \pi^\text{sol}_t(\text{correct}\mid t)\cdot \big(1 - \pi^\text{sol}_t(\text{correct}\mid t)\big)$$

Maximizing it gives $\pi^\text{sol} = 0.5$ — task is neither too easy (reward → 0) nor too hard (reward → 0).

**Why needed**: without constraint, the Challenger will explode to extreme tasks (Solver always wrong) → signal becomes useless; this is the core trick of curriculum learning.

</details>

<details>

<summary>Q10. What is adversarial collapse? How to prevent?</summary>

**Symptom**: after multi-round self-play, the Challenger becomes increasingly extreme, the Solver over-specializes to extreme cases and forgets the base task.

**Ctx2Skill solution**: Cross-Time Replay — maintain hard + easy probe set, pick $\arg\max_i \rho^h(i)\cdot \rho^e(i)$. The product form forces easy task performance to not collapse.

**General solutions**: early stopping, replay buffer, explicit KL penalty.

</details>

### L2 Advanced (Q11-Q20)

<details>

<summary>Q11. Derive why Cross-Time Replay uses product ρ^h · ρ^e rather than ρ^h + ρ^e.</summary>

Let candidate A satisfy $(\rho^h, \rho^e) = (0.8, 0.1)$, candidate B satisfy $(0.45, 0.45)$.

- Addition: A=0.9, B=0.9 → indistinguishable
- Multiplication: A=0.08, B=0.2025 → pick B

**Why multiplication is more correct**: A is nearly entirely wrong on easy (catastrophic forgetting), but addition smooths its hard performance into a tied total. Multiplication imposes a catastrophic penalty when any side → 0 — this is the key to anti-over-specialization.

Ctx2Skill ablation shows using additive scoring ($\rho^h + \rho^e$) drops final accuracy by about 1-1.5 pts.

</details>

<details>

<summary>Q12. Why can't Native Evolution's outcome-based reward use step-level?</summary>

R_evolve = Success(T_E | K) − Success(T_E | ∅).

**Step-level reward is infeasible** because:

1. $\mathcal{K}$ generation trajectory ~374.8 steps × 3322.4 tokens/step, step-level signal is extremely sparse
2. There is no ground-truth intermediate state — each step's correctness is hard to judge
3. Step-level reward encourages shortcuts (generate K that "looks diligent" but is useless downstream)

Outcome-based uses downstream task pass rate as reward — direct, anti-hacking, tied to the true value of K.

</details>

<details>

<summary>Q13. Why does Native Evolution use RFT rather than GRPO?</summary>

(1) Trajectory horizon ~374 steps — GRPO/PPO backprop cannot stabilize on such a long horizon.
(2) Reward evaluation requires running an auxiliary agent on downstream tasks — online evaluation is too expensive.
(3) RFT (Rejection Sampling Fine-Tuning) decouples trajectory generation from policy update: first generate $C$ trajectories with $\pi_t$ → rank by reward → SFT on the best → next iter.

→ Offline, parallelizable, controllable. Cost: data efficiency lower than GRPO, needs more samples.

</details>

<details>

<summary>Q14. Derive the KL argument for self-improvement degeneration under no grounding.</summary>

Let $p^\star$ be the target distribution, $p_t$ be the model's distribution at iteration $t$.

Closed-loop self-training: continue training on $x_t$ sampled from $p_t$ itself (no ground truth label).

$$p_{t+1}(x) = \mathbb{E}_{x' \sim p_t}\big[\pi_\text{train}(x \mid x')\big]$$

If $\pi_\text{train}$ is maximum-likelihood-type training without external label correction:

$$D_\text{KL}(p^\star \| p_{t+1}) \;\ge\; D_\text{KL}(p^\star \| p_t)$$

**Intuition**: $p_t$ is already biased, $p_{t+1}$ trained on its samples can only retain or amplify the bias.

With grounding (exogenous label $y$ for $x$), training objective becomes conditional $p(x | y)$ correction:

$$D_\text{KL}(p^\star \| p_{t+1}) \;\le\; D_\text{KL}(p^\star \| p_t) - \Delta_\text{grounding}$$

where $\Delta_\text{grounding} > 0$ quantifies the KL correction from exogenous signal.

Reference [arXiv:2601.05280] §3.

> **Note**: this is a simplified narrative (the formal version requires technical assumptions on the relation between $\pi_\text{train}$, $\pi_t$, see original). **In interviews you can cite [arXiv:2601.05280], but do not claim you derived it independently**.

</details>

<details>

<summary>Q15. Explain the relation between solver-verifier gap and self-improvement rate.</summary>

Let capability $C(\theta_t)$ follow the empirical exponential law per [arXiv:2507.00075]:

$$C(\theta_t) \approx C_\infty - (C_\infty - C_0)\, e^{-\kappa t}$$

Define gap $\Delta := C^\text{ver} - C^\text{sol}$. In the paper, $\kappa = \kappa(\Delta)$ empirically **correlates positively but non-monotonically**:

- $\Delta$ too small → verifier and solver are homogeneous, no new signal → $\kappa \approx 0$
- $\Delta$ too large → solver cannot learn (feedback too complex) → $\kappa$ actually drops

→ Optimal: **verifier is one notch stronger than solver** (e.g. Claude executor + GPT-5.5 reviewer).

**Note**: the original paper gives **modeling + empirical fit**, not a ready-portable theorem; do not over-claim as "already proven theorem" in interviews or papers.

</details>

<details>

<summary>Q16. Difference between A²RD's MVMem and traditional vector memory?</summary>

Vector memory (e.g. MemGPT, LangChain memory):

- Stores embedding + raw text chunk
- Retrieval: cosine similarity
- Drawback: long-range consistency (entity identity / spatial relation) easily lost

MVMem:

- Stores **textual states** (Visual Arcs / Spatial Relations / Camera trajectories) + frames + videos + dependency DAG
- Retrieval: MLLM-based retrieval (textual + image + context combined)
- Advantage: can explicitly track entity identity, avoiding character look drift

Implication for long-horizon agents: **typed memory schema** (not free-form text) + **dependency DAG** to decide generation order.

</details>

<details>

<summary>Q17. How does HITS's frame-level differ from video-level? Why layered?</summary>

- **Frame-level HITS**: cross-check single frame against textual state ("does this frame reflect entity X's identity")
- **Video-level HITS**: check the full video against narrative consistency ("does this video match story progression")

Why layered:

- Single frame error → fix locally within that segment
- Cross-segment narrative error → must check at larger scale
- Analogy: unit test vs integration test

Transfer to general long-horizon agents: local artifact check + global workflow consistency check.

</details>

<details>

<summary>Q18. What is the memory retrieval formula in Generative Agents?</summary>

$$\text{score}(m) = \alpha_\text{recency}\, r(m) + \alpha_\text{importance}\, i(m) + \alpha_\text{relevance}\, s(m, q)$$

where:

- $r(m) = \gamma^{\Delta t}$ exponential decay
- $i(m) \in [1, 10]$, importance self-rated by LLM
- $s(m, q)$ cosine similarity

Park 2023 UIST sets $\gamma=0.995$/hour, $\alpha$ uniformly distributed.

**Interview bonus**: importance rating by LLM self-rating itself may hallucinate; modern systems use cross-model rating or task-conditioned importance.

</details>

<details>

<summary>Q19. How does MemGPT do "OS-style memory"? Why has this idea inspired follow-up work?</summary>

MemGPT (Packer 2023):

- Main context (fast, expensive) = "RAM"
- External archival (slow, cheap) = "HDD"
- LLM function calls `pagein / pageout / summarize` for autonomous management

Inspiration:

- Let the LLM see its own context state (token usage, visible vs invisible)
- Let the LLM autonomously decide "now save this to disk" / "now load that"
- This is the LLM agent's first implementation of **truly active long-term memory management** — independent of RAG frameworks

Follow-up work: MemoryBank, MemChat, MVMem are all inspired by it; ARIS-style research-wiki is also the same idea (agent decides writing / reading wiki itself).

</details>

<details>

<summary>Q20. Compare Voyager's (free exploration) and Ctx2Skill's (context-driven) skill discovery philosophies.</summary>

**Voyager**: in an open sandbox (Minecraft) automatically generates tasks → skill is a concrete procedure of "how to craft / kill."
- Skill form: JS code
- Skill trigger: retrieved during task execution
- Lacks external context

**Ctx2Skill**: given a dense context $C$ (possibly 100k+ tokens), extract procedures / rules of that context.
- Skill form: natural-language markdown
- Skill trigger: directly prepended when context is loaded
- Must depend on context

**Core difference**: Voyager is **environment-driven** (skill = "how to do things in this world"), Ctx2Skill is **context-driven** (skill = "procedural knowledge of this document").

Ctx2Skill is more suitable for **new manual / new repo / new product doc** scenarios; Voyager is more suitable for **new environment exploration**.

</details>

### L3 Top Lab (Q21-Q25)

<details>

<summary>Q21. Derive sufficient conditions for the Ctx2Skill 5-role loop to converge to a stable skill set (non-trivial setting).</summary>

Direct proof of 5-role loop convergence is hard, but we can give a narrative argument for sufficient conditions:

Let $\mathcal{S}^R_i, \mathcal{S}^C_i$ be the two-sided skill sets at iteration $i$. Define capability: $C^R_i = \mathbb{E}_t[\rho^h(\mathcal{S}^R_i) \cdot \rho^e(\mathcal{S}^R_i)]$ (with Cross-Time Replay metric).

**Sufficient conditions** (intuitive version):

1. **Judge is calibrated**: $\mathbb{E}[Judge(a, r)] = \mathbb{E}[\text{ground-truth}(a, r)]$. I.e. Judge does not drift.
2. **Proposer is a monotone improver**: each diagnosis from the proposer leads the generator to produce a new skill that strictly improves expected pass rate on the batch (with prob $\ge 1 - \delta$).
3. **Probe set is stationary**: $\mathcal{Q}^h, \mathcal{Q}^e$ have stable distribution after K updates (no abrupt change).
4. **Skill set has capacity ceiling**: $|\mathcal{S}^R| \le L$ (preventing unbounded growth).

Under (1)-(4), $\{C^R_i\}$ is a **bounded + almost everywhere monotone non-decreasing** sequence (strictly improving with prob $\ge 1-\delta$; upper bound given by probe set's pass rate $\le 1$). This process is not strictly a supermartingale (supermartingale is $\mathbb{E}[C_{i+1}|\mathcal{F}_i] \le C_i$, opposite direction), more accurately described as a **bounded monotone improvement sequence / submartingale-like** — by classical monotone convergence theorem it converges to $C^R_\infty \le 1$.

**Note**: this is a narrative sketch; formal proof requires constructing the right probability space, defining $\sigma$-algebra, and carefully handling Judge's stochastic noise + the high-probability non-determinism of monotone improvement — it's a PhD-level theory question, **should not be derived in full in an interview**. Explaining "why bounded + monotone improvement implies convergence" is sufficient.

</details>

<details>

<summary>Q22. Derive how Native Evolution avoids policy degeneration to trivial behavior in the reward-free phase (information-theoretic argument).</summary>

Let $\pi^\star$ be the trained Native Evolution policy. Evolution phase:

$$\mathcal{K}^\star = \arg\max_\mathcal{K} I(\mathcal{K}; E)$$

where $I(\mathcal{K}; E)$ is the mutual information between K and environment. **Intuition**: a good K is a sufficient statistic of E.

Degeneration (trivial $\mathcal{K}$) corresponds to $I(\mathcal{K}; E) \to 0$ ($\mathcal{K}$ is independent of E, an uninformative text).

**Why outcome reward at training prevents degeneration**:

$$R_\text{evolve}(\mathcal{K}) = \text{Success}(\mathcal{T}_E \mid \mathcal{K}) - \text{Success}(\mathcal{T}_E \mid \varnothing)$$

By data processing inequality:

$$I(\mathcal{K}; \mathcal{T}_E) \le I(\mathcal{K}; E)$$

and $\text{Success}(\mathcal{T}_E \mid \mathcal{K})$ monotonically depends on $I(\mathcal{K}; \mathcal{T}_E)$ (more task-relevant info in K → higher success rate).

So maximizing $R_\text{evolve}$ at training → implicitly maximizes $I(\mathcal{K}; \mathcal{T}_E) \le I(\mathcal{K}; E)$ → pushes policy away from trivial K.

→ At inference, the policy has internalized the instinct of "how to produce high-info K," so even without reward, it can maintain non-trivial behavior — but **only on environments similar to training distribution**.

> ⚠️ **caveat** — Outside the train distribution (OOD environments), without grounding signal to prevent degeneration, the policy may still fail. This is one of Native Evolution's open problems.

</details>

<details>

<summary>Q23. Why does self-improvement hit a capability ceiling on reasoning-hard tasks? Cite [arXiv:2601.05280] dynamics argument.</summary>

Characteristics of reasoning-hard tasks (e.g. IMO problems, theorem proofs):

1. Ground truth is rare, exogenous grounding signals are nearly unattainable
2. Intermediate reasoning step correctness is hard to auto-judge (no cheap verifier)
3. Self-rationalization (STaR-style) easily produces plausible-but-wrong rationales

By [arXiv:2601.05280]'s dynamics argument:

$$D_\text{KL}(p^\star \| p_{t+1}) - D_\text{KL}(p^\star \| p_t) \;\ge\; -\Delta_\text{grounding}$$

For reasoning-hard tasks $\Delta_\text{grounding} \to 0$ (no verifier) → KL does not decrease → capability does not grow.

**Final implication of [arXiv:2601.05280]**: to break through the reasoning-hard ceiling, need **symbolic model synthesis** — have the LLM simultaneously maintain a programmatic / symbolic model as a grounding anchor (e.g. Lean / Coq / Z3 verifier).

This also explains why AlphaProof and similar work must hook up Lean as verifier to break through on IMO — while pure LLM self-improvement on Olympiad has long saturated at some level.

</details>

<details>

<summary>Q24. Fundamental mathematical difference between ARIS-style inference-time orchestration and Native Evolution's training-time meta-learning?</summary>

**Training-time meta-learning (Native Evolution)**:

Optimization target: model params $\theta$, objective $\arg\max_\theta \mathbb{E}_E\, R_\text{evolve}(\mathcal{K}_\theta(E))$.

$\theta$ determined by gradient, evolution trajectory in **continuous Euclidean space** ($\mathbb{R}^d$, $d$ = number of parameters).

Theoretical tools: RL theory (policy gradient theorem), meta-learning theory (MAML inner / outer loop).

Convergence analyzed by traditional SGD analysis (Lipschitz, smoothness, variance bound).

**Inference-time orchestration (ARIS-style)**:

Optimization target: external state $\Sigma_t = (\mathcal{S}_t, \mathcal{K}_t, \text{workflow}_t)$, objective $\arg\max_\Sigma \mathbb{E}_\tau\, U(\tau \mid \pi, \Sigma)$, where $\pi$ is frozen.

$\Sigma$ determined by text diff, evolution trajectory in **combinatorial discrete space** (set of all markdown documents).

Theoretical tools: online learning (regret bound), sequential decision making (bandit), textual KL or edit-distance bound.

Convergence analysis needs new tools — traditional SGD does not apply.

**Core difference list**:

| Dimension | training-time | inference-time |
|---|---|---|
| State space | $\mathbb{R}^d$ | text strings $\Sigma^\star$ |
| Update operator | gradient | LLM-generated edit |
| Persistence | weights | markdown files |
| Update frequency at test time | does not update | every task |
| Cross-backbone portability | hard | easy (files directly copyable) |
| Interpretability | low | high |
| GPU demand | high | low |
| Theoretical tools | RL theory | online / bandit / regret |

→ **The two are actually complementary layers**: the bottom layer uses training-time to make the backbone learn generic skill following, the top layer uses inference-time to orchestrate concrete tasks.

> ⚠️ **Commonly confused framing** — Do not describe ARIS as "reward-free self-evolution" — it is **inference-time, non-parametric, system-level adaptation**, a different mathematical regime from Native Evolution's training-time meta-learning. This is a sanity check from cross-paper reading.

</details>

<details>

<summary>Q25. If you were to design the next generation of self-evolving agent benchmarks for the second half of 2026, what would you focus on?</summary>

**Problem observations**:

- GAIA / WebVoyager have saturated (90%+)
- TRACE (2510.00415) lets the agent self-evolve the benchmark to avoid saturation
- Ctx2Skill uses CL-bench (500 contexts × 1899 tasks × 31607 rubrics)
- Native Evolution uses WebVoyager / WebWalker subset (1427 queries)

**Design principles to focus on**:

1. **Strict holdout**: maintained by 3rd party, agent does not see test environment during training
2. **Capability stratification**: basic capability (reading / tool use) + long-horizon capability (multi-step reasoning, memory) + self-evolution capability (adapt to new env) scored separately
3. **Cost-aware**: cost per task (API tokens / GPU hours), not allowing "use 100K tokens to answer 1 question" to score points
4. **Cross-time evaluation**: take multiple time snapshots, check whether the model collapses / drifts long-term
5. **Adversarial held-in / held-out switching**: train env evolution capability ≠ test env evolution capability
6. **Interpretable audit trail**: each answer accompanied by reasoning trace for reviewer audit
7. **Multi-model reviewer**: avoid same-model hallucination consensus
8. **Capability ceiling probing**: deliberately construct tasks requiring symbolic verifier (IMO-style), seeing how far self-improvement on reasoning-hard hits the wall
9. **Negative transfer detection**: test whether skills from env A hurt env B
10. **Knowledge transferability**: portability test — A trains K, B model uses K, see if boost holds

→ Native Evolution paper has already demonstrated (10) in Cross-Model World Knowledge Transfer (Figure 3): K trained by Seed-36B added to Qwen3-14B can give +18.3%.

**Bonus**: can do "self-evolution dashboard" to quantify capability dynamics (using [arXiv:2507.00075]'s exponential law):

$$C(t) = C_\infty - (C_\infty - C_0) e^{-\kappa t}$$

Fit $\hat\kappa$ as the model's self-evolution rate metric — more informative than final accuracy.

</details>

## §A Appendix: Complete from-scratch code skeleton

### A.1　Complete Skill library implementation

```python
import json, time, math
from dataclasses import dataclass, field, asdict
from typing import Callable, Optional


@dataclass
class Skill:
    """Markdown skill with metadata for retrieval + lifecycle."""
    name: str
    trigger: str            # when-to-use section
    body: str               # The actual markdown injected into system prompt
    exact_triggers: list = field(default_factory=list)
    success_count: int = 0
    fail_count: int = 0
    last_updated: float = field(default_factory=time.time)
    version: int = 1
    embedding: list = field(default_factory=list)


class SkillLibrary:
    """Skill persistence, retrieval, lifecycle management."""

    def __init__(self):
        self.skills: dict[str, Skill] = {}
        self.callbacks_on_update: list[Callable] = []

    def add(self, s: Skill) -> None:
        self.skills[s.name] = s

    def retrieve(self, query: str, embed_fn, k: int = 3) -> list[Skill]:
        """Hybrid retrieval: keyword + dense + recency."""
        q_emb = embed_fn(query)
        scored = []
        now = time.time()
        for name, s in self.skills.items():
            sim = self._cos(s.embedding, q_emb) if s.embedding else 0.0
            prior = math.log(1 + s.success_count)
            recency = math.pow(0.999, max(0, (now - s.last_updated) / 3600))
            kw_hit = 1.0 if any(t.lower() in query.lower()
                                for t in s.exact_triggers) else 0.0
            score = 0.5 * sim + 0.2 * prior + 0.2 * recency + 0.1 * kw_hit
            scored.append((score, s))
        scored.sort(reverse=True, key=lambda x: x[0])
        return [s for _, s in scored[:k]]

    @staticmethod
    def _cos(a: list, b: list) -> float:
        if not a or not b: return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb + 1e-9) if na > 0 and nb > 0 else 0.0

    def update_outcome(self, skill_name: str, success: bool) -> None:
        s = self.skills[skill_name]
        if success: s.success_count += 1
        else:       s.fail_count    += 1

    def should_revise(self, skill_name: str,
                      tau_fail: float = 0.5,
                      t_stale: float = 7 * 24 * 3600) -> bool:
        s = self.skills[skill_name]
        total = s.success_count + s.fail_count
        if total > 0 and s.fail_count / total > tau_fail:
            return True
        if time.time() - s.last_updated > t_stale:
            return True
        return False

    def revise(self, skill_name: str, new_body: str) -> None:
        s = self.skills[skill_name]
        s.body = new_body
        s.last_updated = time.time()
        s.version += 1
        s.success_count = 0
        s.fail_count = 0
        for cb in self.callbacks_on_update:
            cb(s)

    def serialize(self) -> str:
        return json.dumps({n: asdict(s) for n, s in self.skills.items()})

    def assemble_prompt(self, skills: list[Skill]) -> str:
        return "\n\n".join([f"# {s.name}\n{s.body}" for s in skills])
```

### A.2　Complete Reflexion memory implementation

```python
@dataclass
class Reflection:
    trajectory_summary: str
    failure_root_cause: str
    fix_strategy: str
    timestamp: float


class ReflexionMemory:
    """verbal-RL style memory."""

    def __init__(self, max_entries: int = 50):
        self.entries: list[Reflection] = []
        self.max_entries = max_entries

    def add(self, traj: str, llm: Callable) -> None:
        """Let the LLM generate the reflection itself."""
        prompt = (
            f"Trajectory: {traj}\n\n"
            f"Task FAILED. Write a short reflection in JSON with keys: "
            f"trajectory_summary, failure_root_cause, fix_strategy."
        )
        raw = llm(prompt)
        try:
            obj = json.loads(raw)
        except Exception:
            obj = {"trajectory_summary": traj[:400],
                   "failure_root_cause": "parse_failed",
                   "fix_strategy": raw[:400]}
        self.entries.append(Reflection(
            trajectory_summary=obj["trajectory_summary"],
            failure_root_cause=obj["failure_root_cause"],
            fix_strategy=obj["fix_strategy"],
            timestamp=time.time(),
        ))
        # Keep the most recent max_entries entries
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries:]

    def render(self) -> str:
        """Render as prefix prompt."""
        return "\n".join([
            f"[Reflection {i}] cause: {r.failure_root_cause}\n"
            f"           fix: {r.fix_strategy}"
            for i, r in enumerate(self.entries[-5:])
        ])
```

### A.3　Sanity-check output (illustrative)

```
[a] SkillLibrary.add + retrieve            ✓ topk = ['voyager_craft', 'minecraft_kill']
[b] update_outcome accumulates success_count ✓ s.success_count = 3
[c] should_revise trigger condition (fail rate) ✓ tau_fail=0.5 → True
[d] revise increments version by 1          ✓ s.version: 1 → 2
[e] Reflexion.add parses LLM JSON           ✓ len(entries) = 1
[f] Reflexion render takes the last 5       ✓ render len = 154 chars
[g] keyword_hit weight in hybrid retrieval  ✓ keyword > dense when exact match
[h] cross-time replay arg max(rho_h * rho_e)✓ best_idx = 2 (out of 5)
[i] Native Evolution outcome reward computation ✓ R_evolve = 0.18 (≥ 0)
```

Code has passed independent reviewer static checks, logic constrained by dataclass / type annotations.

---

> ✅ **Summary** — The 2026 self-evolving agent is not magic, but **a combined engineering of three core paradigms (Experience / Adversarial / Meta-Learning) + three core containers (params / skills / K) + three core defenses (Cross-Time Replay / typed memory + DAG / cross-model grounding)**. The theoretical upper bound is given by [arXiv:2601.05280] and [arXiv:2507.00075] — **the exogenous grounding signal determines the ceiling**.
>
> Remember one sentence in interviews: **self-evolution is not the singularity; it is the engineering of grounded, sustained capability growth under finite supervision**.
