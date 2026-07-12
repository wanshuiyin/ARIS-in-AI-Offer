## §0 TL;DR Cheat Sheet

> 💡 **9 sentences to nail Agentic RL** — RL for LLM agents is the 2024-2026 paradigm pushing reasoning RL into real tool use, the web, code, and GUI (see §1-§9 for derivations + §10 for the 25 frequently-asked questions).

1. **The fundamental difference between Agentic RL and RLHF**: RLHF is single-turn preference alignment with reward from an RM scoring an entire response; **Agentic RL is multi-turn decision-making, state is (obs, history), action is (thought, tool_call), and leans on programmatically verifiable objective outcomes (test-pass, task success, verifier) as the backbone rather than RM-style subjective preference scoring** (though it can still use a learned reward model, e.g. WebRL's ORM — the two are not a strict either/or). Trajectory length grows from RLHF's hundreds of tokens to an agent's thousands or even tens of thousands of tokens, taking credit assignment difficulty up a notch.

2. **Key modifications PPO/GRPO need on agents** (must memorize): the **token mask** must restrict the loss to tokens the agent itself generates — observation tokens (the stdout / search snippet returned by tools) belong to the environment, the policy gradient must not flow there; otherwise the model will try to "teach the tool how to respond" and behavior collapses. GRPO's advantage is more pronounced: on long-horizon trajectories, a value model can hardly learn per-token V (almost all middle rewards are 0); intra-group normalization is a more stable baseline.

3. **Three-tier reward design pyramid**: (a) **Outcome reward** is cheapest and sparsest — final answer/task success 0/1; (b) **Process reward** scores each step, requiring a PRM or step verifier; (c) **Hybrid / shaping** — tool-call shaping (encourage calling the right tool), length penalty (prevent the agent from dragging too long), format reward (strictly constrain output schema). The R1 line uses rule-based outcome reward (math correctness + format), SWE-RL uses test-pass (rule-based), while WebRL trains an ORM to score task success (a learned-RM route, see §6.9) — **outcome-level verification signal (rule-based or learned ORM) + dense format shaping** is the most stable combination empirically in 2025 industry.

4. **Representative early work**: **AgentTuning** (Zeng et al. 2023 arXiv 2310.12823 THU) — agent SFT dataset + multi-task training; **Agent-FLAN** (Chen et al. 2024 ACL Findings arXiv 2403.12881) — splits agent corpus into multi-turn / formatted / negative example three classes; **ReFT** (Luong et al. 2024 ACL arXiv 2401.08967) — SFT warm-start + online RL on math reasoning, PPO gains +9pp on GSM8K. These three are the "SFT first, then RL" standard three-stage of Agentic RL.

5. **Tool-augmented reasoning RL**: **ToolRL** (Qian et al. 2025 arXiv 2504.13958) — embeds tool calls into GRPO with reward = correctness + format + tool-use efficiency; **ReSearch** (Chen et al. 2025 arXiv 2503.19470) — treats search calls as first-class actions and learns multi-hop search with rule-based reward; **RAGEN / StarPO** (Wang et al. 2025 arXiv 2504.20073) — a multi-turn RL training framework with state-action token-level loss + critic-free GRPO variant. Common thread: **outcome-only reward + format shaping + token-mask loss + GRPO**.

6. **Web / GUI agent RL**: **WebRL** (Qi et al. 2024 ICLR-25 arXiv 2411.02337) — self-evolving curriculum + ORM + retrospective rollout, pushing 8B Llama to 43% on WebArena; **AgentQ** (Putta et al. 2024 arXiv 2408.07199) — MCTS search + AI critique + DPO offline training; **Computer-Use** (Anthropic Claude 3.5/3.7/4 Sonnet, since 2024-10-22) — RLHF + RL on screenshot + mouse/keyboard action space for GUI control (public knowledge: training details undisclosed, but system card mentions extensive human + AI feedback).

7. **Code agent RL**: **CodeRL** (Le et al. 2022 NeurIPS arXiv 2207.01780) was the first to use unit tests as reward signal + actor-critic; **PPOCoder** (Shojaee et al. 2023 arXiv 2301.13816) adds a composite reward of compilable + functional correctness; **SWE-RL** (Wei et al. 2025 Meta FAIR arXiv 2502.18449) does RL with rule-based reward (patch similarity + test-pass) on GitHub PR data, with Llama-3.3-70B pushing SWE-bench Verified to 41%.

8. **Self-rewarding & exploration**: **Self-Rewarding LM** (Yuan et al. 2024 Meta arXiv 2401.10020) has the policy serve as judge for iterative DPO with LLM-as-judge; but self-rewarding is more dangerous in agents than in single-turn alignment — the judge is also the agent itself, and **reward drift / model collapse easily occur**. In production the common trio is LLM-as-judge ensemble + rule-based grounding (test-pass, math checker) + human spot check.

9. **Three weapons for long-horizon credit assignment**: (a) **GAE + γ < 1** propagates credit along the trajectory but degenerates to MC return under sparse outcome reward; (b) **Hindsight relabeling** (the agent-side counterpart to HER) — failed trajectories are relabeled with "intermediate state as goal"; (c) **subgoal decomposition + process reward** — slice a 50-step trajectory into 5 subgoals × 10 steps and have a PRM score each subgoal. The L3 interview question "why is GRPO more sample efficient than PPO on long-horizon agents" — the answer is **trace-level reward directly matches trace-level credit**, bypassing the pain that a value model can hardly learn anything on long CoT.

## §1 Intuition: from RLHF to Agentic RL

### 1.1　Upgrading an LLM from "policy that writes" to "agent that acts"

RLHF trains an LLM into a policy that "writes per human preference"; but an RLHF policy is still fragile on tool calling / multi-turn interaction / long-horizon tasks:

- **single-turn preferences** do not directly transfer to multi-turn task success
- **The RM learns "which style humans prefer"**, not "which call order solves the problem"
- **Reward is over the entire response**, with no way to distinguish "the first 100 tokens reasoned correctly but token 101 chose the wrong tool"

The core difference between Agentic RL and RLHF is not a strict either/or over "reward must come from an RM vs. must come from the environment" — it's the **timescale of credit assignment** (RLHF's immediate reward on a single response vs. Agentic RL's credit assignment over a trajectory dozens of steps long) and the **mode of verification** (RLHF relies on an RM's subjective scoring of style/preference; Agentic RL leans more on programmatically verifiable objective outcomes such as test pass, math correct, or web task completed — though it can still use a learned reward model, e.g. §6.9's WebRL ORM, just with more emphasis on verifiable signals as the backbone than traditional RLHF-RM). This step upgrades alignment-style RL into **decision-making RL**.

### 1.2　Mental model: MDP / POMDP formulation

| Element | RLHF (single-turn) | Agentic RL |
|---|---|---|
| State $s_t$ | prompt | $(o_0, a_0, o_1, \dots, o_{t-1}, a_{t-1})$ (history) |
| Action $a_t$ | entire response | one `(thought, tool_call)` step or token-level sub-action |
| Reward $r_t$ | terminal RM score | terminal task success (most steps are 0) |
| Horizon $T$ | 1 (one response) | 10-200 steps (agent loop) |
| Trajectory length (tokens) | $10^2$-$10^3$ | $10^3$-$10^5$ |
| Environment | RM (neural net) | Real environment (shell / browser / search / Python) |

> 📝 The table above contrasts **typical** configurations, not a strict binary — Agentic RL can equally use a learned reward model for scoring (e.g. §6.9's WebRL ORM); it just leans more on programmatically verifiable signals (test-pass, task success) as the backbone than traditional RLHF-RM. The real dividing line is the timescale of credit assignment and the mode of verification, not "whether reward comes from an RM" per se.

```

      ┌──────────────────────────────┐
      │   Policy π_θ (LLM)           │  agent
      └──────────────┬───────────────┘
                     │ action a_t = (thought, tool_call)
                     ↓
      ┌──────────────────────────────┐
      │   Environment / Tool         │
      │   - search / shell / browser │
      │   - Python / unit test       │
      └──────────────┬───────────────┘
                     │ observation o_t
                     ↓
      ┌──────────────────────────────┐
      │   History buffer             │
      │   (concat back to prompt)    │
      └──────────────┬───────────────┘
                     │
                     └─→ back to π_θ
```

### 1.3　Agentic RL relative to three RL neighbors

| Neighbor | Common ground | Differences |
|---|---|---|
| **RLHF / DPO** | LLM + KL-anchored RL | Agentic must be multi-turn + tool I/O; reward comes from the environment, not an RM |
| **Reasoning RL (R1, R1-Zero)** | rule-based outcome reward + GRPO | R1 only on math/code answers, no tools; Agentic RL on tool calls + multi-step interaction |
| **Classical robotic RL (VPT, OpenVLA)** | Sparse terminal reward + long horizon | LLM agent action space = token sequence; robotic RL action = continuous control |

> 💡 **Interview framing** — when asked "what is Agentic RL", **disambiguate first**.

- (1) Strict definition: multi-turn + tool I/O + outcome reward RL
- (2) Boundary vs RLHF: RLHF is single-turn alignment, Agentic is multi-turn decision-making
- (3) Boundary vs reasoning RL: reasoning RL only checks answer correctness; Agentic RL evaluates task success on a real environment

These three sentences anchor the interviewer's expectations within 30 seconds.

## §2 Key modifications PPO / GRPO need for agents

### 2.1　Token mask: loss only on agent tokens

**This is the first iron law of Agentic RL**. Tokens in an agent trajectory split into two classes:

- **Agent token**: generated by the policy $\pi_\theta$ (thought, action JSON, final answer)
- **Environment token**: observation returned by the tool (search snippet, stdout, screenshot caption)

PPO/GRPO log-prob ratios and loss **must be computed only on agent tokens**. If loss is also applied on observation tokens:

- The policy will try to "teach the tool how to respond" (meaningless and reward-hackable)
- The gradient will be diluted by huge amounts of low-information observation tokens
- KL penalty will mistakenly estimate the policy's own distribution from environment text

The implementation is an **action_mask: [B, L]** tensor: 1 marks agent-generated tokens, 0 marks prompt / observation / padding. Loss computation: `(loss * action_mask).sum() / action_mask.sum()`.

> ⚠️ **Common bug** — early open-source implementations (including the early TRL agent example) missed the observation mask, causing training metrics to look like they were improving while task success dropped — a classic reward hack on tool output. OpenRLHF / verl / TRL 2024 versions have all fixed this; if you write your own trainer you must add it explicitly.

### 2.2　Trajectory-level GAE for agents

Agent trajectories are long (50-200 steps) and rewards are extremely sparse (only at the terminal). Let:

- $r_t \in \mathbb{R}$: reward at step $t$ (mostly $r_t = 0$, terminal $r_T = R \in \{0, 1\}$)
- $V_\phi(s_t)$: critic estimate

TD residual:

$$\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$$

GAE:

$$A_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}$$

**How to choose $\gamma$ for LLM agents?** Depends on the definition of "step":

- If step = **single token**: $\gamma$ close to 1 (token-level discount has no meaning)
- If step = **one thought-action-obs cycle**: $\gamma \in [0.95, 0.99]$ is reasonable, controlling long-horizon discounting

**How GAE degenerates in LLM agents**: under sparse terminal reward, $\lambda = 1$ + $\gamma = 1$ is equivalent to sequence-level MC return minus baseline. This is exactly the implicit explanation of GRPO directly using trace-level reward.

### 2.3　PPO loss adapted to agents

Masked PPO-Clip (outer expectation over the whole trajectory, inner sum over tokens):

$$\boxed{\;L^{\text{CLIP-agent}}(\theta) = \mathbb{E}_{\tau \sim \pi_\text{old}}\!\left[\frac{\sum_{t=1}^{T} m_t \cdot \min\!\big(\rho_t A_t,\, \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon) A_t\big)}{\sum_{t=1}^{T} m_t}\right]\;}$$

where $\tau$ is the trajectory (containing all $T$ tokens / steps), $m_t \in \{0, 1\}$ is the agent action_mask (1 for agent-generated tokens, 0 for observation/system tokens), and $\rho_t = \pi_\theta(a_t \mid s_t) / \pi_{\theta_\text{old}}(a_t \mid s_t)$ is the token-level importance ratio. Note that the outer expectation index is the trajectory $\tau$ while the inner sum index is the token $t$; do not confuse them.

Per-token KL penalty (written into the reward):

$$\tilde{r}_t = m_t \cdot \big(\text{rule}\_\text{reward}_t - \beta \log \tfrac{\pi_\theta(a_t \mid s_t)}{\pi_\text{ref}(a_t \mid s_t)}\big)$$

Note: KL is computed only on agent tokens; the $\pi$ on observation tokens is meaningless (those are emitted by the environment, not sampled by the model).

### 2.4　GRPO for agents: trace-level group-relative advantage

GRPO fits agents better because:

1. **Saves the critic** — agent value is hard to learn (long horizon + sparse reward)
2. **Trace-level reward directly corresponds to trace-level advantage** — no per-step value needed
3. **Multiple rollouts per prompt automatically reduce variance** — agent tasks are usually deterministic-env, multi-rollout gives the true reward variance

The formula (keep the PPO-Clip structure, advantage switched to intra-group normalization):

$$\hat{A}_i = \frac{r_i - \text{mean}(\{r_1, \dots, r_G\})}{\text{std}(\{r_1, \dots, r_G\}) + \epsilon}$$

All **agent tokens** in the whole trajectory share the same $\hat{A}_i$ (observation tokens are still masked out):

$$L^{\text{GRPO-agent}}(\theta) = \mathbb{E}\!\left[\frac{1}{G}\sum_{i=1}^G \frac{1}{\sum_t m_{i,t}}\sum_{t=1}^{T_i} m_{i,t} \cdot \Big(\min(\rho_{i,t} \hat{A}_i, \text{clip}(\rho_{i,t}, 1-\epsilon, 1+\epsilon) \hat{A}_i) - \beta\, \text{KL}_{i,t}\Big)\right]$$

KL typically uses the K3 estimator (Schulman 2020 blog): $\text{KL}_{i,t} = \exp(\log\pi_\text{ref} - \log\pi_\theta) - (\log\pi_\text{ref} - \log\pi_\theta) - 1$.

> ✅ **The "four savings" of GRPO-agent** — listed below.

- Saves the value model (one model's VRAM)
- Saves per-token credit assignment (trace-level advantage)
- Saves reward shaping (rule-based terminal is enough)
- Saves hyperparameters ($c_v, \lambda$ are not needed)

> ⚠️ **The "three pains" of GRPO-agent** — three trade-offs still need to be acknowledged.

- On long trajectories the trace-level advantage is too coarse (all agent tokens share the same $\hat{A}$) — credit dilution on long trajectories
- All-success / all-fail groups have $\text{std} = 0$ degeneracy (agent task reward is binary, this happens frequently)
- On-policy rollout is slow (agent rollout includes tool I/O latency, far slower than chat completion)

## §3 Reward design for agents (core)

Reward is the lifeline of Agentic RL. **If the reward is wrong, no model size or new algorithm will save you**; if the reward is right, plain GRPO can hit SOTA.

### 3.1　Outcome reward vs process reward

| Dimension | **Outcome reward** | **Process reward** |
|---|---|---|
| Supervision granularity | 1 reward at trajectory terminal | 1 reward per step (or per subgoal) |
| Source of labels | Task success (test pass / answer match) | PRM / step verifier / human |
| Sparsity | Very sparse (most steps 0) | Dense |
| Credit assignment | Hard (GAE is hard on long horizons) | Easy (each step is scored directly) |
| Reward hacking | Lower (with rule-based) | Higher (PRM can be hacked) |
| Implementation difficulty | Easy (test-pass / answer match) | Hard (PRM training costs) |

**Interview take**: reasoning RL (R1) chose outcome reward because math/code can be programmatically verified; mainstream agent RL also chose outcome reward because agent task ground truth is clearer (task completed yes/no). **Process reward is mainly used for reasoning-heavy tasks** (PRM on math steps); it is rare in tool-use agents.

### 3.2　Verifier-based reward (rule-based)

This is the cleanest reward form for Agentic RL: write the reward as an **executable verifier function**.

```python
def verifier_reward(trajectory) -> float:
    """
    trajectory: list of (thought, action, observation)
    Returns an outcome reward of 0 or 1
    """
    final_answer = trajectory[-1].final_answer

    # 1. Math: exact-match ground truth
    if task_type == "math":
        return 1.0 if normalize_math(final_answer) == ground_truth else 0.0

    # 2. Code: run unit tests
    if task_type == "code":
        code = extract_code(final_answer)
        pass_count = run_unit_tests(code, test_cases)
        return pass_count / len(test_cases)   # partial credit

    # 3. SWE-bench: apply patch + run test
    if task_type == "swe":
        try:
            apply_patch(repo, final_answer)
            return 1.0 if run_test(repo, expected_test) else 0.0
        except PatchError:
            return 0.0

    # 4. Web agent: task-specific verifier
    if task_type == "webshop":
        return webshop_grader(final_state)    # provided by the benchmark
```

**Key advantages of verifier-based reward**:

- Close to ground truth, **avoiding the main failure mode of reward hacking on learned RMs**
- Repeatable (the same trajectory gets the same reward), making advantage estimates well-defined
- Memory / compute overhead is minimal (executing a verifier is hundreds of times cheaper than one LLM forward pass)

**Core limitation**: usable only on "verifiable tasks" — math, code, formal verification, grader-able web/GUI tasks. Open-ended tasks (writing, dialogue) still need an RM.

### 3.3　Format reward / shaping reward

With outcome reward alone, agents often learn "format-broken but happens to be correct" trajectories — e.g. skipping the `<think>` block and going straight to `Action: answer(42)`. A **format reward** adds a lightweight format constraint signal:

```python
def format_reward(trajectory) -> float:
    """
    Check whether the trajectory matches the expected format schema
    Returns a continuous score in [0, 1]
    """
    score = 0.0

    # Must contain a <think>...</think> block
    if "<think>" in trajectory.text and "</think>" in trajectory.text:
        score += 0.3

    # Tool calls must be valid JSON
    for action in trajectory.actions:
        if is_valid_json(action.tool_call):
            score += 0.1
        else:
            score -= 0.2   # serious error

    # The final answer must be wrapped in \boxed{...} (math task)
    if has_boxed_answer(trajectory.final):
        score += 0.2

    return max(0.0, min(1.0, score))
```

A typical **composite reward** formulation:

$$r_\text{total} = \alpha \cdot r_\text{outcome} + \beta \cdot r_\text{format} + \gamma \cdot r_\text{shaping}$$

R1 / R1-Zero use a simple `accuracy_reward + format_reward` sum; ToolRL / RAGEN etc. add tool-call efficiency shaping.

### 3.4　Length penalty (prevent the agent from dragging too long)

An emergent failure mode of agent RL: **the model learns that "dragging out the trajectory raises the probability of a correct answer"** — for a 5-step task the agent goes 50 steps. This is a form of reward hacking.

Mitigation:

$$r_\text{adjusted} = r_\text{outcome} - \lambda \cdot \max(0, T - T_\text{target})$$

or a softer sigmoid form:

$$r_\text{adjusted} = r_\text{outcome} \cdot \sigma\!\big(-(T - T_\text{target}) / \tau\big)$$

DAPO reports "overlong reward shaping" — once generation length enters the buffer zone $(L_\text{max}-L_\text{cache}, L_\text{max}]$, reward ramps down piecewise-linearly from 0 to -1 (and is fixed at -1 beyond $L_\text{max}$), rather than decaying exponentially, replacing a hard cutoff with a smooth penalty to avoid training instability.

### 3.5　Tool-call shaping reward

Reward "calling the right tool", penalize "calling the wrong tool / repeated calls":

```python
def tool_shaping(trajectory) -> float:
    score = 0.0

    # Calling the right tool (task-relevance heuristic)
    if task_needs_search and any(a.tool == "search" for a in trajectory.actions):
        score += 0.1

    # Penalize consecutive duplicate calls
    consecutive_dup = count_consecutive_duplicate_calls(trajectory.actions)
    score -= 0.05 * consecutive_dup

    # Penalize calls to non-existent tools
    invalid_calls = sum(1 for a in trajectory.actions if a.tool not in TOOL_REGISTRY)
    score -= 0.3 * invalid_calls

    return score
```

> ⚠️ **Risks of shaping reward** — if shaping is given improperly, the agent will over-fit the shaping signal and ignore the outcome. **Mainstream practice**: shaping reward weight ≪ outcome reward weight (typically 0.1 : 1), and shaping must be capped (cannot accumulate without bound).

### 3.6　RLAIF for agents: LLM as reward

When an external verifier is hard to write (open-ended tasks), use a strong LLM as judge:

```python
def llm_judge_reward(trajectory, judge_model) -> float:
    """
    Use a judge LLM to score the trajectory
    """
    prompt = f"""
    Judge whether the agent completed this task successfully.
    Task: {trajectory.task}
    Final state: {trajectory.final_state}
    Return JSON: {{"success": bool, "reasoning": str}}
    """
    judgment = judge_model(prompt)
    return 1.0 if judgment["success"] else 0.0
```

**Risks**:

- The judge LLM has its own biases (length preference, sycophancy) → amplified into the student
- The judge itself may be prompt-injected (the agent trajectory contains malicious instructions)
- High compute cost (one judge call per trajectory)

Mainstream mitigations: (a) **judge ensemble** (3-5 different models judge, majority vote); (b) **judge with rubric** (strictly constrain output structure); (c) **rule + LLM hybrid** (verifiable parts use rule, open-ended parts use LLM).

## §4 Long-horizon credit assignment

### 4.1　Sparse reward is the core pain of agent RL

Classical game RL (Atari, Mujoco) has dense reward; an LLM agent on a long trajectory **gets reward 0 almost everywhere and a signal only at the terminal**. This causes:

- The value model can barely learn anything (what should middle V be?)
- Per-token policy gradient has huge variance
- The causal chain from early steps to reward is diluted — did "step 3 picking search tool" cause "step 50 being correct"?

### 4.2　Degeneration of discount + GAE on agents

Recall GAE:

$$A_t = \sum_l (\gamma\lambda)^l \delta_{t+l}, \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

Under sparse terminal reward ($r_t = 0$ for $t \lt T$, $r_T = R$), expanding $\delta_{t+l}$ and summing over $l$ gives:

$$A_t = (\gamma\lambda)^{T-t} R - V(s_t) + \gamma(1-\lambda)\sum_{l\ge0}(\gamma\lambda)^l V(s_{t+l+1})$$

Only when $\lambda=1$ does the trailing term telescope away exactly, degenerating to $A_t = \gamma^{T-t} R - V(s_t)$ (pure discounted MC return minus baseline, i.e. **advantage ≈ discounted return - baseline**); for general $\lambda<1$ a string of $\lambda$-dependent value-correction terms remains. If $V_\phi$ is inaccurate (often the case in agents), this $\lambda=1$ limiting case degenerates into raw MC return; GAE's bias-variance trade-off fails.

**Practical take**: on agent RL, GAE is less stable than group-relative advantage (GRPO); this is one of the fundamental reasons why GRPO is more sample efficient than PPO on agents.

### 4.3　Hindsight relabeling for agents

Hindsight Experience Replay (Andrychowicz et al. 2017 NeurIPS) originated in robot manipulation: failed trajectories aren't thrown away; instead the **state actually reached is treated as the goal** and reward is relabeled.

LLM agent version:

```python
def hindsight_relabel(trajectory):
    """
    Turn a failed trajectory into a successful trajectory for an "alternative task"
    """
    if trajectory.outcome == 1:
        return [trajectory]   # successful trajectories untouched

    # Suppose the agent meant to buy item A on the web but ended on item B's page
    # → relabel as "find item B"; reward = 1
    alt_task = describe_terminal_state(trajectory.final_state)
    relabeled = trajectory.with_task(alt_task)
    relabeled.outcome = 1
    return [trajectory, relabeled]
```

> 💡 **Difficulty of agentic hindsight** — requires that "any terminal state can be described as a reasonable task". Easy for open web environments (shopping, navigation); hard for math/code tasks (a wrong answer can't be rewritten as "the correct answer to another problem").

### 4.4　Subgoal decomposition + process reward

Slicing a long trajectory into subgoals is another credit-assignment route:

- Slice a 100-step trajectory into 5 subgoals × 20 steps
- Each subgoal terminal gets a process reward (whether the subgoal is completed)
- Subgoal rewards sum to the trajectory reward

Implementations:

- **Hand-crafted subgoals**: humans write the criterion for each subgoal (e.g. "reaching the cart page" triggers subgoal-1 reward)
- **LLM-decomposed subgoals**: a planner LLM splits the task into subgoals; a verifier judges each
- **PRM-style step reward**: train a PRM to score each step (Math-Shepherd line)

> ⚠️ **Cost of subgoal RL** — wrong subgoal boundaries make the agent learn "deliberately trigger subgoal reward without actually completing the task". This is the general problem of process reward.

### 4.5　Per-step KL penalty to prevent policy collapse

Over long agent trajectories the policy easily "all-ins" (generates high-confidence tokens at every step), causing entropy collapse:

$$\tilde{r}_t = m_t \cdot \big(r_t - \beta \cdot \text{KL}(\pi_\theta(\cdot \mid s_t) \| \pi_\text{ref}(\cdot \mid s_t))\big)$$

**Key**: KL must be per-step + only on agent tokens. Once KL is computed on observation tokens, the policy gets penalized for "mimicking environment text" — but this penalty is meaningless (the policy shouldn't mimic the environment, only use observations as conditioning).

R1-Zero uses $\beta = 0.001$; agent tasks commonly use $\beta \in [0.001, 0.05]$.

## §5 Self-rewarding & exploration in agent RL

### 5.1　Self-Rewarding LM (Yuan et al. 2024 Meta arXiv 2401.10020)

Core idea: have the policy serve as judge, iterative DPO:

```

  Iteration k:
    1. policy_k generates multiple responses per prompt
    2. policy_k self-scores via LLM-as-judge (also a policy_k prompt)
    3. High vs low scores form a preference pair
    4. policy_{k+1} = DPO(policy_k, preference_pair)
```

Effect: on AlpacaEval, iterative self-scoring continuously improves.

### 5.2　Risks of self-rewarding on agents

Biggest difference between agent task and alignment task: alignment has an "objective preference distribution" (humans find it helpful, polite) suitable for LLM judging; **agent tasks have objective ground truth (test pass / task success)** — self-rewarding then:

- The judge itself may be wrong (the agent got it wrong but self-evaluates correct) → training collapse
- Iterative drift: each round reinforces trajectories "I thought were correct", drifting away from ground truth
- Exploration degeneration: high self-scores prefer known patterns, suppressing exploration of new tools

**Mainstream practice**: agent RL **prefers rule-based ground truth**, with self-rewarding only as an auxiliary signal (e.g. open-ended task fallback).

### 5.3　Exploration in agent RL

Agent action space includes:

- **Token-level exploration**: sampling temperature, controlling token-choice diversity
- **Tool-call-level exploration**: each thought-action cycle picks different tools / different queries
- **Trajectory-level exploration**: entirely different trajectory plans

Classical practices:

| Method | Implementation |
|---|---|
| Temperature schedule | At rollout $T \in [0.7, 1.2]$, anneal down with training |
| Top-p / Top-k | Restrict sampling range to avoid outlier tokens |
| ε-tool-choice | With probability $\epsilon$, pick a random tool (replacing the LLM-policy choice) |
| Diverse beam | Multi-trajectory with diverse beam search ensures diversity |
| GRPO group sampling | $G = 16$ rollouts per prompt, natural exploration |

The key insight of **RAGEN / StarPO** (Wang et al. 2025 arXiv 2504.20073): in multi-turn agent RL **rollout diversity is the "firebreak" for collapse** — single-trajectory training causes the policy to degenerate into deterministic mode.

### 5.4　Curriculum & difficulty scheduling

Sort agent tasks from easy to hard, schedule based on the model's current ability:

- **WebRL** (Qi et al. 2024 ICLR-25 arXiv 2411.02337) uses a self-evolving curriculum — failed tasks are recorded and added to the buffer in the next round
- **Absolute Zero** / R-Zero use a learnability reward: pick tasks where the model's success rate is ≈ 50% (maximum learning signal)

> 💡 **Curriculum is the "hidden component" of long-horizon agent RL** — not an algorithmic contribution, but empirically it **affects sample efficiency more than swapping the algorithm**. R1 / R1-Zero's reasoning RL also uses an implicit curriculum (data difficulty rises gradually).

## §6 Specific algorithms (representative Agentic RL papers)

Sorted by "time + data/task type", each with one sentence + key formula.

### 6.1　VPT (Baker et al. 2022 NeurIPS OpenAI, arXiv 2206.11795)

**Setting**: Minecraft; first pretrain an inverse dynamics model (IDM) on 70k hours of YouTube videos, then have the IDM auto-label action → behavior clone → RL fine-tune.

**Key**: the first large-scale "video → action label → policy → RL" pipeline. The early blueprint for agent RL, proving **scaling RL with imitation pretrain** is feasible.

### 6.2　AgentTuning (Zeng et al. 2023 THU arXiv 2310.12823)

**Setting**: construct the AgentInstruct dataset (demonstrations from 6 agent tasks), multi-task SFT.

**Key**: not RL, but **agent SFT** — yet this is the standard warm-start step for Agentic RL. After AgentTuning, Llama-2 averages +50% on agent tasks.

> 💡 **Position of AgentTuning** — in the Agentic RL pipeline, AgentTuning-style SFT is a necessary warm-start before RL. Doing RL on an agent from scratch is extremely hard because the base model does not know how to emit legal tool-call schemas.

### 6.3　Agent-FLAN (Chen et al. 2024 ACL Findings, arXiv 2403.12881)

**Setting**: Agent SFT data split into three classes — multi-turn dialogue / formatted tool calls / negative examples (refusal / failure cases).

**Key**: **negative examples significantly reduce hallucinated tool calls** — SFT doesn't just see "how to do it right", it also sees "why this is wrong". An important milestone for agent SFT engineering.

### 6.4　ReFT (Luong et al. 2024 ACL arXiv 2401.08967)

**Setting**: math reasoning agent; SFT warm-start, then PPO with rule-based outcome reward (answer correctness).

**Key formula** (standard PPO + verifier):

$$r(\tau) = \mathbb{1}[\text{answer}(\tau) = y^*] - \beta \cdot \text{KL}(\pi_\theta \| \pi_\text{SFT})$$

**Result**: GSM8K +9pp over SFT, MathQA +7pp. Proves PPO + outcome reward is stable and feasible on reasoning agents — the predecessor experiment for R1.

### 6.5　DeepSeek-R1 (DeepSeek-AI 2025 arXiv 2501.12948) extension on agents

R1 itself isn't an agent paper, but R1's GRPO + rule-based reward + format reward methodology was directly inherited by ToolRL / ReSearch / RAGEN. **R1 = the algorithmic baseline template of Agentic RL**.

Review of GRPO on agents:

- $G$ rollouts per prompt
- Trace-level reward (rule-based)
- Group-relative advantage
- Per-step KL with K3 estimator
- Agent token mask

### 6.6　ToolRL (Qian et al. 2025 arXiv 2504.13958)

**Setting**: GRPO on a tool-augmented LLM with reward = correctness + format + tool-use efficiency.

**Key formula**:

$$r(\tau) = r_\text{correct} + \alpha \cdot r_\text{format} + \gamma \cdot r_\text{tool-eff}$$

where $r_\text{tool-eff}$ penalizes redundant/invalid tool calls.

**Result**: 7B model approaches GPT-4 on BFCL (Berkeley Function Calling Leaderboard). Open-source validation of GRPO + tool shaping stability.

### 6.7　ReSearch (Chen et al. 2025 arXiv 2503.19470)

**Setting**: search-augmented agent treating search as a first-class action; reward = answer correctness only (rule-based).

**Key idea**: multi-hop search can be learned without process reward; **outcome-only + GRPO is enough** — provided the base model, after SFT warm-start, can already emit legal search queries.

### 6.8　RAGEN / StarPO (Wang et al. 2025 arXiv 2504.20073)

**Paper**: "Understanding Self-Evolution in LLM Agents via Multi-Turn Reinforcement Learning".

**Setting**: multi-turn RL agent training framework with state-action token-level loss + critic-free.

**Key contributions**:

1. **StarPO** (**S**tate-**T**hinking-**A**ctions-**R**eward Policy Optimization): critic-free, the whole trajectory shares the advantage, with a strict token mask that limits loss to agent tokens. The **StarPO-S** variant introduces fine-grained reasoning-aware reward + optional critic incorporation, further mitigating multi-turn reward sparsity (the paper abstract's exact wording).
2. **Rollout diversity = collapse firebreak**: empirically, group size $G = 16$ is significantly more stable than $G = 4$
3. **Trajectory length signal**: when failed-trajectory length is large, reward shaping should add a length penalty

### 6.9　WebRL (Qi et al. 2024 ICLR-25 arXiv 2411.02337)

**Setting**: web agent (WebArena), self-evolving curriculum + ORM + retrospective rollout.

**Key components**:

- ORM (Outcome Reward Model) trained from task success → gives reward online
- Failed tasks enter the curriculum buffer; the next round increases their weight
- Retrospective rollout: failed trajectories are rewritten into "correct trajectories" by an LLM and re-SFT'd

**Result**: Llama-3.1-8B reaches 43% on WebArena (vs GPT-4 14.4%), proving that small model + good RL pipeline > large model + zero-shot.

### 6.10　AgentQ (Putta et al. 2024 arXiv 2408.07199)

**Setting**: web agent, MCTS + AI critique + offline DPO.

**Key idea**: MCTS searches for "reward-balanced" preference pairs (high-score trajectory vs low-score trajectory) and DPO offline-trains them. No online RL infra needed; practical for compute-limited scenarios.

### 6.11　WebGUM (Furuta et al. 2024 ICLR arXiv 2305.11854)

**Setting**: HTML + screenshot multimodal web agent, offline SFT from demonstrations (**not RL fine-tune** — the paper is the imitation/supervised paradigm).

**Key**: feeds DOM + screenshot together into the model, improving grounding accuracy over pure text. It's the dataset + base stage for Computer-Use / web agent RL (e.g. WebRL, AgentQ). Subsequent web agent RL often does RL fine-tune on top of WebGUM-class bases.

### 6.12　CodeRL (Le et al. 2022 NeurIPS arXiv 2207.01780)

**Setting**: code generation + actor-critic, reward = unit-test pass.

**Key formula**: classical actor-critic with the critic as a token-level baseline; the PG signal comes from final test-pass.

### 6.13　PPOCoder (Shojaee et al. 2023 arXiv 2301.13816)

**Setting**: code generation + PPO with composite reward = compilable + functional correctness.

**Key**: early attempt at PPO for code-gen. Proves multi-component reward is more stable than single test-pass training.

### 6.14　SWE-RL (Wei et al. 2025 Meta FAIR arXiv 2502.18449)

**Setting**: SWE-bench / GitHub PR data, rule-based reward = patch similarity + test-pass, GRPO.

**Key features**:

- Data scale: Meta constructs 76M+ context-issue-patch triples from GitHub PR commit history (check the original paper's Table/Section for the exact scale — this tutorial could not verify offline whether 76M+ is the final triple count or the raw PR/commit event count)
- Reward: edit similarity (oracle patch ↔ predicted patch) + binary test pass
- Algorithm: pure GRPO + format reward

**Result**: Llama-3.3-70B + SWE-RL reaches 41% on SWE-bench Verified (no scaffold), proving **rule-based RL on real PR data can let the model learn emergent reasoning behavior** — e.g. file-level retrieval planning, root-cause analysis, test self-validation.

### 6.15　OpenVLA (Kim et al. 2024 arXiv 2406.09246)

**Setting**: a 7B vision-language-action model for robot manipulation, fine-tuned on 970K real robot demonstrations (open-source base + LoRA fine-tuning recipe); the paper focuses on imitation learning + parameter-efficient adaptation, **not task-specific RL fine-tune**.

**Key**: a robot agent rather than an LLM agent; often used to compare "LLM agent vs robot agent". Subsequent work (OpenVLA-OFT, π-RL family, etc.) does RL on top of the OpenVLA base, but the OpenVLA paper itself does not.

### 6.16　Anthropic Computer-Use (public knowledge — training details undisclosed)

Anthropic Claude 3.5 Sonnet (new) supports Computer-Use since 2024-10-22; 3.7 / 4.0 / 4.5 / Opus 4.x continue to iterate.

**Public materials describe only capability + safety guardrails; the training algorithm / reward form is undisclosed**:

- Action space = screen coordinates + keyboard events + mouse events (screenshot as observation)
- Safety: constitutional-AI-style guardrails + red-teaming + prompt-injection defense
- The system card mentions training involves human demonstrations + synthetic data, but **does not disclose** specifics like RLHF / GRPO / verifier reward

Community **speculation** (**speculative only, not official**): possibly GRPO/PPO + verifier-based reward + Constitutional-AI-style AI feedback, but with no official confirmation. When discussing Computer-Use training in interviews, clearly distinguish "public capability" from "speculated internals".

## §7 Code patterns (PyTorch / pseudocode)

The most easily mis-implemented pieces when writing Agentic RL. Each block is independently readable.

### 7.1　Agent rollout (state, action, reward collection)

```python
import torch
from dataclasses import dataclass

@dataclass
class Step:
    obs: str                   # previous-step observation or prompt
    thought_tokens: list[int]  # thought generated by the agent
    action_tokens: list[int]   # tool_call JSON generated by the agent
    tool_name: str
    tool_args: dict
    observation: str           # tool's returned result
    done: bool

def rollout(policy, env, prompt, max_steps=20, max_tokens_per_step=512):
    """
    One agent trajectory rollout
    Returns: trajectory (list of Step), final reward
    """
    trajectory = []
    history = prompt
    for step_idx in range(max_steps):
        # ── Agent generates (thought, action) ──
        agent_output = policy.generate(
            prompt=history,
            stop_tokens=["</action>"],
            max_new_tokens=max_tokens_per_step,
            temperature=0.7,
        )
        thought, action_json = parse_thought_action(agent_output)

        # ── Call the tool ──
        tool_name = action_json["tool"]
        tool_args = action_json["args"]
        if tool_name == "final_answer":
            obs = action_json["answer"]
            done = True
        else:
            obs = env.call(tool_name, tool_args)
            done = False

        # Update history (append back to the prompt)
        history = history + agent_output + f"\n<obs>{obs}</obs>\n"

        trajectory.append(Step(
            obs=history,                    # history up to this point
            thought_tokens=tokenize(thought),
            action_tokens=tokenize(action_json),
            tool_name=tool_name,
            tool_args=tool_args,
            observation=obs,
            done=done,
        ))

        if done or step_idx == max_steps - 1:
            break

    # ── Terminal reward ──
    final_reward = env.compute_reward(trajectory)
    return trajectory, final_reward
```

### 7.2　Trajectory-level GAE advantage

```python
import torch

def trajectory_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """
    Compute step-level GAE advantages
    rewards: [T]    per-step reward (mostly = 0, terminal = R)
    values:  [T+1]  V(s_0)...V(s_T); V(s_T) should be 0 (terminal)
    dones:   [T]    1 if terminal else 0
    Returns: advantages [T], returns [T]
    """
    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    gae = 0.0
    for t in reversed(range(T)):
        non_term = 1.0 - dones[t]
        delta = rewards[t] + gamma * values[t + 1] * non_term - values[t]
        gae = delta + gamma * lam * non_term * gae
        advantages[t] = gae
    returns = advantages + values[:T]
    return advantages, returns
```

### 7.3　PPO loss adapted to agents (with action_mask)

**This is the most important piece of Agentic RL code**. Difference from RLHF PPO: observation tokens must be explicitly masked out.

```python
import torch
import torch.nn.functional as F

def ppo_agent_step(policy, value, batch, eps_clip=0.2, c_v=0.5, c_e=0.01):
    """
    batch:
      input_ids:       [B, L]    full trajectory tokens (prompt + thought + action + obs ...)
      action_mask:     [B, L]    1 = agent-generated token, 0 = prompt/observation/pad
      old_log_probs:   [B, L]    log π_θ_old at sample time, 0 at masked positions
      advantages:      [B, L]    step-level GAE advantages (broadcast to all agent tokens of that step)
      returns:         [B, L]    GAE returns for value loss
    """
    logits = policy(batch["input_ids"]).logits           # [B, L, V]
    log_probs = F.log_softmax(logits[:, :-1], dim=-1)    # [B, L-1, V]
    targets = batch["input_ids"][:, 1:].unsqueeze(-1)
    new_log_probs = log_probs.gather(-1, targets).squeeze(-1)  # [B, L-1]
    new_log_probs = F.pad(new_log_probs, (1, 0))               # align to [B, L]

    # ── Key: action_mask ──
    mask = batch["action_mask"].float()
    # Compute ratio / loss only on agent-generated tokens
    ratio = torch.exp((new_log_probs - batch["old_log_probs"]) * mask)
    # At observation positions: ratio = exp(0) = 1, does not affect surr1/surr2

    A = batch["advantages"]
    surr1 = ratio * A
    surr2 = torch.clamp(ratio, 1.0 - eps_clip, 1.0 + eps_clip) * A
    # Mean over masked positions (avoid observation tokens lowering loss scale)
    policy_loss = -((torch.min(surr1, surr2) * mask).sum() / mask.sum().clamp_min(1.0))

    # Value loss: only on agent tokens — this is an engineering simplification, not a claim that V at observation positions is itself meaningless (see note below)
    V = value(batch["input_ids"]).squeeze(-1)            # [B, L]
    value_loss = (((V - batch["returns"]) ** 2) * mask).sum() / mask.sum().clamp_min(1.0)

    # Entropy bonus: only on agent tokens
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(-1)               # [B, L-1]
    entropy = F.pad(entropy, (1, 0))
    entropy_bonus = (entropy * mask).sum() / mask.sum().clamp_min(1.0)

    loss = policy_loss + c_v * value_loss - c_e * entropy_bonus

    # Monitoring
    with torch.no_grad():
        approx_kl = ((ratio - 1) - torch.log(ratio.clamp_min(1e-8))) * mask
        approx_kl = approx_kl.sum() / mask.sum().clamp_min(1.0)
    return loss, {
        "policy": policy_loss.item(),
        "value": value_loss.item(),
        "entropy": entropy_bonus.item(),
        "approx_kl": approx_kl.item(),
    }
```

> 📝 **The critic mask can be defined independently of the policy mask** — the policy loss must exclude observation tokens (to keep the policy gradient from flowing into environment-generated text); but the value function is conceptually $V(s_t)$, whose natural anchor sits at the state boundary (i.e. the last observation token / right before the next action begins), so $V$ at that boundary position is meaningful — it is exactly the source of the $V(s_0)\dots V(s_T)$ bootstrap values used by GAE in §7.2. Restricting value loss to agent tokens above is an engineering simplification; readers should understand this simplification in relation to the boundary-based definition of $V$, rather than reading it as "V at observation positions has no meaning."

> ⚠️ **5 common mistakes with agent_mask** —

- Must cover **prompt tokens**: prompts aren't agent-generated; mask = 0
- Must cover **all observation tokens**: every token returned by the tool (including the `<obs>` tag itself) mask = 0
- Must cover **all padding**: right-pad positions mask = 0
- **Separator tokens** (e.g. `<action>`, `</thought>`) count as agent tokens; mask = 1
- Multi-turn batches have **different mask patterns for different trajectories**; must be computed per-sample

### 7.4　GRPO group-relative reward on agent trajectories

```python
import torch
import torch.nn.functional as F

def grpo_agent_loss(policy, ref_policy, batch, eps_clip=0.2, beta=0.04):
    """
    batch:
      input_ids:     [N, L]   N = sum_b G samples in batch
      action_mask:   [N, L]   agent-generated token mask
      old_log_probs: [N, L]   detached log probs at rollout time
      rewards:       [N]      trajectory-level outcome reward
      group_id:      [N]      same prompt → same group_id
    """
    rewards = batch["rewards"]
    gid = batch["group_id"].long()

    # ── Intra-group normalization ──
    num_groups = int(gid.max().item()) + 1
    counts = torch.zeros(num_groups, device=rewards.device).scatter_add_(
        0, gid, torch.ones_like(rewards))
    sums = torch.zeros(num_groups, device=rewards.device).scatter_add_(
        0, gid, rewards)
    group_mean = sums / counts.clamp_min(1.0)
    diff_sq = (rewards - group_mean[gid]) ** 2
    sq_sums = torch.zeros(num_groups, device=rewards.device).scatter_add_(
        0, gid, diff_sq)
    group_std = (sq_sums / counts.clamp_min(1.0)).sqrt()

    A = (rewards - group_mean[gid]) / (group_std[gid] + 1e-8)   # [N]
    A = A.unsqueeze(-1)                                          # [N, 1] shared across the trajectory

    # ── log-prob ratio ──
    logits = policy(batch["input_ids"]).logits[:, :-1]
    log_probs = F.log_softmax(logits, dim=-1)
    tgt = batch["input_ids"][:, 1:].unsqueeze(-1)
    new_log_probs = log_probs.gather(-1, tgt).squeeze(-1)
    new_log_probs = F.pad(new_log_probs, (1, 0))                 # [N, L]
    mask = batch["action_mask"].float()

    ratio = torch.exp((new_log_probs - batch["old_log_probs"]) * mask)

    # ── PPO-Clip surrogate (advantage broadcast to the whole trajectory) ──
    surr1 = ratio * A
    surr2 = torch.clamp(ratio, 1.0 - eps_clip, 1.0 + eps_clip) * A

    # ── KL with K3 estimator (Schulman 2020 blog) ──
    with torch.no_grad():
        ref_logits = ref_policy(batch["input_ids"]).logits[:, :-1]
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)
        ref_token_lp = ref_log_probs.gather(-1, tgt).squeeze(-1)
        ref_token_lp = F.pad(ref_token_lp, (1, 0))
    delta = ref_token_lp - new_log_probs                          # log(π_ref / π_θ)
    kl_per_token = torch.exp(delta) - delta - 1.0                 # K3, non-negative

    token_obj = torch.min(surr1, surr2) - beta * kl_per_token     # [N, L]
    seq_len = mask.sum(dim=-1).clamp_min(1.0)                     # [N]
    per_seq = (token_obj * mask).sum(dim=-1) / seq_len            # [N]
    loss = -per_seq.mean()

    return loss, {
        "reward_mean": rewards.mean().item(),
        "advantage_std": A.squeeze(-1).std().item(),
        "kl": (kl_per_token * mask).sum().item() / mask.sum().clamp_min(1.0).item(),
    }
```

### 7.5　Outcome + step reward combination

```python
def composite_reward(trajectory, weights=None):
    """
    Combine outcome / format / shaping rewards into the final reward
    weights: dict[str, float]
    """
    if weights is None:
        weights = {"outcome": 1.0, "format": 0.2, "tool_eff": 0.1, "length": -0.05}

    r = {}
    r["outcome"] = outcome_verifier(trajectory)                  # in {0, 1}
    r["format"]  = format_score(trajectory)                      # in [0, 1]
    r["tool_eff"] = tool_efficiency_score(trajectory)            # in [-1, 1]
    r["length"]  = max(0, len(trajectory.steps) - target_len)    # excess steps

    total = sum(weights[k] * r[k] for k in r)
    return total, r
```

> 💡 **Process-then-outcome composite form** — if you have a PRM, you can let the PRM provide step-level shaping first, but **at least 50% of the final trajectory reward should come from outcome**, otherwise the agent will exploit a local optimum that only cares about PRM scores.

### 7.6　Verifier-based reward (code test / math match)

```python
import re
import subprocess

def verifier_reward(trajectory, task_type, ground_truth):
    """
    Rule-based outcome reward
    """
    final = trajectory.final_answer

    if task_type == "math":
        return float(extract_boxed_answer(final) == normalize(ground_truth))

    if task_type == "code":
        code = extract_python_code(final)
        if code is None:
            return 0.0
        passed = 0
        for test in ground_truth["tests"]:
            try:
                result = subprocess.run(
                    ["python", "-c", code + "\n" + test],
                    capture_output=True, timeout=5,
                )
                if result.returncode == 0:
                    passed += 1
            except subprocess.TimeoutExpired:
                continue
        return passed / len(ground_truth["tests"])

    if task_type == "swe":
        patch = extract_unified_diff(final)
        if patch is None:
            return 0.0
        success = apply_patch_and_run_test(
            repo=ground_truth["repo"],
            patch=patch,
            test=ground_truth["test"],
        )
        return float(success)

    if task_type == "webshop":
        return webshop_grader(trajectory.final_state, ground_truth)

    raise ValueError(f"Unknown task type: {task_type}")
```

## §8 Frontier (key 2024-2026 trends)

### 8.1　Critic-free RL is the new default

After DeepSeek-R1 (Jan 2025), **critic-free RL (the GRPO family) became the open-source agent RL mainstream**. Reasons:

- The value model can hardly learn anything on long-horizon agents
- One model's worth of VRAM saved → bigger batches / group sizes
- Easier tuning (no $c_v$, value lr to tune)

verl (ByteDance 2024+), OpenRLHF, TRL have all made GRPO / RLOO / ReMax first-class trainers.

### 8.2　The victory of rule-based reward on agents

ReSearch / SWE-RL / RAGEN are all **outcome reward + format reward, rule-based**; WebRL, however, trains an ORM (learned outcome reward model) to score (see §6.9) — a learned-RM route, unlike the other three's pure rule-based verification. What all four share is that they use **outcome-level (rather than token-level subjective preference) signal**, not that they're all rule-based; it's the rule-based branch specifically that avoids the main failure mode of reward hacking on learned RMs.

**Why has rule-based suddenly become viable**:

- Agent tasks are more "programmatically verifiable" than alignment tasks — test pass / answer match / task complete
- DeepSeek-R1 proved rule-based reward is stable and scalable for LLM RL
- Learned RMs are more easily hacked in multi-turn settings (the trajectory space is large)

### 8.3　The infra challenges of long-horizon training

Agent RL rollout is slow because every trajectory includes tool I/O latency. Infra trends:

- **vLLM / SGL async rollout**: decouple generation from training; pool rollouts
- **Sandboxed execution**: tool execution inside isolated containers, parallelizable
- **Trajectory queue**: rollout workers / training workers are async; trajectories flow through a message queue
- **Off-policy correction**: there's lag between rollout and update; use IS clip or V-trace to correct

Representative implementations: **verl** (ByteDance Seed open source), **AReaL** (Ant Group + Tsinghua, async RL system arXiv 2505.24298), **OpenRLHF v0.5+**.

### 8.4　Tool-RL on TAU-bench / SWE-bench / OSWorld

Industry-benchmark SOTA trends (public numbers in 2025-2026):

| Benchmark | Task | 2024 SOTA | 2025-2026 SOTA | Key approach |
|---|---|---|---|---|
| **TAU-bench (retail)** | customer service multi-turn | ~50% (GPT-4) | 70-80% (Claude 4.x, GPT-5) | RLHF + agent SFT |
| **SWE-bench Verified** | GitHub PR fix | ~25% (Claude 3.5) | **70-80%+** (Claude 4.x, o3) | Agent scaffold + RL |
| **OSWorld** | OS GUI task | ~12% (GPT-4V) | ~50-60% (Claude 4.x, Operator) | Computer-Use RL |
| **WebArena** | web nav | 14.4% (GPT-4) | 43% (WebRL Llama-8B) | curriculum + RL |
| **GAIA** | general assistant | 15% (GPT-4) | 60-70% (Claude 4.x, o3) | Agent + tool RL |

Note: benchmark contamination risk is high; **as of 2026Q1, OpenAI has retired SWE-bench Verified** (reasons: contamination + test flaws); more credible benchmarks now are SWE-Lancer, SWE-bench Multilingual, private holdouts.

### 8.5　Anthropic Computer-Use (Claude 3.5 → 4.5 → Opus 4.x)

**Public knowledge**:

- 2024-10-22 Claude 3.5 Sonnet (new) first shipped Computer-Use beta
- 2025: Claude 3.7 / 4.0 / 4.5 continued iterating with speed + accuracy improvements
- Training involves many human demonstrations + AI-generated rollouts + RLHF + safety red-teaming
- Action space = screenshots + mouse + keyboard + filesystem access
- Public system cards mention Constitutional AI + Computer-Use specific safety filter

**Algorithm-level inferred information** (academic speculation, not officially confirmed):

- Training likely involves RLHF on screenshot trajectories
- Reward likely includes a task-completion grader (LLM-as-judge) + safety classifier
- Possibly GRPO / RLOO style critic-free RL (multiple public papers mention critic-free)

### 8.6　2025-2026 papers panorama

| Paper | Direction | Core contribution |
|---|---|---|
| **KodCode** (Xu et al. 2025) | code agent RL | High-quality code RL dataset + GRPO baseline |
| **DAPO** (Yu et al. 2025 ByteDance) | GRPO improvement | clip higher / dynamic sampling / token loss / overlong shaping |
| **VAPO** (ByteDance 2025) | GRPO + lightweight critic | Alleviates trace-level credit dilution |
| **CISPO** (MiniMax 2025) | Importance sampling improvement | Solves the instability of negative-advantage large ratios |
| **R-Zero** | Self-Play RL | Challenger-Solver self-play, learnability reward |
| **Absolute Zero** | Self-Play RL | Completely without external tasks; code executor as verifier |
| **Search-R1** | Search agent RL | Search as first-class action + rule reward |
| **Light-R1** / **Sky-T1** | Reasoning + tool RL | Open-source reproduction of R1 + agent extension |
| **OpenAgent** / **Llama-Agent** | Data + framework | Large-scale agent SFT + RL pipeline |

> 💡 **2026Q1-Q2 trend** — agent RL on real environments (OS, browser, IDE) is becoming the open-source mainline. **simulator → real environment → deployment RL** sim-to-real pipelines are the next frontier. Anthropic / OpenAI / DeepSeek / Meta are all working on this but the details are not public.

## §9 Failure modes & engineering experience

### 9.1　The "seven deadly sins" of Agentic RL

| Failure mode | Symptom | Root cause | Mitigation |
|---|---|---|---|
| **Token mask misses observation** | Reward stalls / agent learns weird behavior | Gradient flows to environment tokens | Strict per-sample action_mask |
| **Reward hacking on grader** | Benchmark rises but human eval drops | The grader has loopholes | Grader ensemble + holdout test |
| **Length-explosion** | Agent drags to context limit | Reward positively correlated with length | Length penalty + max_steps cap |
| **Tool-call hallucination** | Agent calls non-existent tools | Insufficient SFT on base model | Agent-FLAN-style negative SFT |
| **Loop / repetition** | Agent repeats the same tool | Not enough exploration | Tool-call diversity bonus + ε-tool |
| **Group $\sigma = 0$ collapse** | Advantage = NaN / 0 | All success or all fail | Data filter + std clamp |
| **KL collapse** | Policy entropy → 0 | β too small | Per-step KL + entropy bonus |

### 9.2　Online vs offline RL trade-offs

| Dimension | Online RL (PPO/GRPO) | Offline RL (DPO/RFT) |
|---|---|---|
| Data efficiency | Low (new rollouts each round) | High (one dataset, multiple epochs) |
| Training speed | Slow (rollout includes tool I/O) | Fast (pure SFT-style) |
| Performance ceiling | High (continually learns new distributions) | Medium (limited by dataset distribution) |
| Implementation complexity | High (trajectory queue + verifier server) | Low |
| Suited for | Long-term investment + real environment | Resource-constrained + existing demo data |

**Practical advice**: start with offline RL (agent SFT + DPO) to establish a baseline; once compute is available, migrate to online RL (PPO/GRPO + verifier).

### 9.3　Rollout optimization tips

- **vLLM PagedAttention** is 5-10× faster than HF generate; a must-install for agent rollout
- **Tool sandbox** with Docker + gVisor; 64-128 trajectories per machine in parallel is fine
- **Async rollout pipeline**: rollout worker doesn't block the trainer worker
- **Trajectory replay buffer**: mix FIFO + priority; weight replay by reward
- **Batch size = group_size × prompt_per_batch**: typical $G = 16$, prompt = 32 → 512 trajectories / batch

### 9.4　Debug checklist

When agent RL crashes, troubleshoot in order:

1. **Look at the reward**: do most trajectories have reward = 0? Is the rule wrong?
2. **Look at the length distribution**: are most trajectories truncated at max_steps? Means the agent can't finish
3. **Look at the action_mask**: does it correctly cover prompt + observation + padding?
4. **Look at KL**: is approx_kl blowing up? Increase β; is entropy collapsing? Increase entropy_bonus
5. **Look at group_std**: are many groups all 0 / all 1? Need a data filter
6. **Look at the tool-call distribution**: over-reliance on a single tool? Add diversity bonus
7. **Look at sample efficiency**: is the reward variance over multiple rollouts per prompt reasonable? Too large → base is too weak

## §10 25 frequently-asked interview questions (L1 must-know / L2 advanced / L3 top-lab)

Sorted into 3 tiers by difficulty: L1 = asked at any agent / LLM RL role; L2 = asked by research / alignment teams; L3 = hardcore questions for top labs. Each links to answer points + footguns.

### L1 must-know (10 questions)

<details>

<summary>Q1. What is the fundamental difference between Agentic RL and RLHF?</summary>

- **RLHF**: single-turn alignment; state = prompt, action = entire response, reward from RM scoring preference, horizon = 1
- **Agentic RL**: multi-turn decision-making; state = (obs, history), action = (thought, tool_call), reward from the external environment (test pass / task success), horizon = 10-200
- Algorithmically both use PPO/GRPO, but Agentic must add **action_mask** (loss only on agent tokens)
- Reward form: RLHF preference score (subjective); Agentic objective outcome (objective)

Treating them as the same thing; or not knowing the necessity of action_mask.

</details>

<details>

<summary>Q2. Why must Agentic RL use action_mask?</summary>

- Agent trajectories contain two kinds of tokens: those generated by the agent itself + observations returned by tools
- Without the mask, PPO/GRPO's ratio and loss flow into observation tokens
- Consequences: (a) the policy tries to "teach the tool how to respond" — meaningless and reward-hackable; (b) gradients diluted by low-information observations; (c) the KL penalty mistakenly estimates its own distribution from environment text
- Implementation: `action_mask: [B, L]`, 1 = agent token, 0 = prompt/obs/pad; loss normalizes by `mask.sum()`

Saying "only on the response" isn't specific enough (agent tasks have no clear "response" boundary); or forgetting that the mask must cover all observation tokens.

</details>

<details>

<summary>Q3. Why is GRPO better than PPO on agents?</summary>

- **Saves the critic**: agent value can hardly be learned under long horizon + sparse reward
- **Trace-level reward directly matches trace-level credit**: no per-token V needed, avoids the pain that value can't be learned
- **Intra-group normalization** does variance reduction automatically, more stable than raw advantage
- **Saves one model's VRAM**: can expand batch / group size
- Limitations: long trajectories have advantage too coarse (shared across the whole trace), credit dilution still exists

Saying only "saves the critic" is incomplete; or not knowing the alignment between trace-level reward and trace-level credit.

</details>

<details>

<summary>Q4. Difference between outcome reward and process reward? Which is common on agents?</summary>

- **Outcome reward**: 1 reward at trajectory terminal (test pass / answer match); very sparse; hard credit assignment; but low reward-hacking risk
- **Process reward**: scored per step; dense; easy credit assignment; but high reward-hacking risk (PRM can be hacked)
- **Mainstream on agents: outcome reward** — agent task ground truth is clear (test/grader); R1, SWE-RL, ReSearch, RAGEN all use outcome-only
- Process reward is mainly for reasoning-heavy tasks (PRM on math steps); rare on agents

Thinking process reward is always good (in practice outcome is more stable on agents); or not knowing the reward-hacking risk difference.

</details>

<details>

<summary>Q5. Where is verifier-based reward better than learned RM?</summary>

- **Close to ground truth**: avoids the main failure mode of learned RMs drifting outside the training distribution
- **Repeatable**: the same trajectory gets the same reward (learned RM outputs are noisy)
- **Cheap memory**: executing a verifier is hundreds of times cheaper than one LLM forward pass
- **Interpretable**: rewards come from objective rules; failures are traceable
- **Limitation**: only for verifiable tasks (math/code/grader-able web)

Saying verifier-based "has no hacking at all" — wrong, it can still be hacked via regex loopholes / format tricks / test leakage, but it's easier to plug than RM hacking.

</details>

<details>

<summary>Q6. Can the R1 / R1-Zero method be directly used on agents?</summary>

- The algorithm can be ported directly: GRPO + rule-based reward + per-step KL + token mask
- But needs additions:
  - **action_mask**: agents have observation tokens; R1 math tasks don't
  - **Trajectory rollout infrastructure**: with tool I/O, more complex than pure generation
  - **Format reward adjustment**: agent task format is JSON tool calls, not just `<think>`
- Representative works ReSearch / RAGEN / ToolRL / SWE-RL are all R1 algorithm + the above adaptations

Saying "R1 cannot be directly ported" — it can be, with wrappers; or not knowing about ReSearch / RAGEN / ToolRL.

</details>

<details>

<summary>Q7. Why is SFT warm-start needed in agent RL?</summary>

- The base model doesn't know how to emit legal tool-call schemas (JSON format / argument names)
- Direct from-scratch RL almost never explores into legal tool calls → reward all 0 → can't learn
- SFT warm-start (AgentTuning / Agent-FLAN data) lets the model "know what the action space looks like"
- Then RL optimizes within the legal-action subspace

Saying only "RL is slow, SFT speeds it up" isn't enough; the core issue is action-space exploration; SFT solves "knowing the action space".

</details>

<details>

<summary>Q8. What role does length penalty play in agent RL?</summary>

- Agents easily learn the shortcut of "drag the trajectory to get the answer correct" (reward hacking)
- Length penalty deducts for over-budget trajectories: $r = r_\text{outcome} - \lambda \max(0, T - T_\text{target})$
- DAPO's "overlong reward shaping" is an industrial implementation (piecewise-linear ramp to -1 once inside the buffer zone, rather than exponential decay)
- Limitation: too large a penalty discourages exploration; cap is needed

Not knowing that length-explosion is a common agent RL failure mode; or unable to write the length-penalty formula.

</details>

<details>

<summary>Q9. How to handle Group std = 0?</summary>

- When all G rollouts in a group have the same reward (all success / all fail) → $\sigma = 0$
- With $\epsilon$ added, advantage = 0 and the policy-gradient term zeroes out, **but the KL term still exists** (the policy is still pulled back to reference)
- Without $\epsilon$ it's NaN
- Practice:
  - **Skip the prompt** (data filter): common approach, avoids signal-less updates
  - **Clamp σ at a lower bound** (e.g. 0.1): only a numerical-stability trick to avoid $0/0$ producing NaN — in an all-identical group the numerator $r_i - \mu$ is already exactly 0, so no matter what the denominator is clamped to, the advantage stays exactly 0; this **does not recover any training signal**
  - **DAPO dynamic sampling**: drop all-correct / all-wrong groups — this is the approach that actually fixes the "no signal" problem

Saying it always NaNs (depends on implementation); or not knowing that such prompts indicate the task is too easy/hard.

</details>

<details>

<summary>Q10. How is SWE-RL trained?</summary>

- Data: GitHub PR commit history, constructing ~76M context-issue-patch triples (Meta public figure; whether this is the final triple count or the raw PR/commit event count should be checked against the original paper)
- Reward: rule-based = patch similarity (oracle ↔ pred) + binary test pass
- Algorithm: pure GRPO + format reward
- Model: Llama-3.3-70B
- Result: 41% on SWE-bench Verified (no scaffold), proving rule-based RL lets the model learn emergent reasoning: file retrieval / root cause / test self-validation

Not knowing SWE-RL's reward design; or thinking it's SFT rather than RL.

</details>

### L2 advanced (10 questions)

<details>

<summary>Q11. Derive the form of PPO loss on agents with action_mask added.</summary>

1. Standard PPO-Clip: $L = \mathbb{E}[\min(\rho_t A_t, \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon) A_t)]$
2. Agents have $m_t \in \{0, 1\}$, 1 = agent token, 0 = obs/prompt
3. ratio computation: $\rho_t = \exp((\log\pi_\theta - \log\pi_\text{old}) \cdot m_t)$; at observation positions $\rho = e^0 = 1$, doesn't affect surr1/surr2
4. Loss normalization: $L^\text{agent} = -\sum_t m_t \cdot \min(\rho_t A_t, \text{clip}) / \sum_t m_t$
5. KL term also only on agent tokens: $\text{KL}_\text{total} = \sum_t m_t \cdot \text{KL}_t / \sum_t m_t$

Writing the formula without explaining the mask's role; or forgetting that normalization uses mask.sum() rather than batch size.

</details>

<details>

<summary>Q12. Key contributions of RAGEN / StarPO?</summary>

- **StarPO** (**S**tate-**T**hinking-**A**ctions-**R**eward Policy Optimization): critic-free GRPO variant; whole trajectory shares the advantage
- **Strict token mask**: observation positions mask = 0; loss only on agent tokens
- **Rollout diversity = collapse firebreak**: group_size $G = 16$ is significantly more stable than $G = 4$
- **Trajectory length signal**: when failed trajectories are long, add a length penalty
- Applicable: multi-turn agent tasks (distinguishing from single-turn alignment)

Saying only "GRPO variant" is not enough; mention the stability contribution in multi-turn.

</details>

<details>

<summary>Q13. How does WebRL's self-evolving curriculum work?</summary>

- Start with a small task set $\mathcal{T}_0$, train the policy to roll out trajectories
- Failed trajectories enter the buffer and join the next round's curriculum $\mathcal{T}_{k+1}$
- Retrospective rollout: failed trajectories rewritten into "correct trajectories" by an LLM (hindsight relabel) and re-SFT
- ORM (Outcome Reward Model) trained from task success → gives RL reward online
- Result: Llama-3.1-8B 43% on WebArena (vs GPT-4 14.4%)

Not knowing the loop structure of self-evolving; or treating WebRL as pure SFT.

</details>

<details>

<summary>Q14. Why is self-rewarding LM more risky on agents than on alignment?</summary>

- Alignment tasks have an "objective preference distribution"; LLM judges correlate reasonably with human eval
- Agent tasks have **objective ground truth** (test pass / task success) — the judge itself can be wrong (misjudging which answers are right vs wrong)
- Iterative drift: each round reinforces "self-evaluated as correct" trajectories → drifts from ground truth
- Exploration degeneration: self-eval prefers known patterns → suppresses new-tool exploration
- Mainstream practice: agent RL prefers rule-based ground truth; self-rewarding only as open-ended task fallback

Saying self-rewarding is always dangerous (it can still be used on alignment); or not knowing that objective ground truth is the key on agents.

</details>

<details>

<summary>Q15. Derive the effect of outcome-reward sparsity on critic learning.</summary>

- Value $V_\phi(s_t) = \mathbb{E}_\pi[\sum_{l \ge 0} \gamma^l r_{t+l} \mid s_t]$
- Sparse terminal reward → $V(s_t) \approx \gamma^{T-t} \cdot P(\text{success} \mid s_t)$
- The value of an intermediate state $s_t$ depends almost entirely on "will it succeed in the future" — implicit long-horizon prediction
- **Sparse per-step reward (mostly 0) does not mean the training target is near 0**: the value MSE loss's target is the bootstrapped return, and when $\gamma$ is close to 1 (the tutorial's default $\gamma=0.99$), the target for intermediate states approximates the terminal outcome $P(\text{success} \mid s_t)$, which can stay high for long stretches; if $\gamma=1$, then for an already-successful trajectory the MC target at every step before the end is exactly the terminal reward and never approaches 0
- What actually makes the critic hard to learn is that **this target has high variance** and requires long-horizon prediction of eventual success, not that the target value or gradient is small
- This is also the theoretical basis for GRPO saving the critic: what the critic would have to learn is a high-variance, long-horizon-dependent quantity; saving it avoids that noise

Saying only "the critic can't be learned"; unable to derive $V \approx \gamma^{T-t} P(\text{success})$; or conflating "sparse per-step reward" with "value target near 0" — these are different quantities.

</details>

<details>

<summary>Q16. What is nontrivial about ToolRL's reward design?</summary>

- composite: $r = r_\text{correct} + \alpha r_\text{format} + \gamma r_\text{tool-eff}$
- $r_\text{tool-eff}$ penalizes redundant tool calls (repeated calls of the same tool / calls to invalid tools)
- A typical shaping reward: alleviates two failure modes — length-explosion + tool-overuse
- Weight balance: outcome ≫ format > tool-eff, preventing shaping from overriding outcome
- 7B approaches GPT-4 on BFCL benchmark

Saying only "add a tool-call reward"; not knowing the shaping-weight balance is the key.

</details>

<details>

<summary>Q17. How is hindsight relabeling used in agent RL?</summary>

- Failed trajectories aren't discarded; they're rewritten as successful trajectories for an "alternative task"
- Example: the agent wanted to buy item A but ended up at B → rewrite as "find item B"; reward = 1
- Implementation: `alt_task = describe(trajectory.final_state)`, relabel reward = 1
- Applicable: open web environments, navigation; not applicable: math/code (wrong answers can't become correct answers)
- Originates from HER (Andrychowicz 2017 NeurIPS) for robot manipulation

Not knowing the HER origin; or not knowing the applicability boundary (open env vs answer task).

</details>

<details>

<summary>Q18. Difference between per-step KL penalty and trajectory KL penalty?</summary>

- **Per-step KL**: compute KL($\pi_\theta(\cdot \mid s_t) \| \pi_\text{ref}(\cdot \mid s_t)$) on every agent token; added to reward or loss
- **Trajectory KL**: one KL for the entire trajectory; added to loss
- The two are not fundamentally different divergences — by the chain rule, $\log\pi(\tau) = \sum_t \log\pi(a_t \mid s_t)$, so in expectation "trajectory KL" equals the sum of per-step KLs; the real implementation differences are: (a) whether KL is folded into the reward step by step as shaping, or added as a single scalar term to the overall loss; (b) whether the aggregation uses sum or mean; (c) which numerically stable estimator is used
- GRPO sums a per-token K3 estimator (numerically stable)
- Per-step is the mainstream on agent RL (trajectories are long; taking KL directly on a sequence-level log-ratio is numerically unstable)

Treating the two as fundamentally different divergences (they're the same quantity under the chain rule; the difference is in implementation); or not knowing the K3 estimator solves numerical issues (K3: $\text{KL} \approx \exp(\Delta) - \Delta - 1$ non-negative).

</details>

<details>

<summary>Q19. How to do subgoal decomposition + process reward? When?</summary>

- Slice long trajectories into subgoals: 100-step trajectory → 5 subgoals × 20 steps
- Each subgoal terminal gets a process reward (whether the subgoal is completed)
- Implementation paths:
  - hand-crafted: humans write the subgoal criterion
  - LLM planner: a planner LLM splits subgoals, a verifier judges
  - PRM-style: a PRM scores each step
- Applicable: long-horizon agents + tasks where subgoals can be hand-crafted
- Risk: wrong subgoal boundaries → the agent learns "deliberately trigger subgoal reward without actually completing the task"

Saying process reward is always good — wrong; mention risks and limitations.

</details>

<details>

<summary>Q20. Trade-off between online and offline RL on agents?</summary>

- **Online RL (PPO/GRPO)**: low data efficiency (new rollouts each round), but continually learns new distributions
- **Offline RL (DPO/RFT)**: high data efficiency, but limited by dataset distribution
- Agent rollout is slow (includes tool I/O); online RL training throughput is low
- Practice: start offline (SFT + DPO), then online refinement
- Representatives: AgentQ is offline (MCTS + DPO); WebRL is online; SWE-RL is online

Saying only "online is slow"; not knowing that agent rollout includes tool I/O is the main bottleneck.

</details>

### L3 top-lab (5 questions)

<details>

<summary>Q21. Derive the GRPO advantage formula + the full loss with token mask, and explain the two equivalent ways to place the agent token mask.</summary>

1. **Group-relative advantage**:
   - rollout group $\{r_1, ..., r_G\}$ per prompt
   - $\mu = \frac{1}{G}\sum r_i$, $\sigma = \sqrt{\frac{1}{G}\sum (r_i - \mu)^2}$
   - $\hat{A}_i = (r_i - \mu) / (\sigma + \epsilon)$

2. **Trajectory-level broadcast**: $\hat{A}_{i,t} = \hat{A}_i$ (all agent tokens share)

3. **Token-masked ratio + loss**:
   - Naively $\rho_{i,t} = \exp(\log\pi_\theta(a_{i,t} \mid s_{i,t}) - \log\pi_\text{old}(a_{i,t} \mid s_{i,t}))$ — there's also a value at observation positions (the model assigns a probability to the env text)
   - **Key**: as long as the final objective / gradient only covers agent tokens, whether the mask sits inside the ratio or outside the loss is **mathematically equivalent**:
     - **Inside-ratio**: $\rho_{i,t} = \exp((\log\pi_\theta - \log\pi_\text{old}) \cdot m_{i,t})$ → at obs positions $\rho=1$, the $\min(\cdot)$ term inside clip = $A_{i,t}$ but is zeroed by $m_{i,t}=0$ (in the outer sum of the loss)
     - **Outside-ratio (mask loss only)**: keep the $\rho_{i,t}$ value at obs positions; the final loss = $-\sum_t m_t \cdot \min(...)$, with $m_t = 0$ at obs positions zeroing the contribution
   - **Both have gradients only on agent tokens** (mask is multiplicative; gradient w.r.t. obs positions is 0)
   - But in practice **Inside-ratio is safer**: avoids obs-position $\rho$ values participating in clip-trigger judgments or being mis-read by logging / monitoring (e.g. mean ratio) as anomalies. Production implementations (verl / OpenRLHF) mostly use inside-ratio

5. **Full loss**:
   $$L = -\frac{1}{G} \sum_i \frac{1}{\sum_t m_{i,t}} \sum_t m_{i,t} \cdot \Big(\min(\rho_{i,t} \hat{A}_i, \text{clip}(\rho_{i,t}, 1-\epsilon, 1+\epsilon) \hat{A}_i) - \beta \cdot \text{KL}_{i,t}\Big)$$

Unable to derive step 4 (mask placement affecting ratio values); or memorizing the formula but unable to explain the design philosophy of the mask.

</details>

<details>

<summary>Q22. Fundamental reason GRPO is more sample efficient than PPO on long-horizon agents?</summary>

**Natural alignment of trace-level reward with trace-level credit** (not just "saves the critic"):

1. **Critic can't learn under sparse terminal reward**: $V(s_t) \approx \gamma^{T-t} P(\text{success})$, gradient is tiny; PPO's GAE-advantage is dragged by a noisy critic
2. **GRPO uses reward normalized by the group's (including its own) mean/std as the advantage**: this is a biased-but-lower-variance estimator of sequence-level advantage that is asymptotically unbiased (consistent) as $G$ grows — sources of bias include: (a) a $(1-1/G)$ shrinkage from including the sample itself in the mean (relative to a leave-one-out baseline); (b) a nonlinear, difficulty-dependent bias introduced by std normalization; (c) additional bias from the PPO clip itself
3. **Group baseline is more stable than critic baseline**: G rollouts per prompt → group mean automatically reflects task difficulty; variance reduction is more precise
4. **PPO clipping + group size jointly limit update magnitude**: avoids a single outlier reward sending the policy flying
5. **Saving value-model VRAM** is a secondary benefit, not the primary reason
6. **Rule-based outcome reward is hard to RM-hack**: more stable than learned RM on agents

Saying only "saves the critic" — insufficient; the root cause is that the critic can't learn under sparse reward.

</details>

<details>

<summary>Q23. How to design an RL framework that supports both reasoning RL (R1) and Agentic RL (ReSearch / WebRL)?</summary>

Abstract five layers:

1. **Data layer**:
   - reasoning: (prompt, ground_truth) tuples
   - agent: (task, env_spec, reward_fn) triples
   - Unified as `Task(prompt, verifier)`; verifier is a callable

2. **Rollout layer**:
   - reasoning: generate directly
   - agent: multi-step rollout with tool I/O (vLLM + sandboxed tool executor)
   - Unified as `Trajectory(tokens, action_mask, reward)` interface

3. **Reward layer**:
   - reasoning: rule-based (answer match / test pass)
   - agent: composite (outcome + format + tool_eff + length penalty)
   - Unified as `Reward(traj) -> float`

4. **Loss layer**:
   - PPO with action_mask
   - GRPO with group_id + action_mask
   - DPO with chosen_mask / rejected_mask
   - Via `loss_fn(batch, model, ref_model) -> loss` interface

5. **Infra layer**:
   - vLLM rollout pool
   - Sandboxed tool executor (Docker + gVisor)
   - Trajectory replay buffer (FIFO + priority)
   - Async trainer / rollout

Representative implementations: **verl** (ByteDance) already supports reasoning + agent; **OpenRLHF** partially supports.

Listing only PPO without considering agent rollout infra; or not knowing the current support scope of verl / OpenRLHF.

</details>

<details>

<summary>Q24. Anthropic Computer-Use training method — clear boundary between known vs speculated</summary>

**Officially public (system card / blog)**:

- Action space = screenshot (vision observation) + mouse + keyboard events
- Capability iterates: Claude 3.5 (new) 2024-10-22 → 3.7 / 4.0 / 4.5 / Opus 4.x
- Safety: constitutional-AI-style guardrails + red-teaming + prompt-injection defense
- Training involves human demonstrations + synthetic data (general statement in system card)

**Not public / fully confidential**:

- Specific RL algorithm (PPO? GRPO? Critic-free? Never said)
- Reward signal form (task-completion grader? pair-wise preference? safety classifier weight?)
- Training data scale / source / demonstration vs synthetic ratio
- Whether there's a dedicated screenshot RM / VLM-as-judge

**Reasonable community speculation** (**speculation only, do not state as fact in interviews**):

- May be RLHF on screenshot trajectories (pair-wise preference + task-success outcome mixture)
- Possibly critic-free (echoing open-source trends like DeepSeek-R1 GRPO)
- May use VLM-as-judge for screenshot understanding
- Curriculum may be simple → complex

**In interviews, strictly distinguish "public capability" from "speculated internals"**: saying "Anthropic uses GRPO + screenshot RM" is wrong (no evidence); saying "I speculate it may use critic-free RL because Anthropic tends toward GRPO/RLHF-style in other scenarios" is the honest framing. This ability to distinguish is a plus in advanced interviews.

</details>

<details>

<summary>Q25. If asked to design a next-gen Agentic RL algorithm, how would you improve it?</summary>

Possible directions (answer 3-4; each with a trade-off discussion):

1. **Lightweight critic for long-horizon**: not a full-size value model, but a small step-level critic to alleviate trace-level credit dilution. VAPO has tried this. Trade-off: extra VRAM vs alleviating long-trajectory signal dilution

2. **Hierarchical reward**: subgoal-level reward + outcome reward combination. Trade-off: requires subgoal definition (humans or planner LLM), risk of wrong boundaries

3. **Off-policy correction with V-trace / Retrace**: rollout is slow; let stale samples be used. Trade-off: IS bias vs sample efficiency

4. **Trajectory hindsight relabeling + RL**: failed trajectories auto-rewritten as successful trajectories for alternative tasks, expanding data. Trade-off: applies to open-ended tasks, not closed-form answers

5. **Multi-task reward normalization**: independent normalization per task domain (math/code/web), avoiding reward-scale imbalance

6. **Reward model uncertainty**: multi-RM ensembles, min/mean-std to prevent over-optimization. Trade-off: compute

7. **Async distributed rollout**: rollout and train fully async, trajectory queue + worker pool. Already industry default (verl, OpenRLHF v0.5+)

8. **Self-curriculum + adaptive difficulty**: WebRL idea + R-Zero learnability reward combined; automatically find tasks where model success rate is ~50%

9. **Multi-objective Pareto optimization**: no longer a single scalar reward; jointly optimize task success + safety + efficiency, output Pareto front

Just listing "add attention / add more models" without trade-offs; or not knowing recent work like DAPO / VAPO / CISPO; or ignoring the importance of the infra layer (async rollout).

</details>

## §A Appendix: reference list

Grouped by direction; papers verified via web search + arXiv for authors / year / venue. A few 2025-2026 papers without a fixed conference are listed by arXiv.

**Agent SFT / fundamentals**

- Zeng et al. 2023 arXiv 2310.12823 *AgentTuning: Enabling Generalized Agent Abilities for LLMs* (THU)
- Chen et al. 2024 ACL Findings arXiv 2403.12881 *Agent-FLAN: Designing Data and Methods of Effective Agent Tuning for LLMs*
- Luong et al. 2024 ACL arXiv 2401.08967 *ReFT: Reasoning with Reinforced Fine-Tuning*

**RL for agent / reasoning (fundamental algorithms)**

- Schulman et al. 2017 arXiv 1707.06347 *Proximal Policy Optimization Algorithms*
- Schulman et al. 2016 ICLR *High-Dimensional Continuous Control Using GAE*
- Shao et al. 2024 arXiv 2402.03300 *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models* (GRPO proposed)
- DeepSeek-AI 2025 arXiv 2501.12948 *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*
- Yu et al. 2025 ByteDance arXiv 2503.14476 *DAPO: An Open-Source LLM Reinforcement Learning System at Scale*

**Tool-augmented RL**

- Qian et al. 2025 arXiv 2504.13958 *ToolRL: Reward is All Tool Learning Needs* (arXiv preprint; no formal venue as of 2026-05)
- Chen et al. 2025 arXiv 2503.19470 *ReSearch: Learning to Reason with Search for LLMs via Reinforcement Learning* (accepted to **NeurIPS 2025**)
- Wang et al. 2025 arXiv 2504.20073 *Understanding Self-Evolution in LLM Agents via Multi-Turn Reinforcement Learning* (RAGEN; StarPO = **S**tate-**T**hinking-**A**ctions-**R**eward Policy Optimization)

**Web / GUI agent RL**

- Qi et al. 2024 ICLR-25 arXiv 2411.02337 *WebRL: Training LLM Web Agents via Self-Evolving Online Curriculum Reinforcement Learning*
- Putta et al. 2024 arXiv 2408.07199 *Agent Q: Advanced Reasoning and Learning for Autonomous AI Agents*
- Furuta et al. 2024 ICLR arXiv 2305.11854 *Multimodal Web Navigation with Instruction-Finetuned Foundation Models* (WebGUM)

**Code agent RL**

- Le et al. 2022 NeurIPS arXiv 2207.01780 *CodeRL: Mastering Code Generation through Pretrained Models and Deep Reinforcement Learning*
- Shojaee et al. 2023 arXiv 2301.13816 *Execution-Based Code Generation Using Deep Reinforcement Learning* (PPOCoder)
- Wei et al. 2025 Meta FAIR arXiv 2502.18449 *SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Evolution*

**Embodied / robot agent**

- Baker et al. 2022 NeurIPS arXiv 2206.11795 *Video PreTraining (VPT): Learning to Act by Watching Unlabeled Online Videos*
- Kim et al. 2024 arXiv 2406.09246 *OpenVLA: An Open-Source Vision-Language-Action Model*

**Self-rewarding / exploration**

- Yuan et al. 2024 ICML arXiv 2401.10020 *Self-Rewarding Language Models*
- Andrychowicz et al. 2017 NeurIPS arXiv 1707.01495 *Hindsight Experience Replay*

**RLHF / DPO fundamentals (cross-reference)**

- Ouyang et al. 2022 NeurIPS *Training Language Models to Follow Instructions with Human Feedback*
- Rafailov et al. 2023 NeurIPS *Direct Preference Optimization*
- Bai et al. 2022 Anthropic arXiv 2212.08073 *Constitutional AI*
- Lee et al. 2023 Google arXiv 2309.00267 *RLAIF: Scaling RLHF with AI Feedback*

**Reward model / verification**

- Lightman et al. 2024 ICLR arXiv 2305.20050 (OpenAI 2023) *Let's Verify Step by Step* (PRM800K)
- Wang et al. 2024 ACL arXiv 2312.08935 *Math-Shepherd: Verify and Reinforce LLMs Step-by-Step without Human Annotations*
- Coste et al. 2024 ICLR *Reward Model Ensembles Help Mitigate Overoptimization*

**Infrastructure / frameworks**

- TRL (HuggingFace): https://github.com/huggingface/trl — standard PPO / DPO / GRPO trainer
- OpenRLHF: https://github.com/OpenRLHF/OpenRLHF — industrial-grade PPO / GRPO / RLOO implementation
- verl (ByteDance): https://github.com/volcengine/verl — mainstream framework for GRPO / DAPO / agent RL
- ReaLHF / AReaL (Ant Group + Tsinghua, async RL system, arXiv 2505.24298)

**SOTA benchmarks (2024-2026)**

- Jimenez et al. 2024 ICLR arXiv 2310.06770 *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?*
- Yao et al. 2024 arXiv 2406.12045 *τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains*
- Xie et al. 2024 NeurIPS arXiv 2404.07972 *OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments*
- Zhou et al. 2024 ICLR arXiv 2307.13854 *WebArena: A Realistic Web Environment for Building Autonomous Agents*
- Mialon et al. 2024 ICLR *GAIA: A Benchmark for General AI Assistants*
- Chan et al. 2024 arXiv 2410.07095 *MLE-bench: Evaluating Machine Learning Agents on Machine Learning Engineering* (OpenAI)

**Anthropic Computer-Use (public knowledge)**

- Claude 3.5 Sonnet (new) 2024-10-22 Computer Use beta launch (Anthropic blog + system card)
- Claude 3.7 / 4.0 / 4.5 / Opus 4.x system cards (Anthropic public)

Code framework recommendations:

- Get started with HF TRL's GRPOTrainer + your own verifier
- For industrial use, use verl (GRPO/RLOO/DAPO all supported, includes agent rollout)
- For custom multi-turn use OpenRLHF v0.5+ agent examples + add a tool sandbox
