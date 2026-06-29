## §0 TL;DR Cheat Sheet

> 💡 **Reasoning models in 8 sentences** — The biggest paradigm shift in LLMs from 2024-2026; one page covering interview essentials.

1. **Paradigm shift**: previously we scaled **training compute** (parameters + data); now we scale **inference compute** (reasoning tokens / search / verification). Snell et al. 2024 (arXiv 2408.03314) gave the **compute-optimal test-time scaling** recipe: under the same inference FLOPs, a hybrid strategy of best-of-N + PRM beam search + sequential revision is **>4×** more efficient than pure best-of-N; under FLOPs-matched settings, small model + optimized test-time compute can match or exceed a **14×** larger model on certain tasks.

2. **o1 (OpenAI Sep 2024)**: uses RL to train hidden chain-of-thought; the API only returns a `reasoning_tokens` count, not content. **o3 (Dec 2024)** scored 75.7% (low compute) / 87.5% (high compute, 172× budget) on ARC-AGI—the first time abstract reasoning benchmarks approached human-level performance.

3. **DeepSeek-R1-Zero (arXiv 2501.12948, Jan 2025)**: **pure RL from a base model**, no SFT cold start, rule-based reward (answer correct/wrong + format), using **GRPO** (no critic); the "aha moment" emerged—the model learned to reflect, backtrack, and verify on its own.

4. **DeepSeek-R1**: four-stage pipeline = SFT cold start (thousands of high-quality CoT) → reasoning-oriented RL → rejection sampling + general SFT → all-scenario RL. Matched o1 on math/code benchmarks like MATH-500 and AIME.

5. **GRPO (DeepSeekMath, arXiv 2402.03300)**: removes the critic value network; for each prompt, sample $G$ responses $\{o_i\}$ and replace GAE with **group-relative advantage** $A_i = (r_i - \text{mean}(\mathbf{r})) / \text{std}(\mathbf{r})$. Halves memory and stabilizes training.

6. **PRM vs ORM**: ORM (outcome reward) only scores the final answer; PRM (process reward, Lightman 2023 "Let's Verify Step by Step") scores each reasoning step, allowing best-of-N to select better traces. Math-Shepherd (Wang et al. 2023, arXiv 2312.08935) samples **Monte Carlo completion rollouts** from intermediate steps (not MCTS tree search), uses soft/hard estimation from final-answer correctness to auto-label step labels, eliminating human annotation.

7. **s1 (Muennighoff Feb 2025, arXiv 2501.19393)**: "Wait" budget forcing—forcibly append "Wait" at `</think>` to make the model continue thinking; 1K-sample SFT + inference control surpasses o1-preview by 27% (AIME24).

8. **Common pitfalls**: CoT ≠ reasoning (the model may post-hoc fabricate stories); self-consistency breaks on distribution-shifted problems; PRM training easily overfits step patterns; on long-CoT, GRPO's critic-free design is actually an advantage (critic is hard to learn).

## §1 Why This Is the Biggest Paradigm Shift of 2024-2026

### 1.1　From train-time scaling to test-time scaling

The 2020-2023 scaling laws (Kaplan, Hoffmann/Chinchilla): **performance ∝ log(params × data × FLOPs)**—but this is **training-time compute**. Once a model is trained, inference compute is fixed (one forward pass).

Two anomalies appeared in 2024:
- **OpenAI o1**: spending more inference tokens to think → performance keeps improving, **showing log-linear scaling** (OpenAI's public figure: accuracy vs reasoning compute is a straight line)
- **Snell et al. 2024**: under fixed inference/test-time compute budget, using compute-optimal scaling (best-of-N + PRM beam search + sequential revision hybrid) lets small models in FLOPs-matched settings **match or even exceed 14× larger models**—provided the base model already has non-trivial success rate on the task

> 💡 **Paradigm mental model** — treat reasoning as **search over reasoning paths**:

- **System 1** (fast): one greedy decode = one path
- **System 2** (slow): multiple paths + verifier + backtrack = tree search (Tree-of-Thought) or long CoT (o1)

Think of LLMs as a **policy + value** combination (like AlphaZero); inference is an MCTS-like search process.

### 1.2　Core components of reasoning models

```

       problem prompt
           │
           ↓
   ┌───────────────────────────┐
   │   Policy LLM (sampler)    │ ← generate candidate reasoning traces
   │   - greedy / temperature  │
   │   - long CoT (o1, R1)     │
   │   - tree expansion (ToT)  │
   └───────────────────────────┘
           │  N traces
           ↓
   ┌───────────────────────────┐
   │  Verifier / Reward Model  │
   │  - ORM (outcome only)     │
   │  - PRM (per-step)         │
   │  - rule-based (math/code) │
   └───────────────────────────┘
           │
           ↓
   ┌───────────────────────────┐
   │  Aggregator               │
   │  - majority vote (SC)     │
   │  - best-of-N (verifier)   │
   │  - beam search (PRM)      │
   │  - MCTS (rStar)           │
   └───────────────────────────┘
           │
           ↓
       final answer
```

Different reasoning model lines essentially make choices among these three components:
- **o1 / R1**: internalize search into the policy (long CoT single-pass generation, policy reflects on its own)
- **ToT / rStar**: external explicit search (MCTS / BFS / beam)
- **Best-of-N + PRM**: sampling + verifier (most naive test-time scaling)

## §2 CoT Evolution: from Wei 2022 to o1

### 2.1　Chain-of-Thought (Wei et al., NeurIPS 2022, arXiv 2201.11903)

Core finding: **few-shot prompts that demonstrate "step-by-step reasoning"** trigger emergent reasoning ability on large models (>62B PaLM); GSM8K jumps from 18% → 57%.

```
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?
A: Roger started with 5 balls. 2 cans of 3 balls each is 6 balls. 5 + 6 = 11.
The answer is 11.
```

Key points:
- **Not fine-tuning**—pure prompting; the base model already has the capability
- **Emergence**: on small models, CoT actually **degrades** (noise exceeds signal)
- Kojima et al. 2022 "Let's think step by step" later showed zero-shot CoT also works

### 2.2　Self-Consistency (Wang et al. 2022, arXiv 2203.11171)

Observation: CoT decoding is stochastic (temperature > 0); sampling the same problem multiple times yields different reasoning paths, but **the correct answer usually appears more often** (if the model is capable enough).

Algorithm:
1. Sample $N$ CoTs for the same prompt
2. Extract the final answer from each
3. **Majority vote** for the majority answer

$$\hat{y} = \arg\max_{y \in \mathcal{Y}} \sum_{i=1}^{N} \mathbb{1}[\text{extract}(\text{trace}_i) = y]$$

Result: GSM8K +17.9% (Wang et al.'s reported PaLM-540B number).

> ⚠️ **Common bug** — must perform **answer extraction normalization** before voting (strip units, simplify fractions, normalize integers). Otherwise "1/2" and "0.5" get counted as different answers, splitting the majority.

### 2.3　Tree of Thoughts (Yao et al., NeurIPS 2023, arXiv 2305.10601)

Upgrade CoT from a **chain** to a **tree**: each node is a "thought" (one reasoning step); from the root, BFS/DFS expands multiple candidate children, **the LLM scores them itself** ("is this step promising?"), keeping top-k.

```

          root (problem)
           │
   ┌───────┼───────┐
  step1a  step1b  step1c        ← sample k thoughts
   │       ✗       ✗            ← LLM evaluator score
   │       prune  prune
   ├──────┼──────┐
 step2a step2b step2c           ← expand surviving node
```

Game of 24 (use 4 numbers + arithmetic ops to get 24):
- **GPT-4 CoT**: 4% success rate
- **GPT-4 ToT** (b=5, depth=3): **74%**—huge jump

> 💡 **The essential difference between ToT and CoT** — CoT is **autoregressive decoding** (one path); ToT is **deliberate search** (multiple paths + explicit backtracking + evaluator). The former is fast but easily trapped by early errors; the latter is slow but can escape local optima.

### 2.4　From ToT to long-CoT (o1 / R1 route)

ToT requires an external search framework (recursive prompting + state management). **o1 / R1 take a different route**: they **train search into the policy itself**—a single long CoT, but the model learns on its own to:
- "wait, let me reconsider..."
- "actually that's wrong, the correct way is..."
- "let me verify by trying a different approach"

These backtracking / reflection tokens are rare in base models; only via **repeated sampling + reward signal** in RL can they be amplified into stable behavior. R1-Zero's reported "aha moment" is when this behavior suddenly emerges during RL training.

## §3 PRM vs ORM: Two Routes for Reasoning Verifiers

### 3.1　Definition

| Dimension | ORM (Outcome Reward Model) | PRM (Process Reward Model) |
| --- | --- | --- |
| Supervision granularity | one reward per whole trace | one reward per step |
| Label source | answer correct → +1, wrong → 0 | human-labeled (PRM800K) or MC completion rollout estimated (Math-Shepherd) |
| Training objective | $\max \mathbb{E}[r(\text{trace})]$ | $\max \sum_t \mathbb{E}[r_t(\text{step}_t)]$ |
| Advantages | cheap to label (only ground-truth answer needed) | dense signal; can localize wrong steps |
| Disadvantages | sparse reward, hard credit assignment | expensive labeling; step boundary hard to define |

### 3.2　PRM800K (Lightman et al. 2023, arXiv 2305.20050)

OpenAI human-labeled 800K step-level labels on the MATH dataset: each step is `positive` / `neutral` / `negative`.

Training objective (per-step classification):

$$\mathcal{L}_\text{PRM} = -\sum_{t=1}^{T} \log p_\phi(\ell_t \mid s_{\leq t})$$

where $\ell_t \in \{+, 0, -\}$, $s_{\leq t}$ is the first $t$ reasoning steps.

At inference, use PRM as verifier:

$$\text{score}(\text{trace}) = \prod_{t} p_\phi(\ell_t = +\mid s_{\leq t}) \quad \text{or} \quad \min_t p_\phi(\ell_t = +\mid s_{\leq t})$$

(The product form is more standard; the min form is more pessimistic but catches the weakest step.)

> ✅ **Key finding** — Lightman 2023 reported: using PRM800K as verifier in best-of-1024, achieves 78% on MATH test (vs 72% ORM, 70% majority vote). **PRM > ORM > self-consistency**, but at the cost of 800K human annotations.

### 3.3　Math-Shepherd (Wang et al. 2023, arXiv 2312.08935) — auto-labeling step labels

Human labels are expensive; how to scale? Math-Shepherd's idea: **use Monte Carlo completion rollouts to estimate each step's "potential correctness rate"**.

```
For each step s_t (partial reasoning):
  rollout K completions from s_t
  count how many final answers are correct
  reward_t = (correct count) / K
```

Intuition: if a step is good, walking down from it should easily yield a correct answer; if bad, no path will succeed.

Train PRM with Math-Shepherd labels: MSE regression or BCE classification. Mistral-7B on GSM8K: 77.9% → 84.1%, no human labels needed.

> 💡 **Implicit assumption of MCTS-labels** — the base model must have non-trivial success rate on the task (otherwise all rollouts fail and all labels are 0). So PRM training requires the base model to solve at least some problems—this is a bootstrap problem.

### 3.4　Generative PRM and Critic LM

After 2024, "generative verifiers" became popular: treat PRM as **next-token-prediction** ("is this step correct? yes/no"), directly reusing the LLM architecture without a separate reward head. Representative work: Generative Verifiers (Zhang et al. 2024, arXiv 2408.15240). Advantage: can leverage in-context reasoning for scoring, more accurate than scalar-head PRM.

## §4 Best-of-N + Verifier: The Most Naive Test-Time Scaling

### 4.1　Formula

Given prompt $x$, policy $\pi$, verifier $V$, sample count $N$:

$$i^{*} = \arg\max_{i \in [N]} V(\text{trace}_i), \quad \text{trace}_i \sim \pi(\cdot \mid x); \quad \hat{y} = \text{extract}(\text{trace}_{i^{*}})$$

Theoretical limit ("oracle BoN", assuming perfect verifier):

$$\text{pass}@N = 1 - (1 - \text{pass}@1)^N$$

Reality: the verifier is imperfect, so BoN's saturation curve is far below the oracle. Snell 2024 found:
- On **easy problems**, BoN quickly saturates (marginal returns small after N=4)
- On **hard problems**, BoN keeps improving up to N=64 / 128
- **Compute-optimal**: hard problem → increase N; easy problem → greedy

### 4.2　Best-of-N + PRM code

```python
import torch
from typing import Callable

def best_of_n_prm(
    policy_sample: Callable,          # prompt -> (trace, step_list)
    prm_score: Callable,              # step_list -> scalar in [0, 1]
    prompt: str,
    n: int = 16,
    aggregation: str = "min",         # "min" | "prod" | "mean"
) -> tuple[str, float]:
    """
    Best-of-N with process reward model verifier.
    Returns (best_trace, best_score). Note: in `prod` mode, `best_score`
    is the cumulative log-product (negative); compare scores within the same
    aggregation mode only.
    """
    best_trace, best_score = None, -float("inf")
    for _ in range(n):
        trace, steps = policy_sample(prompt)
        step_probs = [prm_score(steps[: t + 1]) for t in range(len(steps))]
        if aggregation == "min":
            score = min(step_probs)
        elif aggregation == "prod":
            # log-sum to avoid numerical underflow
            score = sum(torch.log(torch.tensor(p) + 1e-9) for p in step_probs).item()
        elif aggregation == "mean":
            score = sum(step_probs) / len(step_probs)
        else:
            raise ValueError(aggregation)
        if score > best_score:
            best_score, best_trace = score, trace
    return best_trace, best_score
```

> ⚠️ **PRM aggregation choice** — Lightman 2023 experiments: **min** and **prod** are close on BoN; **mean** is significantly worse (high-scoring steps mask low-scoring wrong steps). Production typically uses **min**—intuition: "trace strength is determined by its weakest link".

### 4.3　Self-Consistency code

```python
from collections import Counter
from typing import Callable
import re

def _extract_braced(s: str, open_idx: int) -> str | None:
    """From the first `{` after `\boxed{`, extract the inner content via balanced brace matching (supports nested LaTeX)."""
    if open_idx >= len(s) or s[open_idx] != "{":
        return None
    depth = 0
    for i in range(open_idx, len(s)):
        if s[i] == "{": depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return s[open_idx + 1:i]
    return None  # unclosed

def extract_answer(trace: str) -> str:
    """Extract the final answer from a trace. Note: production needs more thorough normalization (units, fractions→decimals, LaTeX simplification, etc.)."""
    # Use balanced brace matching for `\boxed{...}` to avoid truncating `\boxed{\frac{1}{2}}` to `\frac{1`
    pos = trace.find(r"\boxed")
    if pos != -1:
        inner = _extract_braced(trace, pos + len(r"\boxed"))
        if inner is not None:
            return inner.replace(",", "").replace("$", "").strip()
    m = re.search(r"answer is[:\s]+([^.\n]+)", trace, re.IGNORECASE)
    if m:
        return m.group(1).strip().replace(",", "").replace("$", "").strip()
    return ""

def self_consistency(
    policy_sample: Callable,
    prompt: str,
    n: int = 40,
    temperature: float = 0.7,
) -> tuple[str, dict]:
    """Self-Consistency: sample N traces, majority vote on extracted answers."""
    answers = []
    for _ in range(n):
        trace = policy_sample(prompt, temperature=temperature)
        ans = extract_answer(trace)
        if ans:                            # skip parse failures
            answers.append(ans)
    if not answers:
        return "", {}
    counts = Counter(answers)
    return counts.most_common(1)[0][0], dict(counts)
```

> ✅ **Production details** — Wang 2022 reports on GSM8K: returns saturate around N=40; N=64 is near the limit. Temperature too high (>1.0) introduces garbage reasoning; too low (<0.3) makes multiple samples nearly identical. **Sweet spot is usually 0.5-0.7**.

## §5 The RL Route: PPO → GRPO

### 5.1　Vanilla PPO recap (Schulman et al. 2017)

For each token $t$, policy gradient with clipping:

$$\mathcal{L}_\text{PPO} = -\mathbb{E}_t \left[ \min\!\left( \rho_t A_t,\; \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon) A_t \right) \right]$$

where
- $\rho_t = \pi_\theta(o_t \mid s_t) / \pi_{\theta_\text{old}}(o_t \mid s_t)$ is the importance ratio
- $A_t$ is the advantage, commonly GAE-$\lambda$: $A_t = \sum_{l\geq 0} (\gamma\lambda)^l \delta_{t+l}$, $\delta_t = r_t + \gamma V_\psi(s_{t+1}) - V_\psi(s_t)$

**Pain points**:
1. Need to train a **value network** $V_\psi$ (critic), usually the same size as the policy—memory doubles
2. Critic training on long-CoT is hard to converge (reward is extremely sparse, only final answer right/wrong)
3. GAE requires token-level value estimation; on long-CoT (4K+ tokens), each token's value estimate is very noisy

### 5.2　GRPO (Shao et al. 2024, DeepSeekMath, arXiv 2402.03300)

Core idea: **use group statistics to replace the critic**.

Algorithm:
1. For each prompt $x$, sample $G$ completions $\{o_1, \dots, o_G\}$ from $\pi_{\theta_\text{old}}$ (typically $G = 16$ or $64$)
2. Use a reward model to score each completion $\{r_1, \dots, r_G\}$
3. **Group-relative advantage** (trace-level, not token-level):

$$\boxed{\;A_i = \frac{r_i - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r}) + \epsilon}\;}$$

4. Broadcast trace-level $A_i$ to all tokens in that trace: $A_t = A_i \;\forall t \in o_i$
5. Update the policy with the PPO clipping objective (same formula as above, but $A_t$ comes from steps 3-4); DeepSeekMath/R1 also add KL regularization $\beta \cdot \mathrm{KL}(\pi_\theta \,\|\, \pi_\text{ref})$. In practice, use Schulman's **unbiased k3 estimator** $\widehat{\mathrm{KL}}_t = e^{\log\pi_\text{ref}(o_t|s_t) - \log\pi_\theta(o_t|s_t)} - (\log\pi_\text{ref}(o_t|s_t) - \log\pi_\theta(o_t|s_t)) - 1$ per token (always $\geq 0$), then mask + average

> ✅ **GRPO's key insight** — Why is group-relative more useful than a critic?

- **Same prompt, same source**: $G$ completions share prompt difficulty; differences come entirely from policy output; mean automatically subtracts prompt-specific baseline, equivalent to a control variate
- **No value network needed**: directly saves half memory + saves a model's worth of compute; the critic on long-CoT is hard to learn anyway (sparse reward + long episodes)
- **Stability comes from group size $G$**: larger $G$ → lower advantage estimation variance; DeepSeek-R1 uses $G \approx 16$

### 5.3　GRPO advantage computation code

```python
import torch

def grpo_advantage(rewards: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Group-relative advantage estimation.

    Args:
        rewards: [G] tensor of trace-level rewards (one prompt's group).
        eps: numerical stability.

    Returns:
        advantages: [G] tensor, mean ≈ 0, std ≈ 1.
    """
    mu = rewards.mean()
    sigma = rewards.std(unbiased=False)        # use biased std (divide by G, not G-1)
    return (rewards - mu) / (sigma + eps)


def grpo_loss(
    log_probs_new: torch.Tensor,               # [G, T] new policy log p
    log_probs_old: torch.Tensor,               # [G, T] old policy log p (detached)
    log_probs_ref: torch.Tensor,               # [G, T] reference policy log p
    rewards: torch.Tensor,                     # [G] trace-level
    mask: torch.Tensor,                        # [G, T] valid token mask
    clip_eps: float = 0.2,
    kl_beta: float = 0.04,
) -> torch.Tensor:
    """Single-prompt GRPO loss (group size = G traces).
    Requires `log_probs_old` and `log_probs_ref` to both be detached (no gradient);
    if the caller passes a tensor with gradient, it will incorrectly backprop into the old/reference policy.
    """
    adv = grpo_advantage(rewards)              # [G]
    adv = adv.unsqueeze(-1)                    # [G, 1] broadcast to tokens

    # Defensive detach: even if the caller forgot, old / ref policy won't be backproped
    log_probs_old = log_probs_old.detach()
    log_probs_ref = log_probs_ref.detach()

    # PPO clipping
    ratio = torch.exp(log_probs_new - log_probs_old)        # [G, T]
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
    pg_loss = -torch.min(surr1, surr2)                       # [G, T]

    # KL regularization: constrain policy not to drift too far from reference
    # DeepSeekMath/R1 use Schulman's unbiased k3 estimator:
    #   KL ≈ exp(log_ref - log_new) - (log_ref - log_new) - 1   ≥ 0
    # This is a sample unbiased estimator of KL(π_θ || π_ref), always non-negative, much more stable than raw log-ratio
    log_diff = log_probs_ref - log_probs_new               # [G, T]
    kl = torch.exp(log_diff) - log_diff - 1.0              # [G, T], ≥ 0
    loss = pg_loss + kl_beta * kl                          # [G, T]

    # Masked mean over valid tokens
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)
```

> ⚠️ **Common bugs** — Several pitfalls in GRPO implementation:

- **`log_probs_old` must be detached** (not part of gradient), otherwise it forms a strange gradient path with `ratio`
- **`std` biased vs unbiased** has small impact but must be consistent (DeepSeek's public version uses biased)
- **When all $G$ rewards are identical (all wrong or all correct)**, `std = 0` and $r_i - \mu \equiv 0$; after adding `eps`, advantage = 0 → policy-gradient term zeros out; but the KL regularization term still exists, so total loss degenerates to `kl_beta * KL` (still pulling policy back to reference). In production, usually skip such prompts (data filtering) to avoid signal-less updates, or clamp std to a lower bound (e.g., 0.1)
- **Mask must cover all padding tokens**, otherwise padded log-probs pollute the loss

### 5.4　Why GRPO is sample efficient (L3 frequently-asked)

Observation: R1 paper's publicly disclosed ~800K samples (≈ 600k reasoning + 200k non-reasoning) is the **Stage 3 rejection sampling + SFT data**, not RL prompt count; R1 Stage 2 reasoning RL prompt count is not strictly disclosed, but overall training compute (~147K H800-hours) is still significantly less than contemporary RLHF (InstructGPT used millions of human preference pairs); GRPO is the core algorithmic contribution.

**Root causes** (beyond just "saving the critic"):
1. **Trace-level reward naturally aligns with trace-level credit**: GRPO directly broadcasts trace reward as advantage to all tokens; no critic needed for per-token credit assignment. Under settings where reward only sees the final answer, this is the standard policy-gradient estimate of sequence-level Monte-Carlo return (not "optimal", but trace-level is more stable than the critic's noisy per-token V on long-CoT)
2. **Contrast within the same prompt automatically eliminates prompt-level noise**: advantage = $(r - \mu) / \sigma$ stabilizes the advantage distribution; policy updates move in more accurate directions; equivalent to paired comparison within each prompt
3. **PPO clipping limits single-step update magnitude**: together with $G$, ensures the policy doesn't blow up due to a single prompt's outlier reward
4. **Rule-based reward is hard to be RM-hacked**: r1 uses rule reward (answer regex matching + format `<think></think>`), no learned RM to over-optimize; advantage primarily reflects "did you answer correctly". Note rule rewards are not completely unhackable—policy can still exploit regex loopholes, format tricks, or test-set leakage—but they avoid the main failure mode of learned-RM drifting off the training distribution

R1 training GPU hours ≈ 147K H800-hours (DeepSeek public number)—two orders of magnitude smaller than GPT-4 training; the result of GRPO + data refinement + rule reward acting together.

## §6 Complete DeepSeek-R1-Zero / R1 Pipeline

### 6.1　R1-Zero: extreme experiment of pure RL without SFT

**Only assumption**: the base model (DeepSeek-V3-Base, 671B MoE) already has basic language + math capability.

**Training**:
- Reward = `accuracy_reward + format_reward` (rule-based, no RM)
  - `accuracy_reward`: whether the answer can be auto-extracted + matches ground truth
  - `format_reward`: whether reasoning is wrapped in `<think>...</think>`
- Algorithm: GRPO, $G = 16$, KL $\beta = 0.001$
- Data: math (MATH, AIME, etc.) + code (LeetCode-like executable evaluation)

**Emergent behaviors** ("aha moment", DeepSeek-R1 paper Fig 3):
- After a few K steps, CoT length spontaneously grows (200 tokens → thousands of tokens)
- Self-reflection phrases like "Wait, let me reconsider...", "Actually that's wrong", "Let me verify by..." appear
- On AIME 2024, pass@1 goes from base 15.6% → 71.0% (far exceeding GPT-4o-0513's 9.3%)

> 💡 **Historical significance of R1-Zero** — Before R1-Zero, the community widely believed reasoning capability had to come from **SFT cold start** (first show the model many demonstrated CoTs) before RL could bootstrap it. R1-Zero proved that base model + pure rule reward can do it—this is the first large-scale reproduction of "RL from scratch eliciting reasoning" in LLM training history (the LLM equivalent of AlphaGo Zero, though the base was already pretrained).

**Problem**: R1-Zero's output has **readability issues**—reasoning process language mixed (Chinese + English + math symbols jumping around), sometimes not paragraphed, hard for humans to read. So R1 added subsequent stages.

### 6.2　R1's complete 4-stage pipeline

| Stage | Input | Method | Goal |
| --- | --- | --- | --- |
| **Stage 1: Cold-start SFT** | DeepSeek-V3-Base | Thousands of curated long-CoT (partly from R1-Zero output + human readability corrections) | Have the model learn "human-readable reasoning format" |
| **Stage 2: Reasoning-oriented RL** | Stage 1 model | GRPO + rule-based reward + language consistency reward (penalize Chinese-English mixing) | Improve reasoning ability |
| **Stage 3: Rejection sampling + SFT** | Stage 2 model | Use Stage 2 model to mass-sample → filter by PRM/rules → 600K reasoning + 200K general data, re-do SFT | Extend to non-math/code domains; preserve general ability |
| **Stage 4: All-scenario RL** | Stage 3 model | GRPO + (rule reward for math/code) + (RM for helpfulness/harmlessness) | All-scenario alignment |

**Core insights**:
- **Stage 1 = readability injection**, not for reasoning (reasoning comes from RL)
- **Stage 3 = generalization injection**, transferring reasoning capability learned from math/code RL to non-verifiable domains (writing, dialogue, QA)
- **Stage 4 = safety + helpfulness alignment**, equivalent to closing RLHF

### 6.3　R1-Distill: distilling R1 into smaller models

DeepSeek used Stage 3 data (600K reasoning samples) to do **pure SFT** (no RL) on Qwen2.5-{1.5B, 7B, 14B, 32B} and Llama3-{8B, 70B}, producing the R1-Distill series.

**Key findings** (paper Table 5):
- DeepSeek-R1-Distill-Qwen-32B on AIME 2024: 72.6 vs o1-mini's 63.6—**SFT-distilled small model surpasses o1-mini**
- 1.5B model on MATH-500: 83.9, far exceeding original Qwen2.5-Math-1.5B's 51.0
- The public R1-Distill **only did SFT, no additional RL**; authors explicitly note "incorporating RL could substantially boost performance"—so "whether distill + RL gives more" is an open question, not "can't be done"

> ⚠️ **L3 frequently-asked** — "Why is R1-Distill far better than the original model, but doing RL directly on the small model is worse?"

- DeepSeek-R1-Distill-Qwen-32B (SFT only) > DeepSeek-Qwen-32B-RL (direct RL from scratch on Qwen)
- Reason: small model base is too weak; pure RL hardly causes reasoning to emerge; using R1 generated reasoning traces for SFT is equivalent to **distillation through demonstrations**, directly copying the "reasoning behavior pattern".
- Implication: **emergent reasoning needs big base + strong RL; on small models, distillation copy is the most economical route**.

## §7 Test-Time Scaling: Snell 2024 and s1 Budget Forcing

### 7.1　Core conclusions of Snell et al. 2024 (arXiv 2408.03314)

Question: under fixed inference FLOPs, how to allocate optimally?

Experimental setup:
- Same base model (PaLM-2 family)
- Different test-time strategies: BoN, majority vote, ToT-like beam search, PRM beam search, sequential revision (let model see its own answer and revise)
- **Compute-optimal**: dynamically select strategy per prompt based on difficulty (easy → greedy; hard → beam search + PRM)

Core finding:
- **Under fixed budget, optimal test-time scaling > 14× model scaling** (on certain MATH subsets)
- **Easy problems**: majority vote / BoN 4-8 is enough
- **Hard problems**: PRM-guided sequential revision + beam search gives the biggest gains

> ✅ **Compute-optimal scaling formula** —

$$\text{Compute}_\text{optimal}(x) = \arg\min_{(\theta, N)} \mathbb{E}[L(\pi_\theta(x; N))] \;\text{s.t.}\; \text{FLOPs}(\theta, N) \leq B$$
- $\theta$ = model size, $N$ = test-time samples / beam width
- Practical heuristic from the paper: on problems where base model pass@1 > 30%, adding N is more cost-effective than adding parameters

### 7.2　s1: Simple Test-Time Scaling (Muennighoff et al. Feb 2025, arXiv 2501.19393)

The simplest reasoning model: **1000-sample SFT + budget forcing inference control**.

**Data** (s1K, 1000 samples):
- Filtered by three criteria: difficulty, diversity, quality
- Each sample: question + reasoning trace (generated by Gemini Thinking) + answer

**Training**: 26 minutes of SFT on Qwen2.5-32B-Instruct (16×H100, 1 epoch)—possibly the cheapest reasoning model in history.

**Inference trick: budget forcing**

```
Model generates:
<think>
[reasoning tokens...]
[model attempts to output </think>, but current token count < target_budget]
[forcibly replace </think> with "Wait"]
[model continues reasoning...]
...
[when token count >= target_budget or naturally ends]
</think>
Answer: ...
```

Result: on AIME 2024, s1-32B (with budget forcing) = 56.7%, **exceeds o1-preview's 44.6%**.

> 💡 **Why does such a simple "Wait" trick work?**

- SFT taught the base model the `<think>...</think>` format, but reasoning length has a distribution (short problems → short reasoning)
- Forcibly injecting "Wait" makes the model stop at `</think>` boundary, activating its in-context "reflection" ability
- Equivalent to forcing the model to stay in "thinking mode" for more reasoning path samples
- Failure case: when base model has never seen reflection patterns, content after "Wait" is garbage—s1's 1K SFT data already implicitly contains reflection patterns

### 7.3　Sequential vs Parallel test-time compute

| Dimension | Parallel (BoN, Self-Consistency) | Sequential (long CoT, s1 budget forcing, o1) |
| --- | --- | --- |
| Implementation | Sample N times, verifier/vote selects | Single trace stretched out, model self-reflects |
| Latency | $N$× latency (can be parallelized) | Single but very long (cannot be parallelized away) |
| Memory | KV cache × N or sequential reuse | Single KV cache but long sequence |
| Plateau | Early saturation (N=8-16) | Continued scaling (10K-100K tokens) |
| Best for | Shallow reasoning (GSM8K, commonsense) | Deep reasoning (AIME, Codeforces) |

Snell 2024 reports: **parallel is more cost-effective for easy problems; sequential significantly better for hard problems**.

## §8 MCTS for Reasoning: rStar Series

### 8.1　PUCT formula (AlphaGo / AlphaZero origin)

At node $s$, for each action $a$, PUCT score:

$$\boxed{\;U(s, a) = Q(s, a) + c_\text{puct} \cdot \pi(a \mid s) \cdot \frac{\sqrt{N(s)}}{1 + N(s, a)}\;}$$

- $Q(s, a)$ = current $(s, a)$'s mean value (exploitation)
- $\pi(a \mid s)$ = policy prior (from policy network)
- $N(s)$ = total visit count for node $s$
- $N(s, a)$ = how many times action $a$ was selected
- $c_\text{puct}$ = exploration constant (typical 1.0-2.0)

Intuition: actions with fewer visits + favored by policy + currently high value are preferred.

### 8.2　rStar (Microsoft Research, arXiv 2408.06195) key idea

Treat LLM as policy + value plugged into MCTS:
- **State**: current partial reasoning $s_{<t}$
- **Action**: next-step action (rStar defines 5 reasoning actions: propose one-step, propose sub-question, generate full CoT, decompose, rephrase)
- **Reward**: at terminal node, use mutual consistency check (verified by another LLM)

**rStar-Math (arXiv 2501.04519)** goes further: use **MCTS rollouts to auto-label process labels** (similar idea to Math-Shepherd), train a process preference model (PPM), then policy + PPM self-evolve for four rounds. Qwen2.5-Math-7B on MATH: 58.8 → 90.0, approaching o1-preview.

### 8.3　Simplified MCTS for reasoning pseudocode

```python
import math
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class MCTSNode:
    state: str                          # partial reasoning trace
    parent: Optional["MCTSNode"] = None
    prior: float = 0.0                  # policy network prob
    visit_count: int = 0
    value_sum: float = 0.0
    children: dict[str, "MCTSNode"] = field(default_factory=dict)
    is_terminal: bool = False

    @property
    def q_value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0

def puct_score(node: MCTSNode, parent_visits: int, c_puct: float = 1.5) -> float:
    """PUCT: Q + c * P * sqrt(N_parent) / (1 + N)"""
    return node.q_value + c_puct * node.prior * math.sqrt(parent_visits) / (1 + node.visit_count)

def select(root: MCTSNode) -> MCTSNode:
    """Traverse tree by PUCT until reaching a leaf."""
    node = root
    while node.children and not node.is_terminal:
        node = max(node.children.values(),
                   key=lambda c: puct_score(c, node.visit_count))
    return node

def expand(node: MCTSNode, policy_lm, k: int = 4):
    """Sample k next-step candidates from policy LM."""
    if node.is_terminal:
        return
    candidates = policy_lm.sample_next_steps(node.state, k=k)   # list of (step_text, prior)
    for step_text, prior in candidates:
        new_state = node.state + "\n" + step_text
        is_term = step_text.startswith("Final answer:")
        node.children[step_text] = MCTSNode(
            state=new_state, parent=node, prior=prior, is_terminal=is_term,
        )

def rollout(node: MCTSNode, policy_lm, verifier) -> float:
    """Simulate from node to terminal; return reward."""
    if node.is_terminal:
        return verifier(node.state)
    state = node.state
    while len(state) < 4096:
        step = policy_lm.sample_next_step(state, temperature=1.0)
        state += "\n" + step
        if step.startswith("Final answer:"):
            break
    return verifier(state)   # ORM or PRM final score

def backup(node: MCTSNode, reward: float):
    """Propagate reward up the path."""
    while node is not None:
        node.visit_count += 1
        node.value_sum += reward
        node = node.parent

def mcts_search(prompt: str, policy_lm, verifier, n_simulations: int = 100, k: int = 4) -> str:
    """Standard MCTS for reasoning."""
    root = MCTSNode(state=prompt)
    # First expand the root, ensuring root.children is non-empty
    expand(root, policy_lm, k=k)
    for _ in range(n_simulations):
        leaf = select(root)
        if leaf.visit_count > 0 and not leaf.is_terminal:
            expand(leaf, policy_lm, k=k)
            if leaf.children:
                leaf = max(leaf.children.values(),
                           key=lambda c: c.prior)        # first pick highest prior
        reward = rollout(leaf, policy_lm, verifier)
        backup(leaf, reward)
    if not root.children:
        return root.state                                 # safety guard
    # Descend along the highest visit_count path until terminal or leaf (unexpanded)
    # This returns a complete trace rather than just the first-level child; if the
    # strongest mid-level path didn't reach terminal, it returns the deepest expanded state on that path
    node = root
    while node.children and not node.is_terminal:
        node = max(node.children.values(), key=lambda c: c.visit_count)
    return node.state
```

> ⚠️ **The real cost of MCTS for LLM** — The above pseudocode assumes `policy_lm.sample_next_steps` is a cheap op; in reality, each expansion is **one full LLM forward pass** (generating dozens of tokens + scoring). 100 simulations × 4 children = 400 LLM calls, 100-400× slower than a single greedy decode. This is why deployment in practice uses long-CoT (o1/R1) more often than MCTS—internalizing search into the policy.

### 8.4　PRM-Guided Beam Search (lighter alternative)

Instead of full MCTS, keep only top-$b$ partial traces, scoring each step with PRM:

```python
import math

def prm_beam_search(
    policy_lm,
    prm,
    prompt: str,
    beam_width: int = 4,
    expansion: int = 4,
    max_steps: int = 16,
) -> str:
    """PRM-guided step-level beam search."""
    # (cumulative_log_score, partial_trace, is_done)
    beams = [(0.0, prompt, False)]
    for _ in range(max_steps):
        new_beams = []
        for score, trace, done in beams:
            if done:
                new_beams.append((score, trace, True))
                continue
            # Expand `expansion` candidate steps from current trace
            candidates = policy_lm.sample_next_steps(trace, k=expansion, temperature=0.8)
            for step, _ in candidates:
                new_trace = trace + "\n" + step
                step_prob = prm.score_step(new_trace)        # in [0, 1]
                # Accumulate log probability (numerically stable)
                new_score = score + math.log(step_prob + 1e-6)
                is_terminal = step.startswith("Final answer:")
                new_beams.append((new_score, new_trace, is_terminal))
        # Defensive edge case: if all beams expand empty candidates, avoid max(empty)
        if not new_beams:
            break
        # Pick top-b by cumulative log-score
        beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:beam_width]
        if all(b[2] for b in beams):
            break
    # Return the highest-scoring trace (fall back to initial prompt if beams empty)
    return max(beams, key=lambda x: x[0])[1] if beams else prompt
```

> 💡 **Beam search vs MCTS** — Beam search is **deterministic + breadth-bounded** (fixed b per level), no backtracking; MCTS is **stochastic + adaptive** (visit count determines depth), with backtracking. The former is 5-10× faster; the latter has higher quality on deeper search trees. Snell 2024 recommends: medium difficulty → PRM beam search; very hard → MCTS.

## §9 Full Landscape of Reasoning Models

### 9.1　Comparison table of mainstream reasoning models (as of 2026)

| Model | Vendor | Released | Training paradigm | Inference control | Openness |
| --- | --- | --- | --- | --- | --- |
| **o1-preview / o1** | OpenAI | 2024-09 | RL on hidden CoT (details closed) | reasoning_effort: low/med/high | closed |
| **o3** | OpenAI | 2024-12 / 2025 | o1 successor; ARC-AGI 75.7%/87.5% | reasoning budget tunable; high-compute 172× | closed |
| **DeepSeek-R1-Zero** | DeepSeek | 2025-01 | Pure RL (GRPO + rule reward) | natural termination | fully open (weights + paper) |
| **DeepSeek-R1** | DeepSeek | 2025-01 | 4 stages: SFT cold-start + RL + rejection SFT + RL | natural termination | fully open |
| **R1-Distill (1.5B-70B)** | DeepSeek | 2025-01 | SFT only on R1 reasoning data | natural termination | fully open |
| **Claude 3.7 Sonnet** | Anthropic | 2025-02 | hybrid: standard + extended thinking | budget tokens user-configurable (up to 128K) | closed, but thinking content visible |
| **Gemini 2.0 Flash Thinking** | Google DeepMind | 2024-12 | Inference-optimized (method undisclosed) | thinking explicitly shown | closed |
| **s1-32B** | Stanford/Allen AI | 2025-02 | 1K-sample SFT + "Wait" budget forcing | budget forcing controls thinking length | fully open (incl. s1K data) |
| **rStar-Math** | Microsoft | 2025-01 | MCTS + PPM self-evolution | explicit MCTS at inference | partly open |
| **DeepSeek-Prover-V2** | DeepSeek | 2025-04 | subgoal decomposition + Lean 4 + RL | Lean formal verification | fully open |

### 9.2　Selection decision tree (frequently asked in interviews)

```

Task domain?
├── Math/competition (AIME / IMO) ─→ R1 / R1-Distill-32B / o1 / s1-32B
├── Formal theorem proving (Lean / Coq) ─→ DeepSeek-Prover-V2
├── Code (Codeforces / SWE-bench) ─→ o1-pro / R1 / Claude 3.7 + thinking
├── General reasoning (agent / chain task) ─→ Claude 3.7 / Gemini 2 Thinking / R1
├── Small-model deployment (edge / mobile) ─→ R1-Distill-1.5B-7B
└── ARC-AGI / abstract reasoning ─→ o3 (high-compute tier)

Deployment budget?
├── High (thousands of CoT tokens / problem) ─→ o1, o3, R1, Claude 3.7-extended
├── Medium (1-2K CoT tokens) ─→ s1-32B, R1-Distill-32B
└── Low (greedy or BoN=8) ─→ R1-Distill-1.5B + best-of-N with verifier
```

### 9.3　Does CoT really reflect reasoning? (deep question)

Classic controversy: the echo of **Attention is not Explanation** (Jain & Wallace 2019) for CoT—does the reasoning shown reflect the model's internal computation?

Evidence:
- **Supports authentic**: Anthropic Sleeper Agents (Hubinger et al. 2024) shows CoT content influences downstream action (not pure post-hoc rationalization)
- **Supports non-authentic**: Turpin et al. 2023 "Language Models Don't Always Say What They Think" found that with biased exemplars, the model's CoT gives plausible but wrong reasoning (biased but CoT doesn't mention the bias)
- **R1-Zero's "aha moment"** is limited evidence: behavioral changes (longer + more reflection) do correlate with performance gains, but could still be surface pattern

**In interviews, always give balanced view**: CoT is useful + partially trustworthy; but can't be treated as 100% explanation.

## §10 25 Frequently-Asked Interview Questions (L1 must-know / L2 advanced / L3 top labs)

Ranked from the perspective of top lab interviewers simulated by gpt-5.5 xhigh.

### L1 Must-Know (any LLM position will ask)

<details>

<summary>Q1. What is Chain-of-Thought? When to use?</summary>

- Few-shot prompts demonstrating "step-by-step" reasoning examples

- On large models (>62B), reasoning emerges (GSM8K +30+%)

- On small models, CoT actually degrades

- Subsequent Kojima 2022 "Let's think step by step" found zero-shot CoT also works

Treat CoT as a "magic prompt"—it's just a trigger; the capability comes from the base model itself

</details>

<details>

<summary>Q2. How does Self-Consistency work? Why better than greedy?</summary>

- temperature > 0 sample N CoTs

- Extract each one's final answer, do normalize (strip units, simplify fractions)

- Majority vote selects the majority answer

- Intuition: correct answer is an "attractor"; multiple sampling paths converge there

- Wang 2022 reports GSM8K +17.9%

Don't do answer normalization; assume higher temperature is better (actually 0.5-0.7 is optimal)

</details>

<details>

<summary>Q3. Tree-of-Thought (ToT) vs CoT?</summary>

- CoT is autoregressive single-chain path

- ToT is explicit tree search: each node is a thought, sample k children, LLM self-scores

- ToT can backtrack and prune

- But ToT requires external search framework + multiple LLM calls, 5-50× slower than CoT

- Game of 24: CoT 4% → ToT 74% (GPT-4)

Only say ToT is "multiple CoTs"—its core is the explicit evaluator + backtrack

</details>

<details>

<summary>Q4. ORM vs PRM?</summary>

- ORM (Outcome RM): one reward per whole trace (based on final answer correctness)

- PRM (Process RM): one reward per reasoning step

- PRM has dense signal but expensive labeling (PRM800K 800K human-labeled steps)

- Math-Shepherd (Wang et al. 2023, arXiv 2312.08935) samples **Monte Carlo completion rollouts** from intermediate steps (not MCTS tree search), uses soft/hard estimation from "how many rollouts ended correct" to auto-label PRM step labels

- Lightman 2023: on best-of-1024, PRM 78% > ORM 72% > majority vote 70%

Treat PRM as "training reward"—it's mainly for inference-time verify, not necessarily for RL

</details>

<details>

<summary>Q5. How does Best-of-N work? When does it saturate?</summary>

- Same prompt, sample N traces

- Use verifier (ORM or PRM) to pick the highest-scoring one

- Easy problems: N=4-8 saturates

- Hard problems: can keep growing to N=64-128 before saturating

- Theoretical limit oracle pass@N = $1-(1-p_1)^N$, but far below this when verifier is imperfect

Assume larger N is always better—verifier error means BoN can degrade after some point ("verifier overfits to surface features")

</details>

<details>

<summary>Q6. What are o1 / R1 reasoning_tokens?</summary>

- Hidden chain-of-thought tokens generated by the model

- API returns only the token count (e.g., OpenAI o1 returns `reasoning_tokens` field), content is not exposed

- Users pay per reasoning_tokens (this is why o1 is expensive)

- R1 wraps reasoning in `<think>...</think>`, full text visible

- Reasoning compute ≈ reasoning_tokens × model FLOPs/token

Treat reasoning_tokens as "no-cost optimization"—it significantly affects latency and cost

</details>

<details>

<summary>Q7. Main differences between GRPO and PPO?</summary>

- PPO requires a critic (value network), estimating baseline for each token

- GRPO uses **group statistics to replace the critic**: same prompt, sample G traces, advantage = $(r_i - \mu)/\sigma$

- GRPO broadcasts trace-level advantage to all tokens in that trace

- Memory halved (no critic); on long-CoT, critic is hard to learn anyway, so GRPO is more stable

- Shared: PPO clipping, KL regularization to reference policy

Assume GRPO is a "small tweak"—its stability advantage on long-CoT is a qualitative change

</details>

<details>

<summary>Q8. What is R1-Zero's "aha moment"?</summary>

- DeepSeek-R1 paper Fig 3 reports: after a few K steps of pure RL training

- CoT length spontaneously grows (hundreds → thousands of tokens)

- Self-reflection patterns like "Wait, let me reconsider..." spontaneously appear

- Performance jumps (AIME pass@1 15% → 70%)

- Intuition: rule-based reward + GRPO makes "think longer + self-verify" a high-reward strategy

Treat "aha moment" as mystical—it's a predictable emergence from reward shaping + long-episode RL

</details>

<details>

<summary>Q9. Why does R1-Distill use SFT not RL?</summary>

- R1's reasoning ability emerges from big base + strong RL

- Direct RL on small model hardly causes emergence (base too weak, rollouts almost all wrong, reward signal too sparse)

- Use 600K reasoning traces generated by R1 for SFT, equivalent to demonstration learning

- Paper reports: 32B SFT-distill 72.6 on AIME vs direct Qwen-32B RL's 47.0

- Implication: **distillation of reasoning > direct RL on small models** (under current algorithms)

Assume RL is always better than SFT—prerequisite is that base is strong enough

</details>

<details>

<summary>Q10. What does Snell 2024's "test-time compute > parameter scaling" mean?</summary>

- Under fixed inference compute budget, let 1B model sample more + verify

- On certain MATH subsets, can exceed greedy performance of a 14× larger model

- Prerequisite: base model has non-trivial pass@1 (>30%) on the task

- Not universal: on completely-unsolvable tasks, no amount of test-time compute saves you

- Industry deployment often uses R1-Distill + BoN=8 + PRM as substitute for direct R1

Treat it as "end of scaling laws"—it's just "scaling law along another dimension", doesn't replace training scaling

</details>

### L2 Advanced (reasoning direction / research roles)

<details>

<summary>Q11. Derive GRPO advantage formula by hand, and handle the std=0 case</summary>

- For each prompt, sample G traces, get rewards $\{r_1, \dots, r_G\}$

- $\mu = \frac{1}{G}\sum r_i$, $\sigma = \sqrt{\frac{1}{G}\sum(r_i - \mu)^2}$ (biased std)

- $A_i = (r_i - \mu)/(\sigma + \epsilon)$

- When all $G$ rewards are identical (all correct or all wrong) → $r_i - \mu = 0$ and $\sigma = 0$, so
  - With $\epsilon$: advantage $= 0/\epsilon = 0$ → **policy-gradient term zeros out**; but KL regularization term still exists, **total loss may still have KL update** (pulling policy back to reference)
  - Without $\epsilon$: $0/0 = $ NaN
  - Note: not $\pm\infty$ (numerator is also 0)

- In practice: either skip the prompt (GRPO_loss=0), or clamp $\sigma$ to a floor (e.g., 0.1)

- This case indicates the prompt is **too easy or too hard**; data filtering should exclude it

Only write the formula without mentioning std=0 boundary

</details>

<details>

<summary>Q12. What is each stage's goal in R1's 4-stage pipeline?</summary>

- **Stage 1 Cold-start SFT**: have base learn human-readable reasoning format (not for reasoning ability itself)

- **Stage 2 Reasoning RL**: GRPO + rule reward to improve math/code reasoning

- **Stage 3 Rejection sampling + SFT**: extend to non-verifiable domains + preserve general ability

- **Stage 4 All-scenario RL**: safety + helpfulness closing (similar to RLHF)

- Key: reasoning comes from Stage 2 RL; Stages 1 + 3 are readability/generalization injection; Stage 4 is alignment

Treat 4 stages as "cooking steps"—actually each stage has orthogonal function

</details>

<details>

<summary>Q13. Why is R1's reward rule-based rather than learned RM?</summary>

- Math/code can be **programmatically verified** (answer regex matching, unit tests)

- Learned RM is easy to hack (reward model overoptimization → policy finds tricks rather than truly solving)

- Rule reward provides signal close to ground-truth, **avoiding the main failure mode of learned-RM drifting off training distribution** (can still be hacked via regex loopholes, format tricks, test-set pollution, but far easier to plug than RM hacking)

- Cost: only works for verifiable tasks (math/code/format), not for open-ended tasks

- R1 Stage 4 adds RM for helpfulness/harmlessness—rule for verifiable tasks, RM for open tasks

Treat rule-based as "simple"—the key is it's hard to be RM-hacked, not simple

</details>

<details>

<summary>Q14. Why does budget forcing ("Wait" trick) work?</summary>

- s1 does SFT on 1K reasoning traces; model learns `<think>...</think>` format + reflection patterns

- At inference, if model attempts to output `</think>` but token count hasn't reached target budget

- Forcibly replace with "Wait"; model naturally continues with reflection tokens

- Equivalent to forcing model to stay in thinking mode for more internal reasoning path samples

- Failure mode: if base has never seen reflection patterns → after "Wait" comes garbage (s1's 1K SFT is a necessary prerequisite)

Assume "Wait" is a prompting trick—it relies on reflection patterns injected by SFT

</details>

<details>

<summary>Q15. How is PRM used in best-of-N? How to choose aggregation?</summary>

- For N traces, use PRM to score each step: $p_1, \dots, p_T$

- Three aggregations: **min**, **prod (log-sum)**, **mean**

- Lightman 2023: **min ≈ prod >> mean** (mean lets high-scoring steps mask weak ones)

- Min intuition: trace strength is determined by weakest link

- Code detail: prod uses log-sum to avoid numerical underflow

Only using mean—common mistake

</details>

<details>

<summary>Q16. What is the PUCT formula? How to tune c_puct?</summary>

- $U(s, a) = Q(s, a) + c_\text{puct} \cdot \pi(a \mid s) \cdot \sqrt{N(s)} / (1 + N(s, a))$

- $Q$ = exploit; $c_\text{puct} \cdot \pi \cdot \sqrt{N}/(1+N(s,a))$ = explore

- Large $c_\text{puct}$: biased toward exploration (policy prior and unvisited actions matter more)

- Small $c_\text{puct}$: biased toward exploitation (already-discovered high-Q actions dominate)

- AlphaZero uses $c_\text{puct} \approx 1.0$; MCTS-for-LLM usually 1.5-2.0 (because LLM policy prior is more accurate than Go)

- Note: numerator is $\sqrt{N(s)}$ (parent visits); denominator is $1 + N(s, a)$ (child visits)

Misremember $\sqrt{N}$ as child visits (wrong)

</details>

<details>

<summary>Q17. How does Math-Shepherd auto-label step labels? Key assumption?</summary>

- For each step $s_t$ in a trace, rollout K completions from $s_t$

- Count how many final answers are correct; get estimated step quality $\hat{q}_t = (\text{correct count}) / K$

- Use $\hat{q}_t$ as BCE/MSE label to train PRM

- **Key assumption**: base model has non-trivial success rate on the task (otherwise all rollouts wrong, $\hat{q}_t$ all 0)

- Bootstrap problem: weak base → can't use MCTS-label; strong base → no need for PRM

- In practice: use medium-strength base (Mistral-7B post-SFT) as rollout source

Assume MCTS-label is free—it requires base to already have partial capability

</details>

<details>

<summary>Q18. Does CoT really reflect reasoning? How to verify?</summary>

- Partially authentic, partially post-hoc rationalization (consensus view)

- Turpin 2023: with biased exemplars, model CoT gives plausible but wrong explanations (without mentioning bias)

- Anthropic Sleeper Agents (Hubinger 2024): CoT content influences downstream action, not pure post-hoc

- Verification methods: causal intervention (change CoT and see if output changes), faithfulness benchmark

- Interview talking point: keep balanced view, don't go to extremes

Only say "CoT is explanation" or "CoT is all post-hoc"—both extremes are wrong

</details>

<details>

<summary>Q19. How to decide which reasoning model fits your task?</summary>

- Verifiable task (math/code): rule-based RL route (R1 / R1-Distill)

- Open-ended task (writing/dialogue): hybrid RM route (Claude 3.7-extended / o1)

- Abstract reasoning (ARC-AGI): o3 high-compute tier (other models still very weak on ARC-AGI)

- Formal proof: DeepSeek-Prover-V2 (Lean 4 integration)

- Tight deployment budget: R1-Distill-7B/14B + best-of-N + PRM verifier (10×+ cheaper than direct R1)

Recommend o1 without looking at task—mismatch is expensive and underperforms

</details>

<details>

<summary>Q20. What is the role of KL regularization in GRPO? How to tune β?</summary>

- $\mathcal{L}_\text{total} = \mathcal{L}_\text{PG} + \beta \cdot \mathrm{KL}(\pi_\theta \| \pi_\text{ref})$

- $\pi_\text{ref}$ = policy at training start (usually SFT or base)

- Large $\beta$: policy doesn't drift from reference, but can't learn new capabilities

- Small $\beta$: policy freely explores, but may collapse (generating garbage)

- DeepSeek-R1 uses $\beta = 0.001$ (very small, encourages exploration); standard RLHF uses 0.01-0.1

- For long-CoT, KL accumulates token-level; total is large, so $\beta$ must be much smaller than for short CoT

Copy-paste RLHF's $\beta$ to long-CoT—will over-suppress exploration

</details>

### L3 Advanced (top labs / research direction)

<details>

<summary>Q21. What's the root cause of GRPO being more sample efficient than PPO? (beyond "saving the critic")</summary>

- **Trace-level reward perfectly aligns with trace-level credit**: when reward only comes from final answer, PPO's critic-based per-token credit actually introduces noise; GRPO directly broadcasts trace-level advantage—the advantage estimate itself has limited bias introduced by the group baseline (both mean/std are biased sample statistics), but it's more stable than the critic's high-variance estimate on long episodes, and has lower variance for trace-level rewards

- **Within-prompt contrast eliminates prompt-level baseline noise**: advantage = $(r-\mu)/\sigma$ is equivalent to paired comparison, more accurate than critic's globally-estimated baseline

- **Critic is hard to learn on long-CoT**: reward extremely sparse (episodes 4K-32K tokens), $V_\psi$ on intermediate tokens is nearly random; GRPO skips this learning problem

- **Rule-based reward is hard to be RM-hacked**: r1 uses rule reward; no learned RM to over-optimize; policy optimization direction is close to ground-truth (can still be hacked via regex/format loopholes, but avoids RM-distribution-shift as the main path)

- **Group size G controls variance**: variance ∝ $1/G$; $G=16$ gives sufficiently low variance without exploding memory

- Conclusion: GRPO isn't a "small tweak"; it's the algorithmically right answer under the long-CoT + rule reward setting

Only say "save the critic"—surface-level reason

</details>

<details>

<summary>Q22. R1-Zero's pure RL emergence vs historical PPO RLHF (InstructGPT)—what's the difference? Why did the former break through?</summary>

- **Reward source**: R1-Zero uses rule-based; InstructGPT uses learned RM (preference model)

- **Reward density and sparsity**: rule reward on long-CoT is response/trace-level sparse but hard to hack and signal close to ground-truth; InstructGPT's learned RM also outputs **response-level scalar preference reward** (not per-token scoring); token-level advantage is jointly constructed by critic + GAE + KL penalty; RM in RLHF still faces reward overoptimization (policy drifts off RM's training distribution → gets unrealistic high scores)

- **Algorithm**: R1-Zero uses GRPO; InstructGPT uses PPO + critic + RM

- **Reward scope**: R1-Zero trains reasoning (verifiable); InstructGPT trains alignment (open-ended)—former has oracle reward, latter doesn't

- **Base model**: R1-Zero uses V3-Base (671B MoE), already strong pretrained reasoning prior; InstructGPT is GPT-3 (175B dense)

- Historical reason: 2022-2023 PPO+RM paradigm was held back by RM overopt + hard-to-learn critic; rule reward only worked through on math/code

- Meaning: **reasoning RL breakthrough = rule reward + GRPO + strong base + long-CoT joint effect**, not a single technical win

Only say "DeepSeek used GRPO"—miss the whole paradigm shift

</details>

<details>

<summary>Q23. Why can such a simple "Wait" trick surpass o1-preview? What does this tell us?</summary>

- s1 core: 1K curated traces SFT (Qwen2.5-32B) + "Wait" budget forcing

- First-level explanation: 1K SFT made the model learn the "shape" of `<think>` format + reflection patterns

- Second-level: the model actually **already "knows how to think"** (saw lots of human reasoning during pretrain); SFT just activates + formats

- Third-level: "Wait" forces the model to stop at thinking boundary, re-sample—equivalent to forcibly doing in-context self-revision

- Inference: **reasoning capability's core is activation, not injection**—base model already has lots of reasoning prior

- Implication for research:
  - Don't assume reasoning must come from large-scale RL
  - SFT data quality > quantity (s1K 1000 samples > lots of low-quality data)
  - Inference control (budget forcing) is a vastly underexplored dimension

- Reflection: s1 doesn't refute R1 route—R1-Distill is also a kind of distill SFT; s1 is the extreme version of this thinking

Only say "s1 is simple and impressive"—miss the reasoning = activation observation

</details>

<details>

<summary>Q24. Compare the essential differences between sequential test-time compute (long CoT) and parallel test-time compute (BoN / MCTS). When to choose which?</summary>

- **Sequential (o1, R1, s1)**: single trace stretched, model self-reflects + self-verifies
  - Advantages: single KV cache (memory friendly); information passes continuously within trace (later steps see all earlier reasoning)
  - Disadvantages: early errors propagate to the end (no backtrack); hard tasks need very long traces (10K-100K tokens)

- **Parallel (BoN, ToT, MCTS)**: multiple independent traces, external aggregator/verifier selects
  - Advantages: parallelizable → low latency; each trace independent, errors don't propagate
  - Disadvantages: no information exchange between traces; verifier must be accurate or aggregation fails

- **Selection**:
  - Strong sequential dependency in task (math competition, theorem proving) → long CoT (error info can be corrected by later reflection)
  - Multi-solution task (codeforce, creative writing) → BoN (multi-path coverage)
  - Task has well-defined intermediate verifier (math step) → MCTS / PRM beam search
  - Latency-sensitive → parallel (can parallelize on GPU)
  - Single-GPU memory sensitive → sequential (single KV cache)

- **Future direction**: sequential + parallel hybrid—embed multi-path exploration inside a single long CoT (e.g., o1 may be doing this internally, but closed-source unknowable)

Only say "sequential beats parallel"—depends on task

</details>

<details>

<summary>Q25. If you were to design the next-generation reasoning model, which directions would you pursue? (open-ended top-lab interview)</summary>

Credible response framework (no need to cover everything; pick 2-3 to dive into):

- **Direction 1 - Training algorithms**:
  - GRPO is currently trace-level; how to do token-level without introducing a critic? (e.g., learned PRM-as-critic)
  - Reward shaping: rule reward is too sparse; can it be densified while staying hard-to-hack (e.g., formal verification of intermediate steps)?
  - Continual RL: R1 is frozen after training; can we do online RL during deployment?

- **Direction 2 - Test-time compute scaling**:
  - Adaptive budget: dynamically allocate reasoning tokens by problem difficulty (Snell 2024 is the start)
  - Sequential + parallel hybrid: embed sub-tree exploration within long CoT
  - Multi-agent debate: multiple LLMs verifying each other, adversarial

- **Direction 3 - Verifier**:
  - Generative PRM replacing scalar PRM (LLM evaluating step quality is more accurate than scalar head)
  - Self-verifier: let model verify itself (DeepSeek-Prover-V2 on Lean is an embryonic version)
  - Cross-domain transfer: can math PRM transfer to code PRM?

- **Direction 4 - Evaluation**:
  - Current reasoning benchmarks (AIME, MATH) near saturation—next-generation evaluation standard?
  - Robustness: is the reasoning model brittle on adversarial prompts?

- **Direction 5 - Inference interpretability**:
  - CoT faithfulness (cf. Q18): make CoT truly reflect internal computation
  - Mechanistic interpretability: can we localize specific attention heads responsible for "reflection"?

- **Direction 6 - Reasoning + agent**:
  - Reasoning is mostly single-turn currently; how does reasoning persist across turns in agentic settings?
  - How to jointly optimize tool use + reasoning?

Copy-paste existing methods + add a bit—doesn't show research taste

</details>

## §A Appendix: Core paper timeline + one-line summaries

Reverse chronological:

| Date | Paper | arXiv | One-line contribution |
| --- | --- | --- | --- |
| 2025-04 | DeepSeek-Prover-V2 | 2504.21801 | subgoal decomposition + Lean 4 RL, MiniF2F 88.9% |
| 2025-02 | Claude 3.7 Sonnet | (no arXiv) | hybrid model, extended thinking budget user-controllable |
| 2025-02 | s1: Simple Test-Time Scaling | 2501.19393 | 1K SFT + "Wait" budget forcing surpasses o1-preview |
| 2025-01 | DeepSeek-R1 / R1-Zero | 2501.12948 | pure RL (GRPO + rule reward) emerges reasoning; R1 = 4-stage pipeline |
| 2025-01 | rStar-Math | 2501.04519 | MCTS + PPM self-evolution, 7B approaches o1-preview |
| 2024-12 | o3 (OpenAI) | (no arXiv) | ARC-AGI 75.7%-87.5%, first abstract reasoning approaching human |
| 2024-12 | Gemini 2.0 Flash Thinking | (no arXiv) | Google's first reasoning model, thinking explicitly visible |
| 2024-09 | o1 (OpenAI) | (no arXiv) | first commercial reasoning model, hidden CoT + RL |
| 2024-08 | Snell et al. Test-Time Compute | 2408.03314 | optimized test-time compute > 14× model scaling |
| 2024-08 | rStar | 2408.06195 | MCTS + mutual reasoning, large improvement for small LMs |
| 2024-02 | DeepSeekMath / GRPO | 2402.03300 | GRPO algorithm first proposed, removes critic |
| 2023-12 | Math-Shepherd | 2312.08935 | MC completion rollouts auto-label PRM |
| 2023-05 | Tree of Thoughts | 2305.10601 | explicit tree search + LLM evaluator |
| 2023-05 | Let's Verify Step by Step | 2305.20050 | PRM > ORM > majority vote; PRM800K dataset |
| 2022-03 | Self-Consistency | 2203.11171 | sample N + majority vote, GSM8K +17.9% |
| 2022-05 | Zero-shot CoT (Kojima) | 2205.11916 | "Let's think step by step", zero-shot triggers CoT |
| 2022-01 | Chain-of-Thought (Wei) | 2201.11903 | few-shot step-by-step demonstration, CoT emerges on large models |

> 💡 **Recommend reading 4 papers carefully** — when interview prep time is limited, read in this priority:

1. DeepSeek-R1 (2501.12948) —— covers GRPO + 4 stages + R1-Zero "aha moment"
2. DeepSeekMath (2402.03300) —— original GRPO algorithm paper
3. Let's Verify Step by Step (2305.20050) —— PRM foundations
4. Snell et al. (2408.03314) —— test-time compute scaling paradigm

After reading these 4 + this cheat sheet, reasoning model interview questions should cover 80%+.

> ⚠️ **Open-ended question prep** — top-lab interviews often ask open-ended questions (like Q25); the key is showing **research taste**: list 3-5 concrete directions (not "I'll work on reasoning models" empty talk); each direction with a concrete proposal + one expected failure mode. Don't memorize when prepping; instead read recent 6 months of arXiv reasoning papers and build your own taxonomy.
