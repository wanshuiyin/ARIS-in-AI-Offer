## §0 TL;DR Cheat Sheet

> 💡 **Post-training alignment in 7 sentences** — one page covering the interview essentials (see §2–§9 for derivations).

1. **RLHF pipeline (Ouyang 2022 InstructGPT)**: SFT → RM (Bradley-Terry pairwise) → PPO + per-token KL; the value model (value head) is trained separately, and the policy is anchored to the reference policy by KL.

2. **PPO core (Schulman 2017)**: clipped surrogate $L^{\text{CLIP}}(\theta) = \mathbb{E}[\min(r_t A_t,\; \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t)]$, importance ratio $r_t = \pi_\theta / \pi_{\theta_\text{old}}$; advantages use **GAE** $A_t^{\text{GAE}} = \sum_{l \ge 0} (\gamma\lambda)^l \delta_{t+l}$ to balance bias/variance.

3. **DPO closed-form (Rafailov 2023 NeurIPS)**: the optimal policy of KL-regularized RLHF is $\pi^*(y|x) \propto \pi_\text{ref}(y|x)\exp(r(x,y)/\beta)$; inverting gives $r = \beta\log(\pi^*/\pi_\text{ref}) + \beta\log Z(x)$; substituting into Bradley-Terry, **$\log Z$ cancels in the pairwise difference**, leaving a pure SFT-style loss $-\log\sigma(\beta\log\frac{\pi(y_w)}{\pi_\text{ref}(y_w)} - \beta\log\frac{\pi(y_l)}{\pi_\text{ref}(y_l)})$.

4. **GRPO (DeepSeekMath 2024, R1 2025)**: for each prompt sample a group of $G$ responses, advantage uses **within-group normalization** $\hat{A}_i = (r_i - \text{mean}(\mathbf{r}))/\text{std}(\mathbf{r})$; **drops the value model**, saving half the memory, especially suitable for LLM math/code RL.

5. **DPO variant ecosystem**: KTO (only needs thumbs up/down, no pairs), IPO ($\ell_2$ form preventing reward overfit), SimPO (reference-free, length-normalized), ORPO (one-stage SFT + odds-ratio fusion), RLOO (leave-one-out baseline over multiple samples, value-free), ReMax (greedy baseline, further savings).

6. **PRM vs ORM (Lightman 2023 arXiv / 2024 ICLR)**: process-reward supervises each reasoning step, Math-Shepherd (Wang 2024 ACL) auto-labels PRM; outcome-reward only looks at the final answer. **PRM consistently beats ORM on math reasoning**, but annotation cost is high.

7. **Reward hacking is the central pain point**: the model over-optimizes the proxy reward, producing answers that get longer, sycophantic, with abnormal style; mitigations = KL penalty, reward clipping, length penalty, ensemble RM, Constitutional AI / RLAIF (Bai 2022, Lee 2023).

## §1 Post-Training Alignment Intuition

LLM training splits into three stages:

- **Pretraining**: next-token prediction over trillion-token corpora — learns world knowledge and language patterns
- **SFT (Supervised Fine-Tuning)**: next-token on instruction-response pairs — learns instruction format and basic capability
- **Alignment / RL post-training**: make outputs **align with human preferences** (helpful / harmless / honest) — learns "which response is better"

Why isn't SFT enough? Because SFT can only mimic positive examples ("what a good response looks like") and cannot explicitly learn **contrastive signal** ("A is better than B"). RL post-training provides three paradigms:

| Paradigm | Signal | Representative | One-liner |
| --- | --- | --- | --- |
| **RLHF + PPO** | Learn an RM to mimic preferences, then optimize the RM via RL | InstructGPT, ChatGPT, Claude | RM-in-the-loop on-policy RL |
| **DPO family** | Skip the RM, do contrastive loss directly on preference data | DPO, IPO, SimPO, KTO, ORPO | Offline, no sampling needed |
| **GRPO family** | RM scores online but **skip the value model**, advantage via within-group normalization | DeepSeek-R1, Kimi-K1.5 | Top choice for math/code tasks |

> 💡 **Why not pure reward?** — If you directly maximize reward, the model finds shortcuts in the RM (reward hacking). The KL penalty $\beta \cdot \text{KL}(\pi || \pi_\text{ref})$ is the core "anti-drift" mechanism: it keeps the RL-trained policy from drifting too far from the SFT base, acting as an implicit regularizer.

### 1.1　Special characteristics of language-task RL

Classic game RL (Atari, Go) and LLM RL differ significantly, a common interview probe:

| Dimension | Game RL (classic PPO setting) | LLM RL |
| --- | --- | --- |
| **State space** | Image / board | Token sequence, up to $\sim 10^4$ long |
| **Action space** | Tens to hundreds of discrete actions | Large vocab ($\sim 10^5$) |
| **Trajectory length** | Thousands of steps | Typically 1 response (one reward at end of generation) |
| **Reward sparsity** | Intermediate rewards exist | Usually **only terminal reward** (end of response) |
| **Environment** | Independent simulator | RM (also a neural network, **can be hacked**) |
| **on/off-policy** | on-policy (PPO) | RLHF: on-policy; DPO: fully offline |

Because there is only a terminal reward, **LLM RL advantage estimation often uses coarse schemes**: PPO+GAE distributes advantage over each token (but reward is 0 on most tokens); GRPO directly uses the whole response's reward, assigning all tokens the same within-group normalized advantage.

## §2 PPO Core

### 2.1　Vanilla policy gradient recap

Policy gradient theorem:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\!\left[\sum_t \nabla_\theta \log \pi_\theta(a_t | s_t)\, A^{\pi_\theta}(s_t, a_t)\right]$$

REINFORCE uses $A \leftarrow R_t$ (return), which has high variance; Actor-Critic uses $A^{\pi}(s, a) = Q^\pi(s,a) - V^\pi(s)$ to reduce variance.

### 2.2　PPO clipped surrogate (must-know formula)

Define importance ratio:

$$r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_\text{old}}(a_t | s_t)}$$

PPO-Clip objective:

$$\boxed{\;L^{\text{CLIP}}(\theta) = \mathbb{E}_t\!\left[\min\!\Big(r_t(\theta) A_t,\; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t\Big)\right]\;}$$

Intuition:

- If $A_t > 0$ (this action beats baseline), we want to increase $\pi_\theta(a_t|s_t)$, but **only up to $r_t = 1+\epsilon$** (preventing too aggressive a single update).
- If $A_t < 0$ (this action is worse than baseline), we want to decrease $\pi_\theta(a_t|s_t)$, but **only down to $r_t = 1-\epsilon$**.
- `min` picks the smaller of the two → **pessimistic estimate** (pessimistic bound): when we want to "add points," clip the upper bound; when we want to "subtract points," clip the lower bound.

Typical $\epsilon = 0.1 \sim 0.2$. In LLM RLHF practice, $0.1 \sim 0.2$ is common; too large easily blows up the KL.

> ⚠️ **PPO-Clip vs PPO-Penalty** — The original paper also has a PPO-Penalty form: $L = \mathbb{E}[r_t A_t] - \beta\, \text{KL}(\pi_{\theta_\text{old}} \| \pi_\theta)$, with $\beta$ adaptively tuned. In production, the **clipped form is more common** (rl4lms / TRL / OpenRLHF defaults).

### 2.3　GAE: generalized advantage estimation

Define TD residual:

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

GAE is the **exponentially weighted average** of advantages over different step lengths:

$$\boxed{\;A_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}\;}$$

Boundary cases:

- $\lambda = 0$ → $A_t = \delta_t$, pure TD(0), high bias low variance
- $\lambda = 1$ → $A_t = \sum_l \gamma^l r_{t+l} - V(s_t)$, pure Monte Carlo (advantage equals actual return minus baseline), low bias high variance
- Typical $\lambda = 0.95$, $\gamma = 0.99$

**Degeneration of GAE in LLMs**: in RLHF typically $\gamma = 1$ (no discount), and there is only a terminal reward, so $\delta_t = -V(s_t) + V(s_{t+1})$ for intermediate tokens, $\delta_T = R_T - V(s_T)$ for the terminal token. In this case GAE is equivalent to "value baseline + reward backflow."

### 2.4　Full PPO objective in RLHF

In LLMs, each timestep $t$ corresponds to generating the $t$-th token, state = $(x, y_{\lt t})$, action = $y_t$. The reward usually adds a KL penalty:

$$\tilde{r}_t = \mathbb{1}[t = T] \cdot R(x, y) - \beta \log \frac{\pi_\theta(y_t | x, y_{\lt t})}{\pi_\text{ref}(y_t | x, y_{\lt t})}$$

Final objective:

$$L^{\text{PPO}}(\theta) = L^{\text{CLIP}}(\theta) - c_v \cdot \underbrace{\mathbb{E}_t (V_\phi(s_t) - V_t^\text{target})^2}_{\text{value loss}} + c_e \cdot \underbrace{\mathbb{E}_t \mathcal{H}[\pi_\theta(\cdot | s_t)]}_{\text{entropy bonus}}$$

Typical $c_v = 0.5$, $c_e = 0.01$ (in LLMs the entropy bonus is usually very small or 0, because the vocab is large and entropy is already high).

> ⚠️ **Code block convention** — The PPO / RM / DPO / GRPO blocks below are **teaching pseudocode**, each independently readable (each imports `torch / torch.nn.functional as F`). Production implementation needs additional: (1) HF transformer forward passing `attention_mask`; (2) decode-time `position_ids` padding handling; (3) clipping `targets` to vocab before `gather`; (4) RM finding the true end via `attention_mask` for last-token. This article focuses on the core loss derivations.

### 2.5　Code (core 60 lines)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_gae(rewards, values, dones, gamma=1.0, lam=0.95):
    """
    rewards: [B, T]    per-token reward (with KL penalty)
    values:  [B, T+1]  value at s_0...s_T (s_T = terminal, V=0)
    dones:   [B, T]    1 if terminal, 0 otherwise
    returns advantages [B, T], returns [B, T]
    """
    B, T = rewards.shape
    advantages = torch.zeros_like(rewards)
    gae = 0.0
    for t in reversed(range(T)):
        non_term = 1.0 - dones[:, t]
        delta = rewards[:, t] + gamma * values[:, t + 1] * non_term - values[:, t]
        gae = delta + gamma * lam * non_term * gae
        advantages[:, t] = gae
    returns = advantages + values[:, :T]
    return advantages, returns


def ppo_step(policy, value, batch, eps_clip=0.2, c_v=0.5, c_e=0.01):
    """
    batch: dict with keys
      input_ids:    [B, L]     prompt + response tokens
      action_mask:  [B, L]     1 for response tokens, 0 for prompt/pad
      old_log_probs:[B, L]     log π_θold(y_t | s_t) at sample time
      advantages:   [B, L]     GAE advantages (already normalized)
      returns:      [B, L]     GAE returns (V target)
    """
    logits = policy(batch["input_ids"]).logits          # [B, L, V]
    # log-prob of taken action y_t at position t (shifted by 1: predict t+1 from t)
    log_probs = F.log_softmax(logits[:, :-1], dim=-1)
    targets   = batch["input_ids"][:, 1:].unsqueeze(-1)
    new_log_probs = log_probs.gather(-1, targets).squeeze(-1)  # [B, L-1]
    new_log_probs = F.pad(new_log_probs, (1, 0))               # align to [B, L]

    mask = batch["action_mask"].float()
    ratio = torch.exp(new_log_probs - batch["old_log_probs"])  # r_t

    # ── PPO-Clip core ──
    A = batch["advantages"]
    surr1 = ratio * A
    surr2 = torch.clamp(ratio, 1.0 - eps_clip, 1.0 + eps_clip) * A
    policy_loss = -((torch.min(surr1, surr2) * mask).sum() / mask.sum())

    # ── Value loss ──
    V = value(batch["input_ids"]).squeeze(-1)                  # [B, L]
    value_loss = (((V - batch["returns"]) ** 2) * mask).sum() / mask.sum()

    # ── Entropy bonus ──
    probs = log_probs.exp()                                    # [B, L-1, V]
    entropy = -(probs * log_probs).sum(-1)                     # [B, L-1]
    entropy = F.pad(entropy, (1, 0))
    entropy_bonus = (entropy * mask).sum() / mask.sum()

    loss = policy_loss + c_v * value_loss - c_e * entropy_bonus

    # Monitoring (does not participate in backprop)
    with torch.no_grad():
        approx_kl = ((ratio - 1) - torch.log(ratio.clamp_min(1e-8)))
        approx_kl = (approx_kl * mask).sum() / mask.sum()
        clip_frac = (((ratio < 1 - eps_clip) | (ratio > 1 + eps_clip)).float() * mask).sum() / mask.sum()

    return loss, {"policy": policy_loss.item(), "value": value_loss.item(),
                  "entropy": entropy_bonus.item(),
                  "approx_kl": approx_kl.item(), "clip_frac": clip_frac.item()}
```

> ⚠️ **Top 5 PPO engineering pitfalls** —

- `approx_kl` uses $\mathbb{E}[r - 1 - \log r] \ge 0$ (Schulman 2020 blog), more stable than $\mathbb{E}[\log r]$ and never negative.
- Multiple PPO updates per epoch (typically 4), but monitor `approx_kl`: early stop if it exceeds `target_kl` (e.g. 0.02).
- Advantage must do batch-level normalization (subtract mean divide std), otherwise scale will make learning rate ineffective.
- When computing KL penalty with reference policy, KL is token-level (not sequence-level); writing wrong dimensions easily causes gradient explosion.
- `c_v=0.5` is just an empirical value; if value loss is far greater than policy loss, it "eats" the gradient. You can use a separate optimizer or a separate lr for value.

## §3 RLHF Pipeline (InstructGPT Paradigm)

Ouyang et al. 2022 NeurIPS, *Training language models to follow instructions with human feedback*, is the predecessor of ChatGPT and defines today's standard three-stage RLHF:

### 3.1　Stage 1 — SFT (Supervised Fine-Tuning)

Given instruction-response pairs $\{(x, y)\}$ (high-quality human-written), do next-token:

$$\mathcal{L}_\text{SFT}(\phi) = -\mathbb{E}_{(x, y) \sim \mathcal{D}_\text{SFT}}\left[\sum_t \log \pi_\phi(y_t | x, y_{\lt t})\right]$$

Output is $\pi_\text{SFT}$ (also called $\pi_\text{ref}$, since downstream RL uses it as KL anchor).

### 3.2　Stage 2 — RM (Reward Model)

For the same prompt $x$, sample multiple responses with $\pi_\text{SFT}$ and have humans **pairwise compare** to obtain preference pairs $(x, y_w, y_l)$, $y_w \succ y_l$.

**Bradley-Terry preference model**:

$$P(y_w \succ y_l | x) = \sigma(r^*(x, y_w) - r^*(x, y_l))$$

where $r^*$ is the unknown "true" reward function. We fit it with $r_\psi$ (a transformer with a scalar head on the last layer):

$$\boxed{\;\mathcal{L}_\text{RM}(\psi) = -\mathbb{E}_{(x, y_w, y_l)} \log \sigma\!\big(r_\psi(x, y_w) - r_\psi(x, y_l)\big)\;}$$

Implementation details:

- RM is usually initialized from SFT model (shared backbone), with the last token's hidden state going through a linear layer to scalar reward.
- After training, RM parameters are **frozen**.

```python
import torch
import torch.nn.functional as F

def rm_loss(reward_model, batch):
    """
    batch:
      chosen_ids:   [B, L]     prompt + y_w
      rejected_ids: [B, L]     prompt + y_l
    """
    r_w = reward_model(batch["chosen_ids"])      # [B]  scalar at last token
    r_l = reward_model(batch["rejected_ids"])    # [B]
    # Bradley-Terry NLL
    loss = -F.logsigmoid(r_w - r_l).mean()
    accuracy = (r_w > r_l).float().mean()
    return loss, accuracy
```

### 3.3　Stage 3 — PPO + KL (Policy Optimization)

Objective:

$$\boxed{\;\max_{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_\theta(\cdot|x)} \big[r_\psi(x, y)\big] - \beta\, \mathbb{E}_x\, \text{KL}\!\big(\pi_\theta(\cdot|x) \,\big\|\, \pi_\text{ref}(\cdot|x)\big)\;}$$

In implementation, split the KL across each token and merge it with the RM reward into a per-token reward $\tilde{r}_t$ (see §2.4), then run PPO.

```

prompt x
   │
   │   π_θ generates response y
   ↓
(x, y)
   │
   │   r_ψ(x, y) → scalar reward       (Stage 2 RM, frozen)
   │   KL(π_θ || π_ref)               (per-token, on the fly)
   ↓
r̃_t = R · 1[t=T] − β log(π_θ/π_ref)
   │
   │   GAE on r̃_t → advantages
   ↓
PPO update on π_θ and V_φ
```

> ⚠️ **Why do we need a reference policy?** — Without a KL anchor, the policy will **severely over-optimize the RM** (reward hacking): outputs become longer, repetitive nonsense, sycophantic ("As an AI..."), drifting in style to the RM's biased samples. $\beta = 0.01 \sim 0.1$ is the common range for InstructGPT; too small leads to drift, too large prevents learning.

### 3.4　In practice: 4 models resident simultaneously

PPO RLHF training has **4 models in memory simultaneously**:

1. **Policy** $\pi_\theta$ (trainable)
2. **Reference policy** $\pi_\text{ref}$ (frozen, for KL)
3. **Reward model** $r_\psi$ (frozen, for reward)
4. **Value model** $V_\phi$ (trainable, required by PPO)

This is the root cause of RLHF's memory consumption; 4× base model + optimizer state + gradient = easy to blow up. **This is also the fundamental reason DPO / GRPO are widely accepted in LLM RL — they each chop off some of the models.**

## §4 DPO: Closed-Form Direct Preference Optimization

Rafailov et al. 2023 NeurIPS, *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*, is the milestone in RLHF simplification. **Core observation**: the KL-regularized RL problem has a closed-form optimum, which can be inverted to give an implicit reward, replacing PPO with pure supervised learning.

### 4.1　KL-regularized optimal policy (key step)

Consider the RLHF objective (formula §3.3):

$$\max_{\pi} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi(\cdot|x)}\big[r(x, y)\big] - \beta\, \text{KL}\!\big(\pi(\cdot|x) \| \pi_\text{ref}(\cdot|x)\big)$$

**Solve for the optimal $\pi^*(\cdot | x)$ for a single $x$** (Lagrangian):

$$\mathcal{L}_x[\pi] = \sum_y \pi(y|x) r(x, y) - \beta \sum_y \pi(y|x) \log \frac{\pi(y|x)}{\pi_\text{ref}(y|x)} + \mu\!\left(1 - \sum_y \pi(y|x)\right)$$

Take partial derivative with respect to $\pi(y|x)$ and set = 0:

$$r(x, y) - \beta\!\left(\log \frac{\pi(y|x)}{\pi_\text{ref}(y|x)} + 1\right) - \mu = 0$$

Rearranging:

$$\log \pi^*(y|x) = \log \pi_\text{ref}(y|x) + \frac{r(x, y)}{\beta} - \frac{\mu + \beta}{\beta}$$

Let $\log Z(x) = (\mu + \beta)/\beta$ (log of partition function), getting:

$$\boxed{\;\pi^*(y|x) = \frac{1}{Z(x)} \pi_\text{ref}(y|x) \exp\!\left(\frac{r(x, y)}{\beta}\right), \quad Z(x) = \sum_y \pi_\text{ref}(y|x) \exp\!\left(\frac{r(x, y)}{\beta}\right)\;}$$

### 4.2　Inversion: implicit reward

Taking the log of the above and solving for $r$:

$$r(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_\text{ref}(y|x)} + \beta \log Z(x)$$

**Key insight**: the reward is the log-ratio of $\pi^*$ to $\pi_\text{ref}$ (plus a term $\beta \log Z(x)$ that does not depend on $y$). So once we have $\pi^*$, we have the reward; conversely as well.

### 4.3　Plug into Bradley-Terry to get DPO loss

Bradley-Terry gives preference probability:

$$P(y_w \succ y_l | x) = \sigma\!\big(r(x, y_w) - r(x, y_l)\big)$$

Substituting §4.2's inversion (**note $\beta \log Z(x)$ does not depend on $y$ and cancels in $r(x, y_w) - r(x, y_l)$!**):

$$r(x, y_w) - r(x, y_l) = \beta \log \frac{\pi^*(y_w|x)}{\pi_\text{ref}(y_w|x)} - \beta \log \frac{\pi^*(y_l|x)}{\pi_\text{ref}(y_l|x)}$$

Replacing the target $\pi^*$ with the learnable $\pi_\theta$, we get the **DPO loss**:

$$\boxed{\;\mathcal{L}_\text{DPO}(\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}\log \sigma\!\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)}\right)\;}$$

> ✅ **Why is DPO so "counterintuitively" effective?** —

- It has **no explicit reward model**, but the implicit reward = $\beta \log(\pi/\pi_\text{ref})$.
- It **needs no sampling** (unlike PPO requiring on-policy rollouts), the entire training is offline contrastive learning.
- The KL constraint is written **implicitly** into the loss (via $\pi_\text{ref}$ in the log-ratio denominator).
- The whole pipeline becomes SFT-style: one pass of supervised learning and you're done.

### 4.4　Implicit meaning of the DPO gradient

Take the gradient with respect to $\theta$ (derivation details in original paper Appendix):

$$\nabla_\theta \mathcal{L}_\text{DPO} = -\beta \mathbb{E}\big[\sigma(\hat{r}_l - \hat{r}_w)\big(\nabla_\theta \log \pi_\theta(y_w|x) - \nabla_\theta \log \pi_\theta(y_l|x)\big)\big]$$

where $\hat{r} = \beta \log(\pi_\theta/\pi_\text{ref})$ is the implicit reward. **Intuition**:

- The coefficient $\sigma(\hat{r}_l - \hat{r}_w)$ is "the current model's confidence in the wrong direction $y_l \succ y_w$" — the more confidently wrong, the bigger the weight (hard-example mining effect).
- Then increase $\log \pi_\theta(y_w | x)$ and decrease $\log \pi_\theta(y_l | x)$.

### 4.5　Code (core DPO loss in 30 lines)

```python
import torch
import torch.nn.functional as F

def dpo_loss(policy, ref_policy, batch, beta=0.1):
    """
    batch:
      chosen_ids:    [B, L]     prompt + y_w
      rejected_ids:  [B, L]     prompt + y_l
      chosen_mask:   [B, L]     1 for y_w response tokens, 0 elsewhere (prompt + pad)
      rejected_mask: [B, L]
    """
    # log π(y | x) = sum_t log π(y_t | x, y_<t) over response tokens
    def log_prob_sum(model, ids, mask):
        logits = model(ids).logits[:, :-1]                  # [B, L-1, V]
        logp = F.log_softmax(logits, dim=-1)
        tgt = ids[:, 1:].unsqueeze(-1)                      # [B, L-1, 1]
        token_logp = logp.gather(-1, tgt).squeeze(-1)       # [B, L-1]
        token_mask = mask[:, 1:]                            # align with next-token prediction
        return (token_logp * token_mask).sum(dim=-1)        # [B]

    pi_w  = log_prob_sum(policy,     batch["chosen_ids"],   batch["chosen_mask"])
    pi_l  = log_prob_sum(policy,     batch["rejected_ids"], batch["rejected_mask"])
    with torch.no_grad():
        ref_w = log_prob_sum(ref_policy, batch["chosen_ids"],   batch["chosen_mask"])
        ref_l = log_prob_sum(ref_policy, batch["rejected_ids"], batch["rejected_mask"])

    # log-ratios
    logits_w = pi_w - ref_w        # = log(π/π_ref)(y_w)
    logits_l = pi_l - ref_l
    diff = beta * (logits_w - logits_l)

    loss = -F.logsigmoid(diff).mean()
    # implicit reward margin (for monitoring)
    chosen_reward   = beta * logits_w.detach()
    rejected_reward = beta * logits_l.detach()
    margin = (chosen_reward - rejected_reward).mean()
    accuracy = (chosen_reward > rejected_reward).float().mean()
    return loss, {"loss": loss.item(), "margin": margin.item(),
                  "accuracy": accuracy.item()}
```

> ⚠️ **Known DPO failure modes** —

- **Likelihood decreases for both $y_w$ and $y_l$** (Pal et al. 2024, Saeidi et al.): DPO only requires that $\log\pi(y_w) - \log\pi(y_l)$ increases, not that $\log\pi(y_w)$ increases. In practice both often decrease together, with $y_l$ decreasing more — which can make the model "unwilling to say anything."
- **Sensitive to $\beta$**: too small → loses KL constraint → reward hacking; too large → fails to learn. Common range $0.05 \sim 0.5$.
- **Data quality determines everything**: when preference pairs are noisy, DPO is more brittle than PPO (PPO buffers via RM ensemble, DPO eats raw data directly).
- **Off-policy bias**: preference data is sampled by $\pi_\text{SFT}$, but $\pi_\theta$ drifts during training, creating distribution shift.

## §5 GRPO: Group Relative Advantage

DeepSeekMath (Shao et al. 2024 arXiv 2402.03300) proposes **Group Relative Policy Optimization**, and DeepSeek-R1 (DeepSeek-AI 2025 arXiv 2501.12948) pushes it to the flagship of reasoning.

### 5.1　Motivation

PPO's value model is hard to train for LLMs:

- LLM token-level value has no clear semantics (intermediate tokens usually have reward = 0, only terminal has reward)
- Value model is the same size as policy → doubled memory

GRPO's **core idea**: for each prompt $x$, **sample a group of $G$ responses** $\{y_1, \dots, y_G\}$, score them with RM giving $G$ rewards $\{r_1, \dots, r_G\}$, **advantage directly uses within-group normalization**:

$$\boxed{\;\hat{A}_{i, t} = \frac{r_i - \text{mean}(\{r_1, \dots, r_G\})}{\text{std}(\{r_1, \dots, r_G\}) + \epsilon}\;}$$

Within the entire response, **all tokens share the same $\hat{A}_i$** (since there is no value model providing per-token baseline, only sequence-level reward and group-relative advantage).

> ✅ **The essence of GRPO** — Replace PPO's "$A_t = Q - V$" (Critic gives $V$) with "$A_i = (r_i - \bar{r}) / \sigma$" (within-group statistics give baseline). **Drop the value model**, saving half the memory; meanwhile within-group baseline automatically does variance reduction.

### 5.2　GRPO objective

GRPO keeps PPO-Clip's form but puts the KL penalty into the loss (not into the reward):

Let importance ratio $\rho_{i, t}(\theta) = \pi_\theta(y_{i,t}|x_i, y_{i,\lt t}) / \pi_{\theta_\text{old}}(y_{i,t}|x_i, y_{i,\lt t})$ (avoiding confusion with sequence-level reward $r_i$):

$$L^\text{GRPO}(\theta) = \mathbb{E}\!\left[\frac{1}{G}\sum_{i=1}^G \frac{1}{|y_i|}\sum_{t=1}^{|y_i|} \Big(\min(\rho_{i, t} \hat{A}_{i, t},\, \text{clip}(\rho_{i, t}, 1\!-\!\epsilon, 1\!+\!\epsilon) \hat{A}_{i, t}) - \beta\, \text{KL}_{i, t}(\pi_\theta \| \pi_\text{ref})\Big)\right]$$

where KL uses the K3 estimator (Schulman blog 2020):

$$\text{KL}_{i, t} = \frac{\pi_\text{ref}(y_{i,t}|\cdot)}{\pi_\theta(y_{i,t}|\cdot)} - \log\frac{\pi_\text{ref}(y_{i,t}|\cdot)}{\pi_\theta(y_{i,t}|\cdot)} - 1$$

This estimator is guaranteed non-negative and low-variance, more stable than directly using $\log(\pi_\theta/\pi_\text{ref})$.

### 5.3　DeepSeek-R1's key modifications

DeepSeek-R1 (Jan 2025) does two things on top of GRPO:

1. **R1-Zero**: directly run GRPO from pretrain base, **without an SFT stage**, purely with rule-based reward (math correctness + format reward). Emergent long CoT.
2. **R1**: add a small SFT cold-start + multi-stage RL (reasoning RL → SFT → general RL). The open-source 32B/70B compete with o1 on math reasoning.

> 💡 **Why is GRPO particularly effective for math/code RL?** —

- **Rule-based reward**: math has unique answers, code has unit tests, **bypassing the neural RM, reward hacking risk is significantly reduced** (reward signal is close to ground truth, but the policy can still find loopholes in the grader, e.g. guessing answers, writing only "42" without reasoning, exploiting test coverage blind spots, etc., so "no hacking at all" is too strong a claim).
- **Within-group normalization**: same problem sampled $G=16$ solutions, automatically figuring out "which reasoning paths are better," without needing absolute scale.
- **Save value model**: math prompts are many and reasoning is long, saving half the memory enables larger batches.

### 5.4　Code (core GRPO advantage + loss in 50 lines)

```python
import torch
import torch.nn.functional as F

def grpo_loss(policy, ref_policy, batch, eps_clip=0.2, beta=0.04):
    """
    batch:
      input_ids:     [N, L]       N = sum_b G_b samples in the batch
      action_mask:   [N, L]
      old_log_probs: [N, L]
      rewards:       [N]          sequence-level reward (rule-based or RM)
      group_id:      [N]          which prompt each sample belongs to (long tensor)
    """
    rewards = batch["rewards"]
    gid = batch["group_id"].long()
    N = rewards.shape[0]

    # ── Within-group normalization (GRPO core) ──
    # Use scatter to aggregate mean / std per group, no assumption that group_id is sorted or equal-sized.
    num_groups = int(gid.max().item()) + 1
    counts = torch.zeros(num_groups, device=rewards.device).scatter_add_(
        0, gid, torch.ones_like(rewards))                              # [num_groups]
    sums = torch.zeros(num_groups, device=rewards.device).scatter_add_(
        0, gid, rewards)                                                # [num_groups]
    group_mean = sums / counts.clamp_min(1.0)                          # [num_groups]
    diff_sq = (rewards - group_mean[gid]) ** 2
    sq_sums = torch.zeros(num_groups, device=rewards.device).scatter_add_(
        0, gid, diff_sq)                                                # [num_groups]
    group_var = sq_sums / counts.clamp_min(1.0)
    group_std = group_var.sqrt()                                       # [num_groups]

    A = (rewards - group_mean[gid]) / (group_std[gid] + 1e-8)          # [N]
    A = A.unsqueeze(-1)                                                # [N, 1] shared across the whole sequence

    # ── log-prob ratio ──
    logits = policy(batch["input_ids"]).logits[:, :-1]
    log_probs = F.log_softmax(logits, dim=-1)
    tgt = batch["input_ids"][:, 1:].unsqueeze(-1)
    new_log_probs = log_probs.gather(-1, tgt).squeeze(-1)
    new_log_probs = F.pad(new_log_probs, (1, 0))        # [B*G, L]
    mask = batch["action_mask"].float()

    ratio = torch.exp(new_log_probs - batch["old_log_probs"])

    # ── PPO-Clip surrogate (advantage shared across the sequence) ──
    surr1 = ratio * A                                    # [N, L]
    surr2 = torch.clamp(ratio, 1.0 - eps_clip, 1.0 + eps_clip) * A

    # ── KL penalty (K3 estimator: KL ≈ exp(Δ) - Δ - 1) ──
    with torch.no_grad():
        ref_logits = ref_policy(batch["input_ids"]).logits[:, :-1]
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)
        ref_token_lp = ref_log_probs.gather(-1, tgt).squeeze(-1)
        ref_token_lp = F.pad(ref_token_lp, (1, 0))
    delta = ref_token_lp - new_log_probs                 # log(π_ref / π_θ)
    kl_per_token = torch.exp(delta) - delta - 1.0        # K3, non-negative [N, L]

    # Formula: (1/G) Σ_i (1/|y_i|) Σ_t [ min(...) - β·KL_{i,t} ]
    # Implementation: each sample first normalized by response length |y_i|, then averaged over batch
    token_obj = torch.min(surr1, surr2) - beta * kl_per_token   # [N, L]
    seq_len = mask.sum(dim=-1).clamp_min(1.0)            # [N] = |y_i|
    per_seq = (token_obj * mask).sum(dim=-1) / seq_len   # [N]
    loss = -per_seq.mean()                               # Negative sign: maximize -> minimize

    # Monitoring
    with torch.no_grad():
        policy_term = -(torch.min(surr1, surr2) * mask).sum(dim=-1) / seq_len
        kl_term = (kl_per_token * mask).sum(dim=-1) / seq_len
    return loss, {"policy": policy_term.mean().item(),
                  "kl": kl_term.mean().item(),
                  "reward_mean": rewards.mean().item(),
                  "advantage_std": A.squeeze(-1).std().item()}
```

> ⚠️ **GRPO engineering essentials** —

- $G$ is usually 8/16/32: too small means noisy within-group statistics, too large blows up memory.
- The degenerate case where `std` is 0 (all rewards in group are the same, all-pass / all-fail): add $\epsilon$ or drop the group.
- **DAPO** (ByteDance Seed, 2025 arXiv 2503.14476) makes several improvements to GRPO: clip higher (breaking symmetric clip), dynamic sampling (drop all-correct/all-wrong groups), token-level loss, overlong shaping — it is the industrial-grade evolution of GRPO.
- **DeepSeek also discusses GRPO's limitations in the R1 report**: within-group baseline fails when reward variance is small or group size is small; subsequent work (e.g. VAPO, CISPO) tries to improve.

## §6 DPO Variant Ecosystem

After DPO, 2024–2025 saw a surge of improvements, and interviews frequently ask "what's the difference between X and Y." The table below compares **core loss differences**; notation is unified as: $\pi$ = current policy, $\pi_\text{ref}$ = SFT model, $\beta$ = inverse temperature, $r$ = implicit reward.

| Method | Loss | Key change | Paper |
| --- | --- | --- | --- |
| **DPO** | $-\log\sigma(\beta\log\tfrac{\pi(y_w)}{\pi_\text{ref}(y_w)} - \beta\log\tfrac{\pi(y_l)}{\pi_\text{ref}(y_l)})$ | Baseline | Rafailov 2023 NeurIPS |
| **IPO** | $(\log\tfrac{\pi(y_w)/\pi_\text{ref}(y_w)}{\pi(y_l)/\pi_\text{ref}(y_l)} - \tfrac{1}{2\beta})^2$ | sigmoid → squared loss, **prevents reward overfitting** | Azar 2024 AISTATS |
| **KTO** | per-sample sigmoid loss, only needs binary good/bad, **no pairs needed** | Kahneman-Tversky inspired, desirable/undesirable | Ethayarajh 2024 ICML |
| **SimPO** | $-\log\sigma(\beta\cdot\tfrac{1}{\lvert y_w\rvert}\log\pi(y_w) - \beta\cdot\tfrac{1}{\lvert y_l\rvert}\log\pi(y_l) - \gamma)$ | **Reference-free**, length-normalized, with margin $\gamma$ | Meng 2024 NeurIPS |
| **ORPO** | $\mathcal{L}_\text{SFT}(y_w) + \lambda\cdot\mathcal{L}_\text{OR}$, odds-ratio preference | **Single-stage SFT + preference**, no reference needed | Hong 2024 EMNLP |
| **RLOO** | REINFORCE + leave-one-out baseline from $k$ samples | Online RL, value-free, strong baseline | Ahmadian 2024 ACL |
| **ReMax** | REINFORCE + greedy baseline (greedy decode on same prompt as baseline) | Further save sampling | Li 2024 ICML |

### 6.1　IPO: prevent reward overfit

Azar et al. 2024 *A General Theoretical Paradigm to Understand Learning from Human Preferences* points out: Bradley-Terry + DPO will blow up the implicit reward to infinity when preferences are deterministic ("$y_w$ always wins"), causing overfit. **IPO replaces sigmoid with squared loss**:

$$\mathcal{L}_\text{IPO} = \mathbb{E}\!\left[\left(\log\frac{\pi(y_w)/\pi_\text{ref}(y_w)}{\pi(y_l)/\pi_\text{ref}(y_l)} - \frac{1}{2\beta}\right)^2\right]$$

Intuition: aim for log-ratio equal to a **fixed margin** $1/(2\beta)$, not increasingly large.

### 6.2　KTO: no need for pairs

Ethayarajh et al. 2024 ICML *KTO: Model Alignment as Prospect Theoretic Optimization*. Production data is often a single label ("this is good" / "this is bad"), not pairs. KTO is inspired by Prospect Theory (Kahneman-Tversky) and **symmetrically pushes desirable / undesirable samples away from a reference point**:

Let implicit reward $\hat{r}_\theta(x, y) = \beta \log \tfrac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)}$, reference point $z_0 = \mathbb{E}_{x',\, y' \sim \pi_\theta(\cdot | x')}\!\big[\beta \cdot \text{KL}\!\big(\pi_\theta(\cdot | x') \,\|\, \pi_\text{ref}(\cdot | x')\big)\big]$ (estimated via mismatched pairs within the batch, **does not participate in backprop** — treated as a constant). The KTO loss is:

$$\boxed{\;\mathcal{L}_\text{KTO}(\theta) = \mathbb{E}_{(x, y) \sim \mathcal{D}} \big[\, w(y)\big(1 - v(x, y)\big)\,\big]\;}$$

where piecewise definitions ($\hat r$ and $z_0$ already contain $\beta$, **no additional $\beta$** in the sigmoid):

$$v(x, y) = \begin{cases}\sigma\!\big(\hat{r}_\theta(x, y) - z_0\big) & y \text{ desirable}\\[2pt] \sigma\!\big(z_0 - \hat{r}_\theta(x, y)\big) & y \text{ undesirable}\end{cases}$$

$$w(y) = \begin{cases}\lambda_D & y \text{ desirable}\\ \lambda_U & y \text{ undesirable}\end{cases}$$

Intuition: desirable samples want $\hat{r} > z_0$ (push up), undesirable samples want $\hat{r} < z_0$ (push down); $\sigma$ gives a probability score of "how far from the reference point." $\lambda_D, \lambda_U$ are weights for imbalanced data (e.g. if desirable is more, $\lambda_U > \lambda_D$). When data only has single thumbs up/down, KTO is more practical than DPO. See Ethayarajh et al. 2024 ICML, HuggingFace TRL `KTOTrainer` for details.

### 6.3　SimPO: drop the reference

Meng et al. 2024 NeurIPS *SimPO: Simple Preference Optimization with a Reference-Free Reward*. SimPO observes: **DPO does not use $\pi_\text{ref}$ at inference, but stores it during training (occupies memory)**. SimPO directly changes the implicit reward to length-normalized log-prob:

$$r_\text{SimPO}(x, y) = \frac{\beta}{|y|}\log\pi(y|x)$$

Loss:

$$\boxed{\;\mathcal{L}_\text{SimPO} = -\mathbb{E}\log\sigma\!\left(\frac{\beta}{|y_w|}\log\pi(y_w|x) - \frac{\beta}{|y_l|}\log\pi(y_l|x) - \gamma\right)\;}$$

where $\gamma$ is the target reward margin (encourages $r_w - r_l \ge \gamma$, no loss past that).

> ✅ **SimPO advantages** —

- **No $\pi_\text{ref}$ needed during training**, saves a copy of model weights in memory.
- **Length-normalization** mitigates DPO's length bias (DPO favors long answers since their log-probs are more negative, with larger differences).
- Experimentally on AlpacaEval-2 / Arena-Hard, SimPO often outperforms DPO; but the reward-free form **lacks a KL anchor**, requiring more careful tuning of $\beta, \gamma$.

### 6.4　ORPO: single-stage SFT + preference

Hong et al. 2024 EMNLP *ORPO: Monolithic Preference Optimization without Reference Model*. ORPO removes the two-stage "SFT then DPO," doing **SFT and preference simultaneously in one training pass**:

$$\mathcal{L}_\text{ORPO} = \mathcal{L}_\text{SFT}(y_w) - \lambda \cdot \log\sigma\!\left(\log\frac{\text{odds}_\theta(y_w|x)}{\text{odds}_\theta(y_l|x)}\right)$$

where odds = $p / (1 - p)$. **No reference model needed**, but it requires the chosen response to be high quality (driven by the SFT part).

### 6.5　RLOO / ReMax

Ahmadian et al. 2024 ACL *Back to Basics: Revisiting REINFORCE Style Optimization*. On LLM RLHF, **simple REINFORCE + multi-sample baseline empirically outperforms PPO**:

- **RLOO**: for each prompt sample $k$ responses, the baseline for each is the average reward of the other $k-1$ (leave-one-out), $\hat{A}_i = r_i - \frac{1}{k-1}\sum_{j\ne i} r_j$.
- **ReMax** (Li et al. 2024 ICML): baseline directly uses the **greedy decode** reward for the same prompt, only needing 1 sample + 1 greedy per step.

Both **drop the value model**, in the same lineage as GRPO — fundamentally all using **multi-sample or greedy baselines to replace the critic**; the difference is only in baseline form.

> 💡 **GRPO vs RLOO vs ReMax distinctions** —

- **GRPO**: within-group mean+std normalization ($z$-score) + PPO-clip.
- **RLOO**: within-group leave-one-out mean as baseline + REINFORCE.
- **ReMax**: greedy decode as baseline + REINFORCE.
- The three share a philosophy: **bypass the value model, use sample baselines**; differing only in baseline form.

## §7 Advanced Reward Modeling

### 7.1　PRM vs ORM

| Dimension | **ORM** (Outcome RM) | **PRM** (Process RM) |
| --- | --- | --- |
| **Scoring granularity** | Only the final answer | Each reasoning step |
| **Annotation cost** | Low (only result) | High (every step labeled) |
| **Reward signal** | Sparse (end of trajectory) | Dense (each step) |
| **Math reasoning performance** | Medium | **Good** |
| **Typical usage** | Best-of-N + ORM rerank | PRM for step-level search / RL |
| **Representative paper** | InstructGPT (Ouyang 2022) | Let's Verify Step by Step (Lightman 2023 arXiv / 2024 ICLR) |

Lightman et al. *Let's Verify Step by Step* (OpenAI arXiv 2305.20050, 2023; ICLR 2024) compared ORM and PRM on the MATH dataset:

- PRM800K: 800K step-level annotations (each step labeled ✓ / ⚠️ / ✗).
- On GPT-4 generator, PRM rerank over 1860 candidates gives MATH accuracy **78.2%**, significantly higher than ORM's 72.4%.

**Math-Shepherd** (Wang et al. 2024 ACL *Math-Shepherd: Verify and Reinforce LLMs Step-by-Step without Human Annotations*) uses **rollout-based automatic annotation of PRM**: from each step run $K$ subsequent generations, automatically giving soft labels per step based on correctness rate, removing the need for human labels.

### 7.2　Constitutional AI / RLAIF

**Constitutional AI (Bai et al. 2022 Anthropic)** *Constitutional AI: Harmlessness from AI Feedback*:

1. **SL stage**: have the AI rewrite harmful responses itself according to "constitution" principles.
2. **RL stage**: use AI (rather than humans) for preference annotation → train RM → PPO (i.e. RLAIF).

The entire harmless training **needs no human preference annotation**, relying on LLM self-evaluation.

**RLAIF (Lee et al. 2023 Google)** *RLAIF: Scaling RLHF with AI Feedback*: on tasks like summarization, helpful dialogue, systematically verifies AI feedback ≈ human feedback in effect. **Conditions**: the LLM judge must be strong enough (GPT-4 / Claude-3 level), and the prompt must be well designed.

> ⚠️ **RLAIF risks** — If the judging LLM has its own biases (sycophancy, length preference), the biases get amplified to the student model. In production this is usually **used in combination**: rule-based (math/code) + RM (general quality) + RLAIF (safety).

### 7.3　Reward model ensemble & uncertainty

Recent years (Coste 2024, Eisenstein 2024) have found: **single RM is easily hacked**, and ensemble of RMs (5-7 RMs with different seeds) combined with conservative aggregation (min / mean - $k\cdot$std) can significantly mitigate reward hacking without losing performance. The cost is doubled memory — hence industrial deployment uses lightweight RM (e.g. 6B evaluating 70B policy).

## §8 Reward Hacking & Engineering Experience

Reward hacking is the central pain point of RL post-training: the model finds blind spots in the RM to get high scores, but humans find the result worse. Common symptoms:

| Phenomenon | Cause | Mitigation |
| --- | --- | --- |
| **Answers get longer** | Long answers typically score higher in RM training data (information content, polite filler) | length penalty / length normalize (SimPO) / length-regression calibration |
| **Sycophancy** | Users prefer agreement → RM learns "agree better" | sycophancy probe + adversarial data |
| **Repetition / boilerplate** | Closing phrases like "Hope this helps!" score high in RM data | n-gram repetition penalty / diversity reward |
| **Over-refusal** | Safety RM learns "refuse = safe" | helpfulness vs harmlessness dual RM balance |
| **Format hacks** | Using markdown / bullets / emoji to score | format normalization |
| **Number manipulation** | Hard-coding common answer distributions in math problems | rule-based reward replaces RM |

### 8.1　Core mitigation mechanisms

1. **KL penalty** ($\beta \log\pi/\pi_\text{ref}$): the most basic anchor.
2. **Early stopping by KL budget**: stop when $\text{KL}(\pi || \pi_\text{ref}) > K_\text{target}$.
3. **Reward model ensemble** (Coste 2024 ICLR *Reward Model Ensembles Help Mitigate Overoptimization*): take min or mean - std of multiple RMs.
4. **Reward shaping**: decompose reward into multiple terms (helpfulness + length + diversity), capping each individually.
5. **Offline + online mix**: first DPO / KTO to get to 70 points, then PPO + strong RM for the last mile.
6. **Composite reward**: weighted combination of rule-based + RM-based, the rule part cannot be hacked.

### 8.2　Gao 2023 scaling law

Gao, Schulman, Hilton 2023 ICML *Scaling Laws for Reward Model Overoptimization*:

- Proxy reward (RM score) monotonically rises with KL, but gold reward rises then falls (**inverted-U**).
- Best-of-N's over-optimization differs slightly in form from PPO: BoN's gold reward is close to linear-minus-plus-linear in $\sqrt{\text{KL}}$, PPO has more higher-order terms.
- Provides a scaling law that larger RM → slower over-optimization, also discussing different functional forms for fitting.

**Interview takeaway**: reward hacking is not a bug, it is the essence of RL — the inevitable phenomenon of proxy and true objective being separated.

## §9 Complexity / Resource Comparison

### 9.1　Resource comparison of four paradigms

| Dimension | **PPO RLHF** | **DPO family** | **GRPO** | **RLOO/ReMax** |
| --- | --- | --- | --- | --- |
| Need RM? | ✅ frozen | ❌ implicit | ✅ frozen or rule | ✅ frozen or rule |
| Need value model? | ✅ trainable | ❌ | ❌ | ❌ |
| Need reference policy? | ✅ frozen | ✅ frozen (except SimPO/ORPO) | ✅ frozen | ✅ frozen or none |
| Online sampling? | ✅ on-policy | ❌ offline | ✅ on-policy | ✅ on-policy |
| Number of model copies | **4** (π, π_ref, RM, V) | **2** (π, π_ref) | **3** (π, π_ref, RM) | **3** |
| KL anchor | Explicitly in reward | Implicitly in loss | Explicitly in loss | Explicitly in reward or loss |
| Main bottleneck | Memory + slow on-policy | Preference data quality | Multiple-times sampling | Multiple-times sampling |
| Tuning difficulty | **High** ($\beta, \epsilon, \lambda, c_v, c_e$) | Medium ($\beta$) | Medium ($\beta, G$) | Low ($\beta, k$) |
| Representative | InstructGPT, Claude | Llama-3 instruct, Zephyr | DeepSeek-R1, Kimi-K1.5 | Cohere Command R |

### 9.2　Rough LLM RLHF training memory estimation

For 7B base, fp16 forward + fp32 master, AdamW (typical RLHF):

| Copy | Use | bf16 weights + grads | fp32 (master + Adam m,v) | Total |
| --- | --- | --- | --- | --- |
| Policy $\pi_\theta$ | trainable | 14+14=28 GB | 28+28+28=84 GB | **112 GB** |
| Value $V_\phi$ | trainable | 28 GB | 84 GB | **112 GB** |
| Reference $\pi_\text{ref}$ | frozen | 14 GB | — | 14 GB |
| Reward $r_\psi$ | frozen | 14 GB | — | 14 GB |
| **Total** | | | | **~252 GB** (each trainable model = 16 B/param ×7B = 112 GB; matches the distributed-training tutorial) |

Plus activation, KV cache, generation buffer, a single 80GB card is hard to fit; typically using ZeRO-3 + offload or multi-machine sharding.

**DPO with same base**: only needs $\pi_\theta$ + $\pi_\text{ref}$ = 70 + 14 = **84 GB** (half).

**GRPO with same base**: $\pi_\theta$ + $\pi_\text{ref}$ + RM = 70 + 14 + 14 = **98 GB** (saves one value copy).

### 9.3　Training throughput comparison

Measured (open-source 7B, 8×H100, TRL/OpenRLHF report):

- PPO: ~50-100 token/s/GPU (bottleneck: on-policy generation + 4 copies)
- DPO: ~500-1000 token/s/GPU (pure supervised, no generation)
- GRPO: ~100-200 token/s/GPU (needs generation, but one fewer model)
- SimPO: ~600-1200 token/s/GPU (DPO further drops reference)

DPO is fast because it is **fully offline**, almost SFT speed; PPO is slow mainly due to rollout (generate a response before one gradient).

## §10 25 Frequently-Asked Interview Questions

Divided into 3 tiers by difficulty: L1 = asked at any LLM engineering position; L2 = asked by research/alignment teams; L3 = hardcore questions from top labs / DeepSeek-class teams. Each question expands to answer points + common pitfalls.

### L1 Must-Know (10 questions)

<details>

<summary>Q1. What are the three stages of RLHF?</summary>

- Stage 1: SFT (Supervised Fine-Tuning)
- Stage 2: RM (train reward model with Bradley-Terry, frozen)
- Stage 3: PPO + KL penalty (policy optimizes RM score, constrained by KL to reference policy)

Pitfalls: skipping SFT and saying "RM + PPO"; or omitting the KL penalty.

</details>

<details>

<summary>Q2. What is the PPO clipped surrogate formula?</summary>

- $r_t = \pi_\theta / \pi_{\theta_\text{old}}$ is the importance ratio
- $L^\text{CLIP} = \mathbb{E}[\min(r_t A_t,\, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t)]$
- Taking min is a pessimistic estimate (pessimistic bound); upper bound $1+\epsilon$ when advantage is positive, lower bound $1-\epsilon$ when negative

Pitfalls: saying "clip the reward"; or missing the role of min.

</details>

<details>

<summary>Q3. What is GAE? How to pick $\lambda$?</summary>

- $A_t^\text{GAE} = \sum_l (\gamma\lambda)^l \delta_{t+l}$, $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$
- $\lambda = 0$ → TD(0), high bias low variance
- $\lambda = 1$ → Monte Carlo, low bias high variance
- Typical $\lambda = 0.95$, $\gamma = 0.99$

Pitfalls: treating GAE as the advantage itself (it is an estimator); or forgetting that $\gamma\lambda$ is the joint decay coefficient.

</details>

<details>

<summary>Q4. What is the RM loss function?</summary>

- Bradley-Terry pairwise: $\mathcal{L} = -\mathbb{E}\log\sigma(r(x, y_w) - r(x, y_l))$
- Initialize backbone with SFT model, add scalar head on top
- Frozen after training, used by PPO

Pitfalls: saying RM uses BCE / MSE; or writing it as absolute reward supervision.

</details>

<details>

<summary>Q5. What is the DPO loss formula? What's its relation to RM loss?</summary>

- $\mathcal{L}_\text{DPO} = -\log\sigma(\beta\log\frac{\pi_\theta(y_w)}{\pi_\text{ref}(y_w)} - \beta\log\frac{\pi_\theta(y_l)}{\pi_\text{ref}(y_l)})$
- Same form as Bradley-Terry RM loss, but **implicit reward** is $\beta\log(\pi/\pi_\text{ref})$, not a separate RM
- Origin: KL-regularized RLHF optimum $\pi^* \propto \pi_\text{ref}\exp(r/\beta)$, inverting gives $r = \beta\log(\pi/\pi_\text{ref}) + \beta\log Z$, and $\log Z$ cancels in the pairwise difference

Pitfalls: memorizing the formula without knowing the closed-form derivation; or saying "DPO needs no reference" (wrong, needs $\pi_\text{ref}$ to compute log-ratio).

</details>

<details>

<summary>Q6. What data does DPO training need?</summary>

- Preference pairs $(x, y_w, y_l)$: for the same prompt, humans or AI judge $y_w \succ y_l$
- Need $\pi_\text{ref}$ (usually the SFT model)
- **Don't need** RM, value model, online sampling

Pitfalls: saying it needs reward scalar annotations; or forgetting $\pi_\text{ref}$.

</details>

<details>

<summary>Q7. What does GRPO save compared to PPO?</summary>

- **Drops the value model**: advantage uses within-group normalization $\hat{A}_i = (r_i - \bar{r}) / \sigma_r$
- Less memory; less tuning (no $c_v$, no value lr)
- Suitable for LLM RL since LLM's per-token value is hard to learn

Pitfalls: saying it saves the RM (wrong, GRPO still needs RM or rule-based reward); or saying GRPO is offline (wrong, it is on-policy).

</details>

<details>

<summary>Q8. What role does the KL penalty play in RLHF?</summary>

- Add $-\beta\log\pi/\pi_\text{ref}$ to the reward (per-token KL)
- Prevents the policy from drifting too far from SFT, mitigating reward hacking
- $\beta$ too small → drift, $\beta$ too large → fails to learn
- In DPO, implicitly implemented via $\pi_\text{ref}$

Pitfalls: saying KL is part of the RM (wrong, KL is between policy and reference, does not involve RM).

</details>

<details>

<summary>Q9. Why do RL/DPO after SFT?</summary>

- SFT can only mimic positive examples, **cannot learn contrastive signals** (A better than B)
- RM explicitly models "which is better," policy optimizes "obtain high RM score"
- DPO skips the RM but keeps the same contrastive signal
- Empirically, helpfulness, harmlessness, honesty all improve significantly after RLHF/DPO (InstructGPT report)

Pitfalls: only saying "RL is stronger than SFT"; or not knowing about the contrastive signal point.

</details>

<details>

<summary>Q10. What is reward hacking? How to mitigate?</summary>

- The policy over-optimizes the RM, finding blind spots to score high, but humans find it worse
- Typical symptoms: longer answers, sycophancy, repeated boilerplate, over-refusal
- Mitigation: KL penalty, RM ensemble, length penalty, composite reward (rule + RM), early stopping by KL budget

Pitfalls: only saying "add KL," not knowing other mitigations; or not knowing about length bias.

</details>

### L2 Advanced (10 questions)

<details>

<summary>Q11. Derive DPO loss (from KL-regularized RLHF).</summary>

1. Objective: $\max_\pi \mathbb{E}[r] - \beta \text{KL}(\pi || \pi_\text{ref})$
2. Lagrangian + derivative w.r.t. $\pi$ $\Rightarrow \pi^*(y|x) = \frac{1}{Z(x)}\pi_\text{ref}(y|x)\exp(r/\beta)$
3. Inversion $r(x, y) = \beta\log(\pi^*/\pi_\text{ref}) + \beta\log Z(x)$
4. Plug into Bradley-Terry $P(y_w \succ y_l) = \sigma(r_w - r_l)$, **$\log Z$ cancels**
5. Replace $\pi^*$ with learnable $\pi_\theta$, do NLL on preference data → DPO loss

Pitfalls: memorizing the formula, can't answer "why does $\log Z$ cancel" (because it doesn't depend on $y$, it subtracts away).

</details>

<details>

<summary>Q12. Why does PPO need importance ratio $r_t = \pi_\theta/\pi_\text{old}$?</summary>

- Standard PG is on-policy, but PPO does multiple updates on the same batch (K epochs)
- From the 2nd onwards $\pi_\theta \ne \pi_\text{old}$ (the sampling distribution), needs importance sampling correction
- $\nabla \mathbb{E}_{\pi_\theta}[A] = \mathbb{E}_{\pi_\text{old}}[(\pi_\theta/\pi_\text{old}) \nabla\log\pi_\theta A]$
- Clip prevents the ratio from drifting too far across multiple updates

Pitfalls: not knowing about PPO's multiple updates; or unclear on the role of IS.

</details>

<details>

<summary>Q13. DPO vs IPO differences? When to use IPO?</summary>

- DPO uses sigmoid loss: the more extreme the preference, the larger the implicit reward (unbounded)
- IPO uses squared loss + fixed margin $1/(2\beta)$: reward is bounded
- **When preference data is deterministic** (every pair $y_w$ always wins), DPO easily overfits, IPO is more stable
- Azar et al. 2024 AISTATS gives a unified framework ($\Psi$PO)

Pitfalls: saying IPO is just a hyperparameter of DPO; or not knowing the sigmoid issue under deterministic preferences.

</details>

<details>

<summary>Q14. What did SimPO change compared to DPO?</summary>

- **Drop reference**: $r = (\beta / |y|) \log\pi(y)$, no $\pi_\text{ref}$ needed
- **Length normalize**: divide by $|y|$, mitigates DPO's length bias
- **Reward margin** $\gamma$: require $r_w - r_l \ge \gamma$, loss stops once exceeded
- Saves memory by one model; but losing the KL anchor requires more careful tuning of $\beta, \gamma$

Pitfalls: only saying "drop reference," missing length-norm and margin; or not knowing SimPO is also contrastive.

</details>

<details>

<summary>Q15. What is the GRPO advantage formula? Why this design?</summary>

- $\hat{A}_i = (r_i - \text{mean}_g(r)) / (\text{std}_g(r) + \epsilon)$, where $g$ is the group of the same prompt
- Within the response, all tokens share the same $\hat{A}_i$
- **Design rationale**: LLM token-level value is hard to learn; use sequence-level reward + within-group statistics to directly do variance reduction
- The philosophy aligns with RLOO (leave-one-out mean) and ReMax (greedy baseline): use sample baseline to replace critic

Pitfalls: writing per-token advantage (wrong, GRPO shares within sequence); or not knowing the commonality of GRPO/RLOO/ReMax.

</details>

<details>

<summary>Q16. PRM vs ORM differences? When to use PRM?</summary>

- ORM scores only the final answer; PRM scores each reasoning step
- PRM is notably better on **multi-step reasoning tasks** (math, code) (Lightman 2023, MATH dataset 78% vs 72%)
- PRM annotation cost is high → Math-Shepherd uses rollout-based auto-labeling
- PRM can be used for both search rerank and step-level RL (dense reward)

Pitfalls: saying PRM is better at all tasks (wrong, simple tasks have ORM sufficient); or not knowing Math-Shepherd's auto-labeling.

</details>

<details>

<summary>Q17. Key idea of Constitutional AI / RLAIF?</summary>

- Use AI (not humans) for preference annotation, saving manual labor
- Constitutional AI (Bai 2022 Anthropic): SL stage AI self-rewrites harmful → RL stage AI preference scoring → PPO
- RLAIF (Lee 2023 Google): systematically validated AI ≈ human on tasks like summarization
- **Risk**: judging LLM biases get amplified; usually mixed with rule-based / human RM

Pitfalls: saying RLAIF is just GPT-4 labeling data (not entirely correct, CAI emphasizes "by constitution" self-evaluation); or not knowing CAI predates RLAIF by a year.

</details>

<details>

<summary>Q18. Why add KL on the reward in PPO training, not on the loss?</summary>

- Adding on the reward → naturally flows into PPO-clip surrogate via advantage, per-token control
- Adding on the loss → overall KL constraint, but lose token-level resolution
- GRPO conversely puts KL on the loss (using K3 estimator), since GRPO does not unfold per-token advantage
- Both placements are essentially KL anchors, just different implementation details

Pitfalls: confusing the two placements; or not knowing GRPO's K3 estimator.

</details>

<details>

<summary>Q19. How is the value model initialized in RLHF? Why?</summary>

- Usually **initialized from RM or SFT model** for the value backbone, with a new value head added
- Benefit of RM init: RM already "understands reward," value converges faster
- Benefit of SFT init: value shares underlying representations with policy
- DeepSpeed-Chat defaults to RM init; TRL defaults to SFT
- **Shared trunk vs independent model is a trade-off**: A3C/A2C and other classical actor-critic share trunk to save memory, but policy and value loss have different scales that can interfere; LLM RLHF mainstream (DeepSpeed-Chat / TRL / OpenRLHF) chooses independent models + separate optimizer, prioritizing stability

Pitfalls: random initialization of value (impractical, too slow); or saying "value must share trunk" / "must not share," both extremes are wrong — it's a trade-off, not a theorem.

</details>

<details>

<summary>Q20. DPO training broke (margin not increasing / loss not dropping), how to diagnose?</summary>

- Check if `chosen_logp` and `rejected_logp` both drop (typical DPO degeneration)
- Check if reference policy is correctly loaded (forgot to load → log-ratio becomes raw log-prob)
- Check $\beta$: too small (< 0.01) makes loss ≈ sigmoid constant, too large (> 1) easily collapses
- Check data: are $y_w$ and $y_l$ truly significantly different?
- Check length: DPO favors long responses, if $y_w$ is systematically shorter than $y_l$, there's a bug

Pitfalls: only saying "tune lr"; or not knowing the likelihood-decrease-for-both problem.

</details>

### L3 Top Lab (5 questions)

<details>

<summary>Q21. DeepSeek-R1 vs R1-Zero differences? Why need cold-start?</summary>

- **R1-Zero**: directly run GRPO + rule-based reward (math correctness + format) from pretrain base, **no SFT**
  - Pros: emergent long CoT, proves RL can self-trigger reasoning
  - Cons: low readability, language mixing, unstable format
- **R1**: cold-start SFT (thousands of high-quality reasoning) → reasoning RL → SFT → general RL (multi-stage)
  - Fixes R1-Zero's readability issues
  - SOTA math / code reasoning
- **Importance of R1-Zero**: proves RL alone can trigger reasoning without prior SFT; foundation for subsequent self-play / pure-RL paths

Pitfalls: only saying R1 is an improvement of R1-Zero; or not knowing what cold-start fixes (readability, not performance).

</details>

<details>

<summary>Q22. Who is more important in GAE: $\gamma$ or $\lambda$? How are they typically set in LLM RLHF?</summary>

- $\gamma$ is reward discount (task-level), $\lambda$ is TD estimator trace-decay (algorithm-level)
- **In LLM RLHF $\gamma = 1$** (no discount, since reward only at terminal, discount makes early tokens receive too small a signal)
- $\lambda = 0.95$ is still useful: bias-variance trade-off at token dimension
- The joint $\gamma\lambda$ decay is GAE's effective trace-decay; with $\gamma=1, \lambda=0.95$ the actual trace is about 20 tokens
- If $\gamma = 1, \lambda = 1$ → GAE degenerates to MC, equivalent to "reward-to-go - V baseline"

Pitfalls: blindly copying $\gamma = 0.99$ from game RL, not knowing $\gamma = 1$ is more mainstream for LLMs (more reasonable under terminal-reward + short-context generation); or conversely insisting $\gamma$ must be 1 — it depends on whether the reward is dense / long horizon, an engineering choice not a theorem.

</details>

<details>

<summary>Q23. How to design an RL framework that supports DPO / PPO / GRPO simultaneously?</summary>

Abstract into three layers:

1. **Data layer**: preference pair (DPO) / prompt + group (GRPO) / prompt only (PPO) → unified batch interface
2. **Trajectory layer**: on-policy rollout (PPO/GRPO) vs offline (DPO)
   - PPO/GRPO need vLLM/sgl inference acceleration + asynchronous trajectory queue
   - DPO is pure dataloader
3. **Loss layer**: pluggable
   - PPO: clipped surrogate + value + entropy + GAE
   - DPO: log-ratio sigmoid
   - GRPO: clipped surrogate + group-normalized advantage + K3 KL
4. **Reference management**: DPO/PPO/GRPO all need $\pi_\text{ref}$, uniformly encapsulated as frozen + lazy load
5. **RM/rule reward**: pluggable reward backend (neural RM, unit-test, math checker)

Reference: TRL, OpenRLHF, verl (ByteDance), SimplePO. **verl is the mainstream implementation framework for GRPO/RLOO/DAPO**.

Pitfalls: only listing PPO; or not considering trajectory queue and asynchronous rollout.

</details>

<details>

<summary>Q24. What if RM and policy are trained together? Why doesn't mainstream RLHF do this?</summary>

- Intuition: let the RM also learn online, giving it more data from new distributions → adversarial training
- **Issue 1**: RM training objective and policy training objective are coupled, loss landscape is unstable (GAN-like)
- **Issue 2**: RM training needs fresh human labels, otherwise it drifts with the policy
- **Issue 3**: RM signal changes too fast → policy's notion of "what is good" is inconsistent
- Middle-ground: **iterative preference optimization** (Xu et al. 2023 arXiv 2312.16682 *Some things are more CRINGE than others: Iterative Preference Optimization with the Pairwise Cringe Loss*), **self-rewarding LM** (Yuan et al. 2024 *Self-Rewarding Language Models*) — each round, regenerate responses with the latest policy + rescore (same model / human / GPT-4), implicitly updating RM
- **OpenAI / Anthropic mainstream: fixed RM, multi-round PPO**; visually interpretable, more stable

Pitfalls: only saying "would be unstable" without specific reasons; or not knowing the emerging path of iterative DPO / self-rewarding.

</details>

<details>

<summary>Q25. If you were to design a next-gen post-training algorithm, how would you improve GRPO?</summary>

Possible directions (answer 2-3, with trade-off discussion):

- **Adaptive group size**: small $G$ when reward variance is high, large $G$ when low (DAPO dynamic sampling)
- **Token-level credit assignment**: GRPO shares advantage across sequence → signal diluted on long responses. Can introduce a lightweight critic (VAPO) or step-level reward (PRM) based partial credit
- **Off-policy correction**: GRPO is on-policy, rollout is slow; introduce V-trace / Retrace to use stale samples
- **Multi-task reward**: rule + RM + style composed reward, each dimension independently normalized to avoid reward scale imbalance
- **Reward model uncertainty**: use min or mean - std of an ensemble RM to prevent over-optimization
- **Process reward integration**: PRM gives step-level dense advantage, combined with group baseline (Math-Shepherd + GRPO path)
- **CISPO / truncated IS** (Minimax 2025): solves GRPO's stability issue with large ratios on negative advantages
- **DAPO** (ByteDance 2025): clip higher + dynamic sampling + token-level loss + overlong shaping, open-sourced verl implementation

Pitfalls: only listing "add attention / more models" without trade-offs; or not knowing follow-up work like DAPO / VAPO / CISPO.

</details>

## §A Appendix: Reference List

Organized by section, all verified by codex (gpt-5.5 xhigh) reviewer for correct author-year-venue:

**PPO / RL fundamentals**

- Schulman et al. 2017 arXiv 1707.06347 *Proximal Policy Optimization Algorithms*
- Schulman et al. 2016 ICLR *High-Dimensional Continuous Control Using GAE*
- Schulman 2020 blog *Approximating KL Divergence* (source of K3 estimator)

**RLHF**

- Christiano et al. 2017 NeurIPS *Deep Reinforcement Learning from Human Preferences* (first paper on pairwise preference + RM)
- Stiennon et al. 2020 NeurIPS *Learning to Summarize from Human Feedback* (OpenAI summarization report)
- Ouyang et al. 2022 NeurIPS *Training Language Models to Follow Instructions with Human Feedback* (InstructGPT)
- Bai et al. 2022 Anthropic arXiv 2204.05862 *Training a Helpful and Harmless Assistant with RLHF*
- Bai et al. 2022 Anthropic arXiv 2212.08073 *Constitutional AI*
- Lee et al. 2023 Google arXiv 2309.00267 *RLAIF: Scaling RLHF with AI Feedback*

**DPO family**

- Rafailov et al. 2023 NeurIPS *Direct Preference Optimization*
- Azar et al. 2024 AISTATS *A General Theoretical Paradigm to Understand Learning from Human Preferences* (IPO)
- Ethayarajh et al. 2024 ICML *KTO: Model Alignment as Prospect Theoretic Optimization*
- Meng et al. 2024 NeurIPS *SimPO: Simple Preference Optimization with a Reference-Free Reward*
- Hong et al. 2024 EMNLP *ORPO: Monolithic Preference Optimization without Reference Model*
- Tang et al. 2024 ICML *Generalized Preference Optimization* (unifying DPO/IPO/SLiC framework)

**Critic-free RL**

- Ahmadian et al. 2024 ACL *Back to Basics: Revisiting REINFORCE Style Optimization* (RLOO)
- Li et al. 2024 ICML *ReMax: A Simple, Effective, and Efficient Reinforcement Learning Method*
- Shao et al. 2024 arXiv 2402.03300 *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models* (proposes GRPO)
- DeepSeek-AI 2025 arXiv 2501.12948 *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*
- Yu et al. 2025 ByteDance arXiv 2503.14476 *DAPO: An Open-Source LLM Reinforcement Learning System at Scale*

**Reward Modeling**

- Lightman et al. 2024 ICLR / OpenAI arXiv 2305.20050 (2023) *Let's Verify Step by Step* (PRM800K, PRM vs ORM)
- Wang et al. 2024 ACL *Math-Shepherd: Verify and Reinforce LLMs Step-by-Step without Human Annotations*
- Coste et al. 2024 ICLR *Reward Model Ensembles Help Mitigate Overoptimization*
- Eisenstein et al. 2024 COLM (arXiv 2312.09244, 2023) *Helping or Herding? Reward Model Ensembles Mitigate but do not Eliminate Reward Hacking*
- Gao, Schulman, Hilton 2023 ICML *Scaling Laws for Reward Model Overoptimization*

**Iterative / Self-rewarding**

- Xu et al. 2023 arXiv 2312.16682 *Some things are more CRINGE than others: Iterative Preference Optimization with the Pairwise Cringe Loss*
- Yuan et al. 2024 ICML *Self-Rewarding Language Models*
- Pal et al. 2024 arXiv 2402.13228 *Smaug: Fixing Failure Modes of Preference Optimisation with DPO-Positive*

Code frameworks: TRL (HuggingFace), OpenRLHF, verl (ByteDance Seed), Axolotl, LLaMA-Factory, SimplePO. **verl is the current mainstream GRPO/RLOO/DAPO implementation**; most DeepSeek-line work reproduces on verl.
