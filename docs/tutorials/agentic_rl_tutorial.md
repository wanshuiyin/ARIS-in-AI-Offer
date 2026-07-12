## §0 TL;DR Cheat Sheet

> 💡 **9 句话搞定 Agentic RL** — RL for LLM agents 是 2024-2026 把 reasoning RL 推向真实工具使用、Web、代码与 GUI 的核心范式（详见 §1-§9 推导 + §10 25 高频题）。

1. **Agentic RL 与 RLHF 的本质区别**：RLHF 是 single-turn 偏好对齐，reward 来自 RM 对整段 response 的打分；**Agentic RL 是 multi-turn 决策，state 是 (obs, history)，action 是 (thought, tool_call)，更强调用可编程验证的客观结果（test-pass、task success、verifier）做主干，而非 RM 式的主观偏好打分**（但也允许用学出来的 reward model，如 WebRL 的 ORM，二者不是严格二分）。整条轨迹长度从 RLHF 的几百 token 涨到 agent 的数千乃至数万 token，credit assignment 难度上一个台阶。

2. **PPO/GRPO 在 agent 上的关键改造**（必背）：**token mask** 必须只对 agent 自己 generate 的 token 算 loss——observation token（tool 返回的 stdout / search snippet）属于环境，policy gradient 不能流到那里；否则 model 会试图"教 tool 怎么回答"，行为崩坏。GRPO 优势更明显：在 long-horizon trajectory 上，value model 几乎学不动 per-token V（中间几乎全 0 reward），组内归一化是更稳的 baseline。

3. **Reward 设计三层金字塔**：(a) **Outcome reward** 最便宜也最稀疏——final answer/task success 0/1；(b) **Process reward** 给每步打分，需要 PRM 或 step verifier；(c) **Hybrid / shaping**——tool-call shaping（鼓励调对工具）、length penalty（防 agent 拖太长）、format reward（强约束输出 schema）。R1 路线用 rule-based outcome reward（数学正确 + 格式），SWE-RL 用 test-pass（rule-based），WebRL 则训练一个 ORM 给 task success 打分（learned-RM 路线，见 §6.9）——**outcome-level 验证信号（rule-based 或 learned ORM）+ dense format shaping** 是 2025 工业实测最稳的组合。

4. **代表性早期 work**：**AgentTuning** (Zeng et al. 2023 arXiv 2310.12823 THU)——agent SFT 数据集 + 多任务训练；**Agent-FLAN** (Chen et al. 2024 ACL Findings arXiv 2403.12881)——把 agent corpus 拆成 multi-turn / formatted / negative example 三类；**ReFT** (Luong et al. 2024 ACL arXiv 2401.08967)——SFT warm-start + online RL on math reasoning，PPO 在 GSM8K 上 +9pp。这三篇是 Agentic RL 的"先 SFT 后 RL"标准三段式。

5. **Tool-augmented reasoning RL**：**ToolRL** (Qian et al. 2025 arXiv 2504.13958)——把 tool 调用嵌入 GRPO，reward 含 correctness + format + tool-use efficiency；**ReSearch** (Chen et al. 2025 arXiv 2503.19470)——把 search call 当 first-class action，rule-based reward 学 multi-hop search；**RAGEN / StarPO** (Wang et al. 2025 arXiv 2504.20073)——多回合 RL 训练 framework，state-action token level loss + critic-free GRPO 变种。共同点：**outcome-only reward + format shaping + token-mask loss + GRPO**。

6. **Web / GUI agent RL**：**WebRL** (Qi et al. 2024 ICLR-25 arXiv 2411.02337)——self-evolving curriculum + ORM + retrospective rollout，把 8B Llama 推到 WebArena 43%；**AgentQ** (Putta et al. 2024 arXiv 2408.07199)——MCTS 搜索 + AI critique + DPO offline 训练；**Computer-Use** (Anthropic Claude 3.5/3.7/4 Sonnet, 2024-10-22 起)——RLHF + RL 在屏幕截图 + 鼠标键盘 action space 上训练 GUI 控制（公开知识：训练细节未披露，但 system card 说明用了大量人工 + AI 反馈）。

7. **Code agent RL**：**CodeRL** (Le et al. 2022 NeurIPS arXiv 2207.01780) 首次把 unit test 当 reward 信号 + actor-critic；**PPOCoder** (Shojaee et al. 2023 arXiv 2301.13816) 加入 compilable + functional correctness 的 composite reward；**SWE-RL** (Wei et al. 2025 Meta FAIR arXiv 2502.18449) 用 rule-based reward（patch similarity + test-pass）在 GitHub PR 数据上做 RL，Llama-3.3-70B 把 SWE-bench Verified 推到 41%。

8. **Self-rewarding & exploration**：**Self-Rewarding LM** (Yuan et al. 2024 Meta arXiv 2401.10020) 让 policy 同时当 judge，iterative DPO with LLM-as-judge；但 self-rewarding 在 agent 上比单 turn alignment 更危险——judge 也是 agent 自己，**容易 reward drift / model collapse**。生产里多用 LLM-as-judge ensemble + rule-based grounding（test-pass、math checker）+ human spot check 三件套。

9. **长 horizon credit assignment 的"三种武器"**：(a) **GAE + γ < 1** 把信用沿轨迹回传，但在 sparse outcome reward 下退化为 MC return；(b) **Hindsight relabeling**（HER 思路在 agent 上的对应物）——失败轨迹按"中间状态当 goal"重新标 reward；(c) **subgoal decomposition + process reward**——把 50 步轨迹切成 5 个 subgoal × 10 步，PRM 给每个 subgoal 打分。L3 面试常问的"为什么 GRPO 在 long-horizon agent 上比 PPO sample efficient"——答案是 **trace-level reward 直接匹配 trace-level credit**，绕开 value model 在 long-CoT 上几乎学不动的痛点。

## §1 直觉：从 RLHF 到 Agentic RL

### 1.1　把 LLM 从"会写字的策略"升级成"会动手的 agent"

RLHF 把 LLM 训成"会按人类偏好写字"的 policy；但 RLHF policy 在调用 tool / 多轮交互 / 长 horizon 任务上仍然脆弱：

- **single-turn 偏好** 不 directly transfer 到 multi-turn task success
- **RM 学的是"哪种文风讨人喜欢"**，不是"哪种调用顺序能解决问题"
- **整段 response reward**，无法区分"前 100 token 推理对、第 101 token 选错了 tool"

Agentic RL 与 RLHF 的核心区别不在"reward 一定来自 RM 还是一定来自环境"这种严格二分，而在 **credit assignment 的时间尺度**（RLHF 是单条 response 的 immediate reward；Agentic RL 要在数十步的长 trajectory 上做信用分配）与**验证方式**（RLHF 依赖 RM 对文风/偏好的主观打分；Agentic RL 更强调可编程验证的客观结果，如 test pass、math 答对、网页 task 完成——但也允许用学习到的 reward model，如 §6.9 WebRL 的 ORM，只是相比传统 RLHF-RM 更强调可编程验证信号做主干）。这一步让 alignment-style RL 升级为 **decision-making RL**。

### 1.2　Mental model：MDP / POMDP 表述

| 元素 | RLHF (single-turn) | Agentic RL |
|---|---|---|
| State $s_t$ | prompt | $(o_0, a_0, o_1, \dots, o_{t-1}, a_{t-1})$（history） |
| Action $a_t$ | 整段 response | 一步 `(thought, tool_call)` 或 token-level subaction |
| Reward $r_t$ | terminal RM 分 | terminal task success（多数时刻为 0） |
| Horizon $T$ | 1（一段 response） | 10-200 步（agent loop） |
| Trajectory 长度 (token) | $10^2$-$10^3$ | $10^3$-$10^5$ |
| Environment | RM (神经网络) | 真实环境（shell / browser / search / Python） |

> 📝 上表是两者的**典型**配置对比，不是严格二分——Agentic RL 同样可以用学习到的 reward model 打分（如 §6.9 WebRL 的 ORM），只是相比传统 RLHF-RM 更强调用可编程验证信号（test-pass、task success）做主干；真正的分野在 credit assignment 的时间尺度与验证方式，而非"reward 是否来自 RM"本身。

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
      │   (拼回 prompt 给下一步)     │
      └──────────────┬───────────────┘
                     │
                     └─→ 回到 π_θ
```

### 1.3　Agentic RL 与三类 RL 邻居的关系

| 邻居 | 共同点 | 差异 |
|---|---|---|
| **RLHF / DPO** | LLM + KL-anchored RL | Agentic 必须多轮 + tool I/O，reward 来自环境而非 RM |
| **Reasoning RL（R1, R1-Zero）** | rule-based outcome reward + GRPO | R1 只在数学/代码答题，无 tool；Agentic RL 在 tool 调用 + 多步交互上 |
| **经典 robotic RL（VPT, OpenVLA）** | sparse terminal reward + long horizon | LLM agent action space = token sequence；机器人 RL action = 连续控制 |

> 💡 **面试 framing** — 被问到"Agentic RL 是什么"时，**先 disambiguate**。

- (1) 严格定义：multi-turn + tool I/O + outcome reward RL
- (2) 与 RLHF 的边界：RLHF 是 single-turn alignment，Agentic 是 multi-turn decision-making
- (3) 与 reasoning RL 的边界：reasoning RL 只算答题正确，Agentic RL 算 task success on real environment

这三句话能在 30 秒内把面试官的预期 anchor 准。

## §2 PPO / GRPO 在 agent 上的关键改造

### 2.1　Token mask：only loss on agent tokens

**这是 Agentic RL 第一条铁律**。agent trajectory 里的 token 分两类：

- **agent token**：policy $\pi_\theta$ generate 的（thought, action JSON, final answer）
- **environment token**：tool 返回的 observation（search snippet, stdout, screenshot caption）

PPO/GRPO 的 log-prob ratio 与 loss **必须只在 agent token 上算**。如果在 observation token 上也加 loss：

- policy 会试图"教 tool 怎么回答"（毫无意义且会 reward hack）
- gradient 会被大量低信息 observation token 稀释
- KL penalty 也会错误地把 environment text 当成自己的 distribution 估计

实现上是一个 **action_mask: [B, L]** tensor：1 表示 agent 自己 generate 的 token，0 表示 prompt / observation / padding。loss 计算时 `(loss * action_mask).sum() / action_mask.sum()`。

> ⚠️ **常见 bug** — 早期开源实现（包括早期 TRL agent example）漏了 observation mask，导致训练 metric 看起来在涨但 task success 下降——典型 reward hacking on tool output。OpenRLHF / verl / TRL 2024 版本都已修正，自己写 trainer 必须显式加。

### 2.2　Trajectory-level GAE for agent

agent 轨迹长（50-200 步），reward 极稀疏（只在终点）。设：

- $r_t \in \mathbb{R}$：第 $t$ 步 reward（多数 $t$ 上 $r_t = 0$，终点 $r_T = R \in \{0, 1\}$）
- $V_\phi(s_t)$：critic 估计

TD residual：

$$\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$$

GAE：

$$A_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}$$

**LLM agent 中 $\gamma$ 怎么取？** 取决于"step"的定义：

- 如果 step = **单 token**：$\gamma$ 接近 1（token level discount 没意义）
- 如果 step = **一次 thought-action-obs 循环**：$\gamma \in [0.95, 0.99]$ 合理，控制长 horizon 的折扣

**LLM agent 中 GAE 的退化**：sparse terminal reward 下，$\lambda = 1$ + $\gamma = 1$ 等价于 sequence-level MC return 减 baseline。这正是 GRPO 直接做 trace-level reward 的隐含解释。

### 2.3　PPO loss adapted to agent

带 mask 的 PPO-Clip（按整条 trajectory 算，外层是 trajectory 期望，内层 sum over tokens）：

$$\boxed{\;L^{\text{CLIP-agent}}(\theta) = \mathbb{E}_{\tau \sim \pi_\text{old}}\!\left[\frac{\sum_{t=1}^{T} m_t \cdot \min\!\big(\rho_t A_t,\, \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon) A_t\big)}{\sum_{t=1}^{T} m_t}\right]\;}$$

其中 $\tau$ 是 trajectory（包含所有 $T$ 个 token / step），$m_t \in \{0, 1\}$ 是 agent action_mask（agent 自己生成的 token 为 1，observation/system token 为 0），$\rho_t = \pi_\theta(a_t \mid s_t) / \pi_{\theta_\text{old}}(a_t \mid s_t)$ 是 token-level importance ratio。注意外层 expectation 索引是 trajectory $\tau$，内层 sum 索引是 token $t$，不能混淆。

per-token KL penalty（写进 reward）：

$$\tilde{r}_t = m_t \cdot \big(\text{rule}\_\text{reward}_t - \beta \log \tfrac{\pi_\theta(a_t \mid s_t)}{\pi_\text{ref}(a_t \mid s_t)}\big)$$

注意只对 agent token 算 KL；observation token 的 $\pi$ 是没有意义的（它们是环境给的，不是 model sample 出的）。

### 2.4　GRPO for agents：trace-level group-relative advantage

GRPO 在 agent 上更适用，因为：

1. **省 critic**——agent value 难学（长 horizon + sparse reward）
2. **trace-level reward 直接对应 trace-level advantage**——不需要 per-step value
3. **同 prompt 多 rollout 自动 variance reduction**——agent 任务通常 deterministic env，多次 rollout 给出真实 reward 方差

公式（保留 PPO-Clip 结构，advantage 改组内归一化）：

$$\hat{A}_i = \frac{r_i - \text{mean}(\{r_1, \dots, r_G\})}{\text{std}(\{r_1, \dots, r_G\}) + \epsilon}$$

整条 trajectory 的所有 **agent token** 共享同一个 $\hat{A}_i$（observation token 仍然 mask 掉）：

$$L^{\text{GRPO-agent}}(\theta) = \mathbb{E}\!\left[\frac{1}{G}\sum_{i=1}^G \frac{1}{\sum_t m_{i,t}}\sum_{t=1}^{T_i} m_{i,t} \cdot \Big(\min(\rho_{i,t} \hat{A}_i, \text{clip}(\rho_{i,t}, 1-\epsilon, 1+\epsilon) \hat{A}_i) - \beta\, \text{KL}_{i,t}\Big)\right]$$

KL 通常用 K3 estimator（Schulman 2020 blog）：$\text{KL}_{i,t} = \exp(\log\pi_\text{ref} - \log\pi_\theta) - (\log\pi_\text{ref} - \log\pi_\theta) - 1$。

> ✅ **GRPO-agent 的"四省"** — 列举如下。

- 省 value model（一份显存）
- 省 per-token credit assignment（trace-level advantage）
- 省 reward shaping（rule-based terminal 就够）
- 省 hyperparameter（$c_v, \lambda$ 都不需要）

> ⚠️ **GRPO-agent 的"三痛"** — 仍有三个 trade-off 需要承认。

- 长 trajectory 上 trace-level advantage 太粗（所有 agent token 共享同一 $\hat{A}$）——长 trajectory 的 credit dilution
- 全 success / 全 fail group $\text{std} = 0$ 退化（agent 任务 reward 二值，常出现）
- on-policy rollout 慢（agent rollout 含 tool I/O 延迟，远比 chat completion 慢）

## §3 Reward design for agents（核心）

reward 是 Agentic RL 的命门。**Reward 错了，模型再大、算法再新也学不到东西**；reward 对了，简单 GRPO 就能上 SOTA。

### 3.1　Outcome reward vs Process reward

| 维度 | **Outcome reward** | **Process reward** |
|---|---|---|
| 监督粒度 | trajectory 终点 1 个 reward | 每步 (or 每 subgoal) 1 个 reward |
| Label 来源 | task success（test pass / answer match）| PRM / step verifier / human |
| 稀疏度 | 极稀疏（多数 step 0） | dense |
| Credit assignment | 难（长 horizon 上 GAE 也难) | 易（每步直接打分） |
| Reward hacking | 较低（rule-based 时） | 较高（PRM 可被 hack） |
| 实施难度 | 易（test-pass / 答案 match） | 难（PRM 训练成本高） |

**面试 take**：reasoning RL（R1）选 outcome reward 因为数学/代码可程序化验证；agent RL 主流也选 outcome reward 因为 agent 任务 ground truth 更明确（task 完成 yes/no）。**process reward 主要用于 reasoning-heavy 任务**（数学 step 标 PRM），在 tool-use agent 上较少。

### 3.2　Verifier-based reward（rule-based）

这是 Agentic RL 最干净的 reward 形式：把 reward 写成 **可执行的 verifier 函数**。

```python
def verifier_reward(trajectory) -> float:
    """
    trajectory: list of (thought, action, observation)
    返回 0 或 1 的 outcome reward
    """
    final_answer = trajectory[-1].final_answer

    # 1. 数学题：精确匹配 ground truth
    if task_type == "math":
        return 1.0 if normalize_math(final_answer) == ground_truth else 0.0

    # 2. 代码题：跑 unit test
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
        return webshop_grader(final_state)    # 由 benchmark 提供
```

**verifier-based reward 的核心优势**：

- 接近 ground truth，**绕开 learned-RM 的 reward hacking 主要 failure mode**
- 可重复（同一 trajectory reward 一致），便于 advantage estimate
- 显存 / 算力开销极小（执行 verifier 比一次 LLM forward 便宜数百倍）

**核心限制**：只能用于"可验证任务"——math、code、formal verification、可 grader 化的 web/GUI task。开放式任务（写作、对话）仍需 RM。

### 3.3　Format reward / shaping reward

仅有 outcome reward 时 agent 经常学到"格式崩坏但偶然答对"的 trajectory——例如不写 `<think>` 块就直接 `Action: answer(42)`。**Format reward** 给一个轻量的格式约束信号：

```python
def format_reward(trajectory) -> float:
    """
    检查 trajectory 是否符合预期格式 schema
    返回 [0, 1] 的连续 score
    """
    score = 0.0

    # 必须有 <think>...</think> 块
    if "<think>" in trajectory.text and "</think>" in trajectory.text:
        score += 0.3

    # tool call 必须是合法 JSON
    for action in trajectory.actions:
        if is_valid_json(action.tool_call):
            score += 0.1
        else:
            score -= 0.2   # 严重错误

    # final answer 必须用 \boxed{...} 包裹（math task）
    if has_boxed_answer(trajectory.final):
        score += 0.2

    return max(0.0, min(1.0, score))
```

**Composite reward** 的典型写法：

$$r_\text{total} = \alpha \cdot r_\text{outcome} + \beta \cdot r_\text{format} + \gamma \cdot r_\text{shaping}$$

R1 / R1-Zero 用 `accuracy_reward + format_reward` 的简单加和；ToolRL / RAGEN 等加 tool-call efficiency shaping。

### 3.4　Length penalty（防 agent 拖太长）

agent RL 的一个 emergent failure mode：**model 学到"拖长 trajectory 拿到正确答案的概率更高"**——明明能 5 步解决的任务，agent 偏要走 50 步。这是 reward hacking 的一种形式。

缓解：

$$r_\text{adjusted} = r_\text{outcome} - \lambda \cdot \max(0, T - T_\text{target})$$

或更柔和的 sigmoid 形式：

$$r_\text{adjusted} = r_\text{outcome} \cdot \sigma\!\big(-(T - T_\text{target}) / \tau\big)$$

DAPO 报告 "overlong reward shaping"——生成长度进入 $(L_\text{max}-L_\text{cache}, L_\text{max}]$ 缓冲区后，reward 按分段线性 ramp 从 0 降到 -1（超过 $L_\text{max}$ 则固定 -1），而非指数衰减，用平滑惩罚替代硬截断带来的训练不稳定。

### 3.5　Tool-call shaping reward

给"调对工具"奖励、给"调错工具 / 重复调用"惩罚：

```python
def tool_shaping(trajectory) -> float:
    score = 0.0

    # 调对工具（task 关联性 heuristic）
    if task_needs_search and any(a.tool == "search" for a in trajectory.actions):
        score += 0.1

    # 惩罚连续重复同样调用
    consecutive_dup = count_consecutive_duplicate_calls(trajectory.actions)
    score -= 0.05 * consecutive_dup

    # 惩罚调用不存在的 tool
    invalid_calls = sum(1 for a in trajectory.actions if a.tool not in TOOL_REGISTRY)
    score -= 0.3 * invalid_calls

    return score
```

> ⚠️ **shaping reward 的风险** — shaping 给得不当，agent 会 over-fit shaping signal 而忽略 outcome。**主流做法**：shaping reward weight ≪ outcome reward weight（典型 0.1 : 1），且 shaping 必须 capped（不能无限累加）。

### 3.6　RLAIF for agents：用 LLM 当 reward

外部 verifier 难写时（开放式任务），用强 LLM 当 judge：

```python
def llm_judge_reward(trajectory, judge_model) -> float:
    """
    用 judge LLM 给 trajectory 打分
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

**风险**：

- judge LLM 自己有偏见（长度偏好、谄媚）→ 偏见放大到 student
- judge 自己可能被 prompt-injected（agent trajectory 含恶意指令）
- 算力开销高（每个 trajectory 一次 judge call）

主流缓解：(a) **judge ensemble**（3-5 个不同 model judge 取多数）；(b) **judge with rubric**（强约束输出结构）；(c) **rule + LLM 混合**（可验证部分 rule，开放部分 LLM）。

## §4 Long-horizon credit assignment

### 4.1　稀疏 reward 是 agent RL 的核心痛点

经典 game RL（Atari、Mujoco）reward dense；LLM agent 在长 trajectory 上**几乎所有 step reward = 0，只在终点有信号**。这导致：

- value model 几乎学不到东西（中间 V 应该是什么？）
- per-token policy gradient 方差极大
- 早期 step 与 reward 的因果链被稀释——"我第 3 步选了 search tool" 是不是导致了第 50 步答对？

### 4.2　Discount + GAE 在 agent 上的退化

回顾 GAE：

$$A_t = \sum_l (\gamma\lambda)^l \delta_{t+l}, \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

在 sparse terminal reward 下 ($r_t = 0$ for $t \lt T$, $r_T = R$)，把 $\delta_{t+l}$ 展开代入并按 $l$ 求和后：

$$A_t = (\gamma\lambda)^{T-t} R - V(s_t) + \gamma(1-\lambda)\sum_{l\ge0}(\gamma\lambda)^l V(s_{t+l+1})$$

只有当 $\lambda=1$ 时，末项才严格望远镜化抵消，退化为 $A_t = \gamma^{T-t} R - V(s_t)$（纯折扣 MC return 减 baseline，即 **advantage ≈ 折扣回报 - baseline**）；一般 $\lambda<1$ 时中间会残留一串与 $\lambda$ 相关的 value 修正项。如果 $V_\phi$ 学不准（在 agent 上经常），$\lambda=1$ 这个极限情形就退化为 raw MC return；GAE 的 bias-variance trade-off 失效。

**实践 take**：agent RL 上 GAE 不如 group-relative advantage（GRPO）稳，这是 GRPO 比 PPO 在 agent 上 sample efficient 的根本原因之一。

### 4.3　Hindsight relabeling for agents

Hindsight Experience Replay (Andrychowicz et al. 2017 NeurIPS) 起源于 robot manipulation：失败 trajectory 不丢，而是**把"实际达成的状态"当成 goal**重新标 reward。

LLM agent 版本：

```python
def hindsight_relabel(trajectory):
    """
    把 failed trajectory 改造成 "alternative task" 的 successful trajectory
    """
    if trajectory.outcome == 1:
        return [trajectory]   # 成功的不动

    # 假设 agent 在 web 上 navigate 时本来想买商品 A，但最后停在商品 B 页面
    # → 改成"找到商品 B 的 trajectory"，reward = 1
    alt_task = describe_terminal_state(trajectory.final_state)
    relabeled = trajectory.with_task(alt_task)
    relabeled.outcome = 1
    return [trajectory, relabeled]
```

> 💡 **Agentic Hindsight 的难点** — 需要"任意 terminal state 都能被描述成一个合理 task"。对开放 web 环境（购物、导航）容易；对 math/code 任务很难（错答案不能被改成"另一个题的对答案"）。

### 4.4　Subgoal decomposition + process reward

长 trajectory 切成 subgoal 是另一条 credit assignment 路线：

- 把 100 步 trajectory 切成 5 个 subgoal × 20 步
- 每个 subgoal 终点给一个 process reward（subgoal 是否完成）
- subgoal reward 累加成 trajectory reward

实现方式：

- **hand-crafted subgoal**：人写每个 subgoal 的判据（如 "找到购物车页面" 触发 subgoal-1 reward）
- **LLM-decomposed subgoal**：让一个 planner LLM 把 task 拆 subgoal，verifier 判每个 subgoal
- **PRM-style step reward**：训一个 PRM 评每步好坏（Math-Shepherd 思路）

> ⚠️ **subgoal RL 的代价** — subgoal boundary 错画会导致 agent 学到"刻意触发 subgoal reward 而不真正完成 task"。这是 process reward 通病。

### 4.5　Per-step KL penalty 防止 policy collapse

agent 长 trajectory 上 policy 容易"全押"（每步生成 high-confidence token），导致 entropy 崩坏：

$$\tilde{r}_t = m_t \cdot \big(r_t - \beta \cdot \text{KL}(\pi_\theta(\cdot \mid s_t) \| \pi_\text{ref}(\cdot \mid s_t))\big)$$

**关键**：KL 必须 per-step + 只对 agent token 算。一旦 KL 算到 observation token 上，policy 会被"惩罚 mimicking 环境文本"——但这惩罚没有意义（policy 不该 mimick 环境，只是用 observation 做条件）。

R1-Zero 用 $\beta = 0.001$，agent task 上常用 $\beta \in [0.001, 0.05]$。

## §5 Self-rewarding & exploration in agent RL

### 5.1　Self-Rewarding LM（Yuan et al. 2024 Meta arXiv 2401.10020）

核心 idea：让 policy 自己当 judge，iterative DPO：

```

  Iteration k:
    1. policy_k 生成 multiple responses per prompt
    2. policy_k 自己 LLM-as-judge 打分（也是 policy_k 的另一个 prompt）
    3. 高分 vs 低分构成 preference pair
    4. policy_{k+1} = DPO(policy_k, preference_pair)
```

效果：在 AlpacaEval 上 iterative 自打分能持续涨点。

### 5.2　Self-Rewarding 在 agent 上的危险

agent 任务 vs alignment 任务最大差异：alignment 有"客观偏好分布"（人觉得有用、礼貌），可以 LLM judge；**agent 任务有客观 ground truth（test pass / task success）**——self-rewarding 会:

- judge 自己可能错（agent 答错了但自评对）→ training collapse
- iterative drift：每轮把"自己以为对"的轨迹强化，离 ground truth 越走越远
- 探索退化：自评高分 prefer 已知 pattern，反而抑制探索新工具

**主流做法**：agent RL **优先用 rule-based ground truth**，self-rewarding 仅作为辅助信号（如开放式任务 fallback）。

### 5.3　Exploration in agent RL

agent action space 包括：

- **token-level exploration**：sampling temperature，控制 token 选择多样性
- **tool-call-level exploration**：每个 thought-action 周期选择不同 tool / 不同 query
- **trajectory-level exploration**：完全不同 trajectory plan

经典做法：

| 方法 | 实现 |
|---|---|
| Temperature schedule | rollout 时 $T \in [0.7, 1.2]$，训练随轮次降温 |
| Top-p / Top-k | 限制 sampling 范围避免 outlier token |
| ε-tool-choice | 以 $\epsilon$ 概率随机选 tool（替代 LLM-policy 选择） |
| Diverse beam | 多 trajectory 用 diverse beam search 保证多样性 |
| GRPO group sampling | 同 prompt $G = 16$ 个 rollout，天然 exploration |

**RAGEN / StarPO 的关键 insight**（Wang et al. 2025 arXiv 2504.20073）：multi-turn agent RL 中 **rollout 多样性是 collapse 的"防火墙"**——单 trajectory 训练会让 policy 退化为 deterministic mode。

### 5.4　Curriculum & difficulty scheduling

agent 任务从易到难排序，按 model 当前能力 schedule：

- **WebRL** (Qi et al. 2024 ICLR-25 arXiv 2411.02337) 用 self-evolving curriculum——失败任务被记录，下轮加入 buffer
- **Absolute Zero** / R-Zero 用 learnability reward：选 model success rate ≈ 50% 的任务（最有学习信号）

> 💡 **Curriculum 是"长 horizon agent RL 的隐性 component"** — 不是 algorithm contribution，但**实测对 sample efficiency 比换 algorithm 影响更大**。R1 / R1-Zero 的 reasoning RL 也用了 implicit curriculum（数据难度逐渐升级）。

## §6 Specific algorithms（代表性 Agentic RL papers）

按 "时间 + 数据/任务类型" 排序，每个给一句话 + 关键公式。

### 6.1　VPT (Baker et al. 2022 NeurIPS OpenAI, arXiv 2206.11795)

**Setting**: Minecraft，先用 70k hours youtube 视频 pretrain inverse dynamics model (IDM)，再用 IDM 自动标 action label → behavior clone → RL fine-tune.

**Key**: 第一个大规模"视频 → action label → policy → RL"的 pipeline。Agent RL 的早期蓝本，证明了 **scaling RL with imitation pretrain** 可行。

### 6.2　AgentTuning (Zeng et al. 2023 THU arXiv 2310.12823)

**Setting**: 构建 AgentInstruct dataset（6 个 agent task 的 demonstration），多任务 SFT。

**Key**: 不是 RL，是 **agent SFT**——但这是 Agentic RL 的标准 warm-start 步骤。Llama-2 经 AgentTuning 后 agent task 平均 +50%。

> 💡 **AgentTuning 的位置** — 在 Agentic RL pipeline 里，AgentTuning-style SFT 是 RL 之前的必要 warm-start。直接 from-scratch RL 跑 agent 极其困难，因为 base model 不知道怎么 emit 合法 tool call schema。

### 6.3　Agent-FLAN (Chen et al. 2024 ACL Findings, arXiv 2403.12881)

**Setting**: Agent SFT 数据分三类——multi-turn dialogue / formatted tool call / negative examples（拒绝/失败案例）。

**Key**: **negative example 显著缓解 hallucinated tool call**——SFT 不只看"怎么用对"，还看"为什么这样不对"。是 Agent SFT 工程化的重要 milestone。

### 6.4　ReFT (Luong et al. 2024 ACL arXiv 2401.08967)

**Setting**: math reasoning agent，先 SFT warm-start，再 PPO with rule-based outcome reward (answer correctness)。

**Key formula**（标准 PPO + verifier）：

$$r(\tau) = \mathbb{1}[\text{answer}(\tau) = y^*] - \beta \cdot \text{KL}(\pi_\theta \| \pi_\text{SFT})$$

**Result**: GSM8K +9pp over SFT，MathQA +7pp。证明 PPO + outcome reward 在 reasoning agent 上稳定可行——这是 R1 的前身实验。

### 6.5　DeepSeek-R1 (DeepSeek-AI 2025 arXiv 2501.12948) 在 agent 上的扩展

R1 本身不是 agent paper，但 R1 的 GRPO + rule-based reward + format reward 方法论被 ToolRL / ReSearch / RAGEN 直接继承。**R1 = Agentic RL 的算法基线模板**。

复习 GRPO 在 agent 上的应用：

- per-prompt $G$ rollouts
- trace-level reward（rule-based）
- group-relative advantage
- per-step KL with K3 estimator
- agent token mask

### 6.6　ToolRL (Qian et al. 2025 arXiv 2504.13958)

**Setting**: tool-augmented LLM 上做 GRPO，reward = correctness + format + tool-use efficiency。

**Key formula**:

$$r(\tau) = r_\text{correct} + \alpha \cdot r_\text{format} + \gamma \cdot r_\text{tool-eff}$$

其中 $r_\text{tool-eff}$ 惩罚冗余/无效 tool call。

**Result**: 在 BFCL (Berkeley Function Calling Leaderboard) 上 7B model 接近 GPT-4 性能。开源验证 GRPO + tool shaping 的稳定性。

### 6.7　ReSearch (Chen et al. 2025 arXiv 2503.19470)

**Setting**: search-augmented agent，把 search 当 first-class action，reward = answer correctness only（rule-based）。

**Key idea**: 不需要 process reward 也能学会 multi-hop search，**outcome-only + GRPO 足够**——前提是 base model 经 SFT warm-start 已能 emit 合法 search query。

### 6.8　RAGEN / StarPO (Wang et al. 2025 arXiv 2504.20073)

**Paper**: "Understanding Self-Evolution in LLM Agents via Multi-Turn Reinforcement Learning".

**Setting**: multi-turn RL agent training framework，state-action token-level loss + critic-free。

**Key contributions**:

1. **StarPO**（**S**tate-**T**hinking-**A**ctions-**R**eward Policy Optimization）：critic-free，整段 trajectory 共享 advantage，加 token mask 严格只在 agent token 上算 loss。**StarPO-S** 变种引入 fine-grained reasoning-aware reward + 可选 critic incorporation，进一步缓解多轮 reward sparsity（论文摘要原话）
2. **rollout 多样性 = collapse 防火墙**：实测 group size $G = 16$ 比 $G = 4$ 显著更稳
3. **trajectory length 信号**：失败 trajectory length 大时 reward shaping 应加 length penalty

### 6.9　WebRL (Qi et al. 2024 ICLR-25 arXiv 2411.02337)

**Setting**: Web agent (WebArena)，self-evolving curriculum + ORM + retrospective rollout。

**Key components**:

- ORM (Outcome Reward Model) 训自 task success → 在线给 reward
- 失败 task 进入 curriculum buffer，下一轮加大权重
- retrospective rollout：失败 trajectory 用 LLM 改造成 "正确 trajectory" 重新 SFT

**Result**: Llama-3.1-8B 在 WebArena 上 43%（vs GPT-4 14.4%），证明小 model + 好 RL pipeline > 大 model + zero-shot。

### 6.10　AgentQ (Putta et al. 2024 arXiv 2408.07199)

**Setting**: web agent，MCTS + AI critique + offline DPO。

**Key idea**: MCTS 搜出"reward-balanced" preference pair（高分 trajectory vs 低分 trajectory），用 DPO offline 训练。不需要 online RL infra，对算力受限场景实用。

### 6.11　WebGUM (Furuta et al. 2024 ICLR arXiv 2305.11854)

**Setting**: HTML + screenshot multimodal web agent，offline SFT from demonstrations（**不是 RL fine-tune** — 论文是 imitation/supervised 范式）。

**Key**: 把网页 DOM + 截图同时喂给 model，相比纯 text 提升 grounding 准确率。是 Computer-Use / web agent RL（如 WebRL, AgentQ）的 dataset + base 阶段。后续 web agent RL 工作常在 WebGUM 这类 base 上做 RL fine-tune。

### 6.12　CodeRL (Le et al. 2022 NeurIPS arXiv 2207.01780)

**Setting**: 代码生成 + actor-critic，reward = unit test pass。

**Key formula**: 经典 actor-critic + critic 用作 token-level baseline，PG 信号来自最终 test-pass。

### 6.13　PPOCoder (Shojaee et al. 2023 arXiv 2301.13816)

**Setting**: code generation + PPO，composite reward = compilable + functional correctness。

**Key**: 早期把 PPO 用在 code-gen 上的尝试。证明 multi-component reward 比单 test-pass 训练更稳。

### 6.14　SWE-RL (Wei et al. 2025 Meta FAIR arXiv 2502.18449)

**Setting**: SWE-bench / GitHub PR 数据，rule-based reward = patch similarity + test-pass，GRPO。

**Key features**:

- 数据规模：Meta 用 GitHub PR commit history 构造 76M+ context-issue-patch 三元组（具体规模请核对原论文 Table/Section，本教程未能离线核实 76M+ 是否为最终三元组数，还是原始 PR/commit 事件数）
- reward：edit similarity (oracle patch ↔ predicted patch) + test pass binary
- 算法：纯 GRPO + format reward

**Result**: Llama-3.3-70B + SWE-RL 在 SWE-bench Verified 上 41%（无 scaffold），证明 **rule-based RL on real PR data 可以让 model 学到 emergent reasoning behavior**——例如 file-level retrieval planning、root cause analysis、test self-validation。

### 6.15　OpenVLA (Kim et al. 2024 arXiv 2406.09246)

**Setting**: vision-language-action 7B 模型 for robot manipulation，970K 真实机器人 demonstrations 上 fine-tune (开源 base + LoRA fine-tuning recipe)；论文重点是 imitation learning + parameter-efficient adaptation，**不是 task-specific RL fine-tune**。

**Key**: 是 robot agent 而非 LLM agent；常被作为"LLM agent vs robot agent"对比讨论。后续工作（如 OpenVLA-OFT、π-RL 系列）才在 OpenVLA base 上做 RL，但 OpenVLA 论文本身没做 RL。

### 6.16　Anthropic Computer-Use（公开知识 — 训练细节未披露）

Anthropic Claude 3.5 Sonnet (new) 2024-10-22 起支持 Computer-Use；3.7 / 4.0 / 4.5 / Opus 4.x 持续迭代。

**公开资料只描述了能力 + 安全护栏，训练算法 / reward 形式未披露**：

- action space = 屏幕坐标 + 键盘事件 + 鼠标事件（截图作为 observation）
- 安全性：constitutional AI 风格的护栏 + 红队 + prompt-injection 防御
- system card 提到训练涉及人工演示 + 合成数据，但**没有公开** RLHF / GRPO / verifier reward 等具体细节

社区**推测**（**仅推测**，非官方）：可能用 GRPO/PPO + verifier-based reward + Constitutional AI 风格的 AI feedback，但无任何官方确认。面试时讨论 Computer-Use 训练应明确区分 "公开能力" vs "推测内部细节"。

## §7 Code patterns（PyTorch / 伪代码）

实现 Agentic RL 时最容易写错的几段。每段独立可读。

### 7.1　Agent rollout（state, action, reward 收集）

```python
import torch
from dataclasses import dataclass

@dataclass
class Step:
    obs: str                   # 上一步 observation 或 prompt
    thought_tokens: list[int]  # agent 生成的 thought
    action_tokens: list[int]   # agent 生成的 tool_call JSON
    tool_name: str
    tool_args: dict
    observation: str           # tool 返回结果
    done: bool

def rollout(policy, env, prompt, max_steps=20, max_tokens_per_step=512):
    """
    一条 agent trajectory rollout
    返回: trajectory (list of Step), final reward
    """
    trajectory = []
    history = prompt
    for step_idx in range(max_steps):
        # ── agent 生成 (thought, action) ──
        agent_output = policy.generate(
            prompt=history,
            stop_tokens=["</action>"],
            max_new_tokens=max_tokens_per_step,
            temperature=0.7,
        )
        thought, action_json = parse_thought_action(agent_output)

        # ── 调用 tool ──
        tool_name = action_json["tool"]
        tool_args = action_json["args"]
        if tool_name == "final_answer":
            obs = action_json["answer"]
            done = True
        else:
            obs = env.call(tool_name, tool_args)
            done = False

        # 更新 history（拼回 prompt）
        history = history + agent_output + f"\n<obs>{obs}</obs>\n"

        trajectory.append(Step(
            obs=history,                    # 前置 history
            thought_tokens=tokenize(thought),
            action_tokens=tokenize(action_json),
            tool_name=tool_name,
            tool_args=tool_args,
            observation=obs,
            done=done,
        ))

        if done or step_idx == max_steps - 1:
            break

    # ── 终点 reward ──
    final_reward = env.compute_reward(trajectory)
    return trajectory, final_reward
```

### 7.2　Trajectory-level GAE advantage

```python
import torch

def trajectory_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """
    计算 step-level GAE advantage
    rewards: [T]    per-step reward (多数 = 0, 终点 = R)
    values:  [T+1]  V(s_0)...V(s_T), V(s_T) 应为 0 (terminal)
    dones:   [T]    1 if terminal else 0
    返回: advantages [T], returns [T]
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

### 7.3　PPO loss adapted to agent（with action_mask）

**这是 Agentic RL 最重要的一段代码**。区别 RLHF 的 PPO：必须显式 mask out observation token。

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
    new_log_probs = F.pad(new_log_probs, (1, 0))               # 对齐 [B, L]

    # ── 关键: action_mask ──
    mask = batch["action_mask"].float()
    # 只对 agent 自己生成的 token 算 ratio / loss
    ratio = torch.exp((new_log_probs - batch["old_log_probs"]) * mask)
    # observation 位置: ratio = exp(0) = 1, 不影响 surr1/surr2

    A = batch["advantages"]
    surr1 = ratio * A
    surr2 = torch.clamp(ratio, 1.0 - eps_clip, 1.0 + eps_clip) * A
    # mask 后求均值（避免 observation token 拉低 loss scale）
    policy_loss = -((torch.min(surr1, surr2) * mask).sum() / mask.sum().clamp_min(1.0))

    # value loss: 只在 agent token 上算——这是工程简化，不代表 observation 处的 V 本身没意义（见下方说明）
    V = value(batch["input_ids"]).squeeze(-1)            # [B, L]
    value_loss = (((V - batch["returns"]) ** 2) * mask).sum() / mask.sum().clamp_min(1.0)

    # entropy bonus: 只在 agent token 上
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(-1)               # [B, L-1]
    entropy = F.pad(entropy, (1, 0))
    entropy_bonus = (entropy * mask).sum() / mask.sum().clamp_min(1.0)

    loss = policy_loss + c_v * value_loss - c_e * entropy_bonus

    # 监控
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

> 📝 **critic mask 可以独立于 policy mask 定义** — policy loss 必须排除 observation token（防止 policy gradient 流向环境生成的文本）；但 value 函数在概念上是 $V(s_t)$，其自然锚点落在状态边界（即最后一个 observation token / 下一个 action 开始前），因此该边界位置的 $V$ 是有意义的、正是 §7.2 GAE 里 $V(s_0)\dots V(s_T)$ bootstrap 值的来源。上面代码只在 agent token 上算 value loss 是一种工程简化，读者应理解这一简化与 boundary-based $V$ 定义之间的关系，而不要把它读成"observation 处的 V 没有意义"。

> ⚠️ **agent_mask 的 5 个易错点** —

- 必须 cover **prompt tokens**：prompt 不是 agent generate 的，mask = 0
- 必须 cover **all observation tokens**：tool 返回的每一个 token（甚至包括 `<obs>` 标签本身）mask = 0
- 必须 cover **all padding**：右 pad 的位置 mask = 0
- **separator token**（如 `<action>`, `</thought>`）算 agent token，mask = 1
- multi-turn batch 里**不同 trajectory 的 mask pattern 不同**，必须 per-sample 算

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

    # ── 组内归一化 ──
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
    A = A.unsqueeze(-1)                                          # [N, 1] 整段共享

    # ── log-prob ratio ──
    logits = policy(batch["input_ids"]).logits[:, :-1]
    log_probs = F.log_softmax(logits, dim=-1)
    tgt = batch["input_ids"][:, 1:].unsqueeze(-1)
    new_log_probs = log_probs.gather(-1, tgt).squeeze(-1)
    new_log_probs = F.pad(new_log_probs, (1, 0))                 # [N, L]
    mask = batch["action_mask"].float()

    ratio = torch.exp((new_log_probs - batch["old_log_probs"]) * mask)

    # ── PPO-Clip surrogate (advantage broadcast 到整段) ──
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
    把 outcome / format / shaping reward 合成最终 reward
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

> 💡 **process-then-outcome 复合形式** — 若有 PRM，可以：先 PRM 给 step-level shaping，但**最终轨迹 reward 至少 50% 由 outcome 决定**，避免 agent 偷只关心 PRM score 的局部最优。

### 7.6　Verifier-based reward（code test / math match）

```python
import re
import subprocess

def verifier_reward(trajectory, task_type, ground_truth):
    """
    rule-based outcome reward
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

## §8 Frontier (2024-2026 关键趋势)

### 8.1　Critic-free RL is the new default

DeepSeek-R1 (Jan 2025) 之后，**critic-free RL（GRPO 系）成为开源 agent RL 主流**。原因：

- value model 在长 horizon agent 上几乎学不动
- 一份模型显存省下来可以加大 batch / group size
- 调参简单（不用调 $c_v$, value lr）

verl (ByteDance 2024+)、OpenRLHF、TRL 都已把 GRPO / RLOO / ReMax 当 first-class trainer。

### 8.2　Rule-based reward 在 agent 上的胜利

ReSearch / SWE-RL / RAGEN 是 **outcome reward + format reward，rule-based**；WebRL 则是训练一个 ORM（learned outcome reward model）来打分（见 §6.9），属于 learned-RM 路线，与前三者的纯规则验证不同——四者共同点是都用 **outcome-level（而非逐 token 主观偏好）信号**，而非都是 rule-based，rule-based 分支避开的是 learned RM 的 reward hacking 主 failure mode。

**为什么 rule-based 突然变可行**：

- Agent task 比 alignment task 更"可程序化验证"——test pass / answer match / task complete
- DeepSeek-R1 证明 rule-based 在 LLM RL 上 stable 且 scalable
- learned RM 在 multi-turn 上更容易 hack（轨迹空间大）

### 8.3　Long-horizon training 的 infra challenge

agent RL rollout 慢，因为每个 trajectory 包含 tool I/O 延迟。infra 趋势：

- **vLLM / SGL 异步 rollout**：把 generation 与 training 解耦，rollout 池化
- **Sandboxed execution**：tool execution 在 isolated container，并行化
- **Trajectory queue**：rollout worker / training worker 异步，trajectory 通过 message queue 流转
- **off-policy correction**：rollout 与 update 之间有 lag，用 IS clip 或 V-trace 校正

代表实现：**verl** (ByteDance Seed open-source), **AReaL** (Ant Group + Tsinghua, async RL system arXiv 2505.24298), **OpenRLHF v0.5+**.

### 8.4　Tool-RL on TAU-bench / SWE-bench / OSWorld

工业 benchmark 上的 SOTA 趋势（2025-2026 公开数字）：

| Benchmark | Task | 2024 SOTA | 2025-2026 SOTA | Key approach |
|---|---|---|---|---|
| **TAU-bench (retail)** | customer service multi-turn | ~50% (GPT-4) | 70-80% (Claude 4.x, GPT-5) | RLHF + agent SFT |
| **SWE-bench Verified** | GitHub PR fix | ~25% (Claude 3.5) | **70-80%+** (Claude 4.x, o3) | Agent scaffold + RL |
| **OSWorld** | OS GUI task | ~12% (GPT-4V) | ~50-60% (Claude 4.x, Operator) | Computer-Use RL |
| **WebArena** | web nav | 14.4% (GPT-4) | 43% (WebRL Llama-8B) | curriculum + RL |
| **GAIA** | general assistant | 15% (GPT-4) | 60-70% (Claude 4.x, o3) | Agent + tool RL |

注意：benchmark contamination 风险大，**2026Q1 OpenAI 已弃用 SWE-bench Verified**（原因：contamination + test flaw）；当前更可信 benchmark 是 SWE-Lancer、SWE-bench Multilingual、private holdout。

### 8.5　Anthropic Computer-Use（Claude 3.5 → 4.5 → Opus 4.x）

**公开知识**：

- 2024-10-22 Claude 3.5 Sonnet (new) 首发 Computer-Use beta
- 2025: Claude 3.7 / 4.0 / 4.5 持续迭代，速度 + 准确率提升
- 训练涉及大量人工演示 + AI-generated rollout + RLHF + 安全红队
- Action space = 屏幕截图 + 鼠标 + 键盘 + 文件系统访问
- 公开 system card 提及 Constitutional AI + Computer-Use specific safety filter

**算法层面的 inferred 信息**（学术界推测，未官方确认）:

- 训练大概率涉及 RLHF on screenshot trajectories
- reward 含 task completion grader (LLM-as-judge) + safety classifier
- 可能用 GRPO / RLOO style critic-free RL（公开 paper 多次提及 critic-free）

### 8.6　2025-2026 papers 群像

| 论文 | 方向 | 核心贡献 |
|---|---|---|
| **KodCode** (Xu et al. 2025) | 代码 agent RL | 高质量代码 RL 数据集 + GRPO baseline |
| **DAPO** (Yu et al. 2025 ByteDance) | GRPO 改进 | clip higher / dynamic sampling / token loss / overlong shaping |
| **VAPO** (字节跳动 2025) | GRPO 加 lightweight critic | trace-level credit dilution 缓解 |
| **CISPO** (MiniMax 2025) | importance sampling 改进 | 解决 negative advantage 大 ratio 失稳 |
| **R-Zero** | Self-Play RL | Challenger-Solver self-play，learnability reward |
| **Absolute Zero** | Self-Play RL | 完全无外部 task，code executor 当 verifier |
| **Search-R1** | search agent RL | search 当 first-class action + rule reward |
| **Light-R1** / **Sky-T1** | reasoning + tool RL | 开源 reproduction R1 + agent extension |
| **OpenAgent** / **Llama-Agent** | 数据 + 框架 | 大规模 agent SFT + RL pipeline |

> 💡 **2026Q1-Q2 趋势** — agent RL on real environment（OS, browser, IDE）成为开源主线。**simulator → 真实环境 → 部署 RL** 的 sim-to-real pipeline 是下一个 frontier。Anthropic / OpenAI / DeepSeek / Meta 等都在做但细节未公开。

## §9 Failure modes & 工程经验

### 9.1　Agentic RL 的"七宗罪"

| 失败模式 | 症状 | 根因 | 缓解 |
|---|---|---|---|
| **Token mask 漏 observation** | reward 上不去 / agent 学到怪行为 | gradient 流到 environment token | 严格 per-sample action_mask |
| **Reward hacking on grader** | benchmark 涨但人评下降 | grader 有漏洞 | grader ensemble + holdout test |
| **Length-explosion** | agent 拖到 context 上限 | reward 与 length 正相关 | length penalty + max_steps cap |
| **Tool-call hallucination** | agent 调不存在的 tool | base model SFT 不够 | Agent-FLAN-style negative SFT |
| **Loop / repetition** | agent 反复调同样 tool | exploration 不够 | tool-call diversity bonus + ε-tool |
| **Group $\sigma = 0$ collapse** | advantage = NaN / 0 | 全 success 或全 fail | data filter + std clamp |
| **KL collapse** | policy entropy → 0 | β 太小 | per-step KL + entropy bonus |

### 9.2　Online vs offline RL trade-off

| 维度 | Online RL (PPO/GRPO) | Offline RL (DPO/RFT) |
|---|---|---|
| 数据效率 | 低（每轮新 rollout） | 高（一次 dataset 多次训） |
| 训练速度 | 慢（rollout 含 tool I/O） | 快（pure SFT-style） |
| 性能上限 | 高（持续学习新分布） | 中（受限于 dataset 分布） |
| 实现复杂度 | 高（trajectory queue + verifier server） | 低 |
| 适合 | 长期投入 + 真实环境 | 资源受限 + 已有 demo data |

**实践建议**：从 offline RL（agent SFT + DPO）起手，建立 baseline；有算力后迁移到 online RL（PPO/GRPO + verifier）。

### 9.3　Rollout 优化经验

- **vLLM PagedAttention** 比 HF generate 快 5-10×，是 agent rollout 的必装
- **Tool sandbox** 用 Docker + gVisor，单机并行 64-128 个 trajectory 没问题
- **Async rollout pipeline**：rollout worker 不阻塞 trainer worker
- **Trajectory replay buffer**：FIFO + priority 混合，replay 时按 reward 加权
- **Batch size = group_size × prompt_per_batch**：典型 $G = 16$, prompt = 32 → 512 trajectory / batch

### 9.4　Debug checklist

agent RL 跑崩了，按顺序排查：

1. **看 reward**：是否多数 trajectory reward = 0？rule 写错了？
2. **看 length 分布**：是否大多数 trajectory 在 max_steps 截断？说明 agent 不会 finish
3. **看 action_mask**：是否正确 cover prompt + observation + padding？
4. **看 KL**：approx_kl 是否爆？β 调大；entropy 是否塌？entropy_bonus 调大
5. **看 group_std**：是否大量 group 全 0/全 1？需要 data filter
6. **看 tool call distribution**：是否过度依赖某一个 tool？引入 diversity bonus
7. **看 sample efficiency**：单 prompt 多 rollout reward 方差是否合理？太大说明 base 太弱

## §10 25 高频面试题（L1 必会 / L2 进阶 / L3 顶级 lab）

按难度分 3 档：L1 = 任何 agent / LLM RL 岗会问；L2 = research / alignment 团队会问；L3 = 顶级 lab 硬核题。每题点开看答案要点 + 易踩坑。

### L1 必会题（10 题）

<details>

<summary>Q1. Agentic RL 和 RLHF 的本质区别？</summary>

- **RLHF**: single-turn alignment，state = prompt，action = 整段 response，reward 来自 RM 对偏好的打分，horizon = 1
- **Agentic RL**: multi-turn decision-making，state = (obs, history)，action = (thought, tool_call)，reward 来自外部环境（test pass / task success），horizon = 10-200
- 算法层面都用 PPO/GRPO，但 Agentic 必须加 **action_mask**（只对 agent token 算 loss）
- reward 形式：RLHF 偏好打分（subjective），Agentic 客观 outcome（objective）

把它们当成同一件事；或不知道 action_mask 的必要性。

</details>

<details>

<summary>Q2. 为什么 Agentic RL 必须用 action_mask？</summary>

- agent trajectory 含两类 token：agent 自己 generate 的 + tool 返回的 observation
- 如果不 mask，PPO/GRPO 的 ratio 和 loss 会流到 observation token 上
- 后果：(a) policy 试图"教 tool 怎么回答"，无意义且会 reward hack；(b) gradient 被 low-information observation 稀释；(c) KL penalty 错误地把 environment text 当自己分布估计
- 实现：`action_mask: [B, L]`，1 = agent token，0 = prompt/obs/pad；loss 计算时除以 `mask.sum()` 标准化

说"只在 response 上算 loss"不够具体（agent task 没有"response"这个明确边界）；或漏了 mask 必须 cover all observation tokens。

</details>

<details>

<summary>Q3. GRPO 在 agent 上比 PPO 好的核心原因？</summary>

- **省 critic**：agent value 在长 horizon + sparse reward 下几乎学不动
- **trace-level reward 直接匹配 trace-level credit**：不需要 per-token V，避开 value 学不动的痛
- **组内归一化** 自动做 variance reduction，比 raw advantage 稳
- **省一份显存**：可以扩大 batch / group size
- 限制：长 trajectory 上 advantage 太粗（整段共享），credit dilution 仍存在

只说"省 critic"不全面；或不知道 trace-level reward 与 trace-level credit 的匹配关系。

</details>

<details>

<summary>Q4. Outcome reward 和 process reward 的区别？agent 上常用哪个？</summary>

- **Outcome reward**: trajectory 终点 1 个 reward（test pass / answer match），极稀疏，credit assignment 难，但 reward hacking 风险低
- **Process reward**: 每步打分，dense，credit assignment 易，但 reward hacking 风险高（PRM 可被 hack）
- **Agent 上主流: outcome reward**——agent task ground truth 明确（test/grader）；R1, SWE-RL, ReSearch, RAGEN 都用 outcome-only
- Process reward 主要用于 reasoning-heavy 任务（PRM 在数学 step 上），agent 上少用

以为 process reward 总是好（实际 agent 上 outcome 更稳）；或不知道 reward hacking 风险差异。

</details>

<details>

<summary>Q5. Verifier-based reward 比 learned RM 好在哪？</summary>

- **接近 ground truth**：避开 learned RM 漂出训练分布的主要 failure mode
- **可重复**：同一 trajectory reward 一致（learned RM 输出有噪声）
- **显存便宜**：执行 verifier 比一次 LLM forward 便宜数百倍
- **可解释**：reward 来自客观 rule，可 trace 失败原因
- **限制**：只能用于可验证任务（math/code/grader-able web）

说 verifier-based "完全没有 hacking"——错，仍可能被正则漏洞 / format trick / test 泄露 hack，但比 RM hacking 容易堵。

</details>

<details>

<summary>Q6. R1 / R1-Zero 的方法能直接用在 agent 上吗？</summary>

- 算法可以直接搬：GRPO + rule-based reward + per-step KL + token mask
- 但需要补：
  - **action_mask**: agent 有 observation token，R1 数学任务没有
  - **trajectory rollout infrastructure**: 含 tool I/O，比纯 generation 复杂
  - **format reward 调整**: agent task 的 format 是 JSON tool call，不只是 `<think>`
- 代表性工作 ReSearch / RAGEN / ToolRL / SWE-RL 都是 R1 算法 + 上述改造

说"R1 不能直接搬"——其实可以但要改 wrapper；或不知道 ReSearch / RAGEN / ToolRL 这些工作。

</details>

<details>

<summary>Q7. Agent RL 中为什么需要 SFT warm-start？</summary>

- base model 不知道怎么 emit 合法 tool call schema（JSON 格式 / argument 名称）
- 直接 from-scratch RL 几乎不可能 explore 到合法 tool call → reward 全 0 → 学不动
- SFT warm-start（AgentTuning / Agent-FLAN 数据）让 model "知道动作空间长什么样"
- 之后 RL 在合法 action 子空间内 optimize

只说"RL 慢，SFT 加速"——不够；核心是 action space exploration 困难，SFT 解决"知道动作空间"。

</details>

<details>

<summary>Q8. Length penalty 在 agent RL 中起什么作用？</summary>

- agent 容易学到"拖长 trajectory 拿对答案"的捷径（reward hacking）
- length penalty 给超出 budget 的 trajectory 减分：$r = r_\text{outcome} - \lambda \max(0, T - T_\text{target})$
- DAPO 的 "overlong reward shaping" 是工业级实现（进入 buffer 区间后分段线性 ramp 到 -1，而非指数衰减）
- 限制：penalty 过大会让 agent 不敢探索；要 cap

不知道 length-explosion 是 agent RL 常见 failure mode；或不会写 length penalty 公式。

</details>

<details>

<summary>Q9. Group std = 0 的情况怎么处理？</summary>

- 当 group 内 G 个 rollout reward 全相同（全 success / 全 fail）→ $\sigma = 0$
- 加 $\epsilon$ 时 advantage = 0，policy gradient 项归零，**但 KL 项仍存在**（policy 仍被拉回 reference）
- 不加 $\epsilon$ 时是 NaN
- 实践：
  - **Skip 该 prompt**（data filter）：常见做法，避免无信号更新
  - **Clamp σ 下限**（如 0.1）：只是防止 $0/0$ 产生 NaN 的数值稳定手段——全同组里分子 $r_i - \mu$ 已精确为 0，无论分母 clamp 到多少，advantage 恒为 0，**并不能恢复训练信号**
  - **DAPO dynamic sampling**：丢全对全错 group，是真正解决"无信号"的做法

说一定会 NaN（不对，看实现）；或不知道这种 prompt 暗示 task 过易/过难。

</details>

<details>

<summary>Q10. SWE-RL 是怎么训的？</summary>

- 数据：GitHub PR commit 历史，构造 ~76M context-issue-patch 三元组（Meta 公开数字，具体是否为最终三元组数还是原始 PR/commit 事件数，请核对原论文）
- Reward: rule-based = patch similarity (oracle ↔ pred) + test pass binary
- 算法: 纯 GRPO + format reward
- Model: Llama-3.3-70B
- Result: SWE-bench Verified 上 41%（无 scaffold），证明 rule-based RL 让 model 学到 emergent reasoning：file retrieval / root cause / test self-validation

不知道 SWE-RL 的 reward 设计；或以为是 SFT 而非 RL。

</details>

### L2 进阶题（10 题）

<details>

<summary>Q11. 推导 PPO loss 在 agent 上加 action_mask 后的形式。</summary>

1. 标准 PPO-Clip：$L = \mathbb{E}[\min(\rho_t A_t, \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon) A_t)]$
2. agent 有 $m_t \in \{0, 1\}$，1 = agent token，0 = obs/prompt
3. ratio 计算时 $\rho_t = \exp((\log\pi_\theta - \log\pi_\text{old}) \cdot m_t)$，observation 位置 $\rho = e^0 = 1$，不影响 surr1/surr2
4. loss 归一化：$L^\text{agent} = -\sum_t m_t \cdot \min(\rho_t A_t, \text{clip}) / \sum_t m_t$
5. KL 项也只在 agent token：$\text{KL}_\text{total} = \sum_t m_t \cdot \text{KL}_t / \sum_t m_t$

只写公式不解释 mask 的作用；或忘了 normalization 用 mask.sum() 而非 batch size。

</details>

<details>

<summary>Q12. RAGEN / StarPO 的关键贡献？</summary>

- **StarPO**（**S**tate-**T**hinking-**A**ctions-**R**eward Policy Optimization）: critic-free GRPO 变种，整段 trajectory 共享 advantage
- **严格 token mask**：observation 位置 mask = 0，loss 只在 agent token
- **rollout 多样性 = collapse 防火墙**：group_size $G = 16$ 比 $G = 4$ 显著更稳
- **trajectory length 信号**：失败 trajectory length 大时加 length penalty
- 适用：multi-turn agent task（与 single-turn alignment 区分）

只说"GRPO 变种"——不够；要说出 multi-turn 上的 stability 贡献。

</details>

<details>

<summary>Q13. WebRL 的 self-evolving curriculum 是怎么工作的？</summary>

- 初始 task set $\mathcal{T}_0$（small），训 policy 跑 trajectory
- 失败 trajectory 进入 buffer，加到下一轮 curriculum $\mathcal{T}_{k+1}$
- retrospective rollout：失败 trajectory 用 LLM 改造成"正确 trajectory"（hindsight relabel），再做 SFT
- ORM (Outcome Reward Model) 训自 task success → 在线给 RL reward
- Result: Llama-3.1-8B 在 WebArena 上 43%（vs GPT-4 14.4%）

不知道 self-evolving 的循环结构；或把 WebRL 当作纯 SFT。

</details>

<details>

<summary>Q14. 为什么 self-rewarding LM 在 agent 上比在 alignment 上风险更高？</summary>

- alignment 任务有"客观偏好分布"，LLM judge 与人评有较高相关
- agent 任务有**客观 ground truth**（test pass / task success）—— judge 自己可能错（认错对错）
- iterative drift：每轮把"自评对"的轨迹强化 → 远离 ground truth
- 探索退化：自评偏好已知 pattern → 抑制探索新 tool
- 主流做法：agent RL 优先 rule-based ground truth，self-rewarding 仅作开放式任务 fallback

说 self-rewarding 总是危险（alignment 上仍可用）；或不知道 agent 上 ground truth 客观性是关键

</details>

<details>

<summary>Q15. 推 outcome-reward sparsity 对 critic learning 的影响。</summary>

- value $V_\phi(s_t) = \mathbb{E}_\pi[\sum_{l \ge 0} \gamma^l r_{t+l} \mid s_t]$
- sparse terminal reward → $V(s_t) \approx \gamma^{T-t} \cdot P(\text{success} \mid s_t)$
- 中间状态 $s_t$ 的 value 几乎只取决于"未来是否成功"——这是隐含 long-horizon 预测
- **逐步 reward 稀疏（多数 step reward = 0）不等于训练 target 接近 0**：value MSE loss 的 target 是 bootstrapped return，在 $\gamma$ 接近 1（教程默认 $\gamma=0.99$）时中间状态的 target 近似等于终点结果 $P(\text{success} \mid s_t)$，可能长期维持在较高水平；若 $\gamma=1$，对一条已实现的成功轨迹，终点前每一步的 MC target 恒等于终点 reward，完全不趋于 0
- 真正导致 critic 难学的是**这个 target 本身方差大**、且需要长程预测最终成功与否，而不是 target 数值小、gradient 小
- 这也是 GRPO 省 critic 的理论基础：critic 要学的是高方差、长程依赖的量，省了反而省去 noise

只说"critic 学不动"；不会推 $V \approx \gamma^{T-t} P(\text{success})$；或把"逐步 reward 稀疏"和"value target 接近 0"混为一谈——二者是不同的量。

</details>

<details>

<summary>Q16. ToolRL 的 reward 设计有什么 nontrivial 之处？</summary>

- composite: $r = r_\text{correct} + \alpha r_\text{format} + \gamma r_\text{tool-eff}$
- $r_\text{tool-eff}$ 惩罚冗余 tool call（重复调同样工具 / 调用无效 tool）
- 这是典型的 shaping reward：缓解 length-explosion + tool-overuse 两个 failure mode
- weight 平衡：outcome ≫ format > tool-eff，避免 shaping override outcome
- BFCL benchmark 上 7B 接近 GPT-4

只说"加 tool call 奖励"；不知道 shaping reward weight 平衡是关键。

</details>

<details>

<summary>Q17. Hindsight relabeling 在 agent RL 中怎么用？</summary>

- 失败 trajectory 不丢，改造为 "alternative task" 的 successful trajectory
- 例：agent 想买 A 商品但停在 B → 改造为"找到 B 商品"，reward = 1
- 实现：`alt_task = describe(trajectory.final_state)`，relabel reward = 1
- 适用：开放 web 环境、navigation；不适用：math/code（错答案不能改成对答案）
- 起源：HER (Andrychowicz 2017 NeurIPS) for robot manipulation

不知道 HER 起源；或不知道适用边界（开放环境 vs 答题任务）。

</details>

<details>

<summary>Q18. Per-step KL penalty 和 trajectory KL penalty 的区别？</summary>

- **per-step KL**: 每个 agent token 算 KL($\pi_\theta(\cdot \mid s_t) \| \pi_\text{ref}(\cdot \mid s_t)$)；加进 reward 或 loss
- **trajectory KL**: 整段 trajectory 一个 KL；加进 loss
- 二者不是两种本质不同的散度——由链式法则，$\log\pi(\tau) = \sum_t \log\pi(a_t \mid s_t)$，"trajectory KL"在期望意义下就等于逐步 KL 之和；真正的实现差异在于：(a) 是把 KL 逐步加进 reward 做 shaping，还是只在整体 loss 里加一个标量项；(b) 用 sum 还是 mean 聚合；(c) 用什么数值稳定的估计器
- GRPO 用逐 token 的 K3 estimator 再求和（数值稳定）
- agent RL 上 per-step 更主流（trajectory 太长，直接对序列级 log-ratio 求 KL 数值不稳）

混淆两者本质不同（其实链式法则下是同一个量，差异在实现方式）；或不知道 K3 estimator 解决数值问题（K3: $\text{KL} \approx \exp(\Delta) - \Delta - 1$ 非负）。

</details>

<details>

<summary>Q19. Subgoal decomposition + process reward 怎么做？什么时候用？</summary>

- 长 trajectory 切 subgoal：100 步 trajectory → 5 个 subgoal × 20 步
- 每个 subgoal 终点给 process reward（subgoal 是否完成）
- 实现路径：
  - hand-crafted: 人写 subgoal 判据
  - LLM planner: planner LLM 拆 subgoal，verifier 判
  - PRM-style: PRM 评每步
- 适用：long-horizon agent + 可以 hand-craft subgoal 的任务
- 风险：subgoal boundary 错画 → agent 学到"刻意触发 subgoal reward 而不真正完成 task"

说 process reward 总是好——错；要说出风险与限制。

</details>

<details>

<summary>Q20. Online RL vs Offline RL 在 agent 上的 trade-off？</summary>

- **Online RL (PPO/GRPO)**: 数据效率低（每轮新 rollout），但持续学新分布
- **Offline RL (DPO/RFT)**: 数据效率高，但受限于 dataset 分布
- agent rollout 慢（含 tool I/O），online RL 训练吞吐低
- 实践：先 offline 起手（SFT + DPO），再 online refinement
- 代表：AgentQ 是 offline（MCTS + DPO）；WebRL 是 online；SWE-RL 是 online

只说"online 慢"；不知道 agent rollout 含 tool I/O 是主要瓶颈。

</details>

### L3 顶级 lab 题（5 题）

<details>

<summary>Q21. 推导 GRPO advantage 公式 + token mask 的完整 loss，并解释 agent token mask 的两种等价放置方式。</summary>

1. **Group-relative advantage**:
   - rollout group $\{r_1, ..., r_G\}$ per prompt
   - $\mu = \frac{1}{G}\sum r_i$, $\sigma = \sqrt{\frac{1}{G}\sum (r_i - \mu)^2}$
   - $\hat{A}_i = (r_i - \mu) / (\sigma + \epsilon)$

2. **Trajectory-level broadcast**: $\hat{A}_{i,t} = \hat{A}_i$（所有 agent token 共享）

3. **Token-masked ratio + loss**：
   - 朴素 $\rho_{i,t} = \exp(\log\pi_\theta(a_{i,t} \mid s_{i,t}) - \log\pi_\text{old}(a_{i,t} \mid s_{i,t}))$ —— 在 observation 位置也会有数值（model 估算 env text 的概率）
   - **要点**：只要最终 objective / gradient 只覆盖 agent token，mask 放 ratio 内还是 loss 外都**数学等价**：
     - **Inside-ratio**：$\rho_{i,t} = \exp((\log\pi_\theta - \log\pi_\text{old}) \cdot m_{i,t})$ → obs 位置 $\rho=1$，进 clip 后 $\min(\cdot)$ 项 = $A_{i,t}$ 但乘以 $m_{i,t}=0$ 后归零（在 loss 外的 sum 中）
     - **Outside-ratio (mask loss only)**：保留 obs 位 $\rho_{i,t}$ 数值；最终 loss = $-\sum_t m_t \cdot \min(...)$，obs 位贡献 $m_t = 0$ 直接归零
   - **两者梯度都只覆盖 agent token**（mask 是乘法，梯度对 obs 位都是 0）
   - 但实践上 **Inside-ratio 更安全**：避免 obs 位 $\rho$ 数值参与 clip 触发判断或被日志 / 监控（如 mean ratio）误读为异常。生产实现（verl / OpenRLHF）多用 inside-ratio

5. **Full loss**:
   $$L = -\frac{1}{G} \sum_i \frac{1}{\sum_t m_{i,t}} \sum_t m_{i,t} \cdot \Big(\min(\rho_{i,t} \hat{A}_i, \text{clip}(\rho_{i,t}, 1-\epsilon, 1+\epsilon) \hat{A}_i) - \beta \cdot \text{KL}_{i,t}\Big)$$

不会推 step 4 (mask 位置影响 ratio 数值)；或公式背得对但不解释 mask 设计哲学。

</details>

<details>

<summary>Q22. GRPO 在 long-horizon agent 上比 PPO sample efficient 的根本原因？</summary>

**Trace-level reward 与 trace-level credit 的自然对齐**（不止是"省 critic"）：

1. **Sparse terminal reward 下 critic 学不动**：$V(s_t) \approx \gamma^{T-t} P(\text{success})$，gradient 极小；PPO 的 GAE-advantage 受 noisy critic 拖累
2. **GRPO 用组内样本（含自身）的均值/标准差归一化后的 reward 当 advantage**：这是对 sequence-level advantage 的一个有偏但方差更低、且随 $G$ 增大渐近无偏（consistent）的估计——偏差来源包括：(a) 自身计入均值产生的 $(1-1/G)$ 缩放（相比留一法 baseline）；(b) std 归一化引入的与题目难度相关的非线性偏差；(c) PPO clip 本身引入的额外偏差
3. **Group baseline 比 critic baseline 更稳**：同 prompt G rollouts → group mean 自动反映该 prompt 的难度，variance reduction 更精准
4. **PPO clipping + group size 联合限制更新幅度**：避免单 outlier reward 推飞 policy
5. **省 value model 显存** 是 secondary benefit，不是 primary reason
6. **Rule-based outcome reward 难被 RM hacked**：在 agent 上比 learned RM 稳

只说"省 critic"——不够；要说出 sparse reward 下 critic learn 不动是根因。

</details>

<details>

<summary>Q23. 如何设计一个 RL framework 同时支持 reasoning RL（R1）和 Agentic RL（ReSearch / WebRL）？</summary>

抽象出五层：

1. **Data layer**:
   - reasoning: (prompt, ground_truth) tuples
   - agent: (task, env_spec, reward_fn) 三元组
   - 统一为 `Task(prompt, verifier)`，verifier 是 callable

2. **Rollout layer**:
   - reasoning: 直接 generate
   - agent: 含 tool I/O 的 multi-step rollout (vLLM + sandboxed tool executor)
   - 统一为 `Trajectory(tokens, action_mask, reward)` 接口

3. **Reward layer**:
   - reasoning: rule-based (answer match / test pass)
   - agent: composite (outcome + format + tool_eff + length penalty)
   - 统一为 `Reward(traj) -> float`

4. **Loss layer**:
   - PPO with action_mask
   - GRPO with group_id + action_mask
   - DPO with chosen_mask / rejected_mask
   - 通过 `loss_fn(batch, model, ref_model) -> loss` interface

5. **Infra layer**:
   - vLLM rollout pool
   - Sandboxed tool executor (Docker + gVisor)
   - Trajectory replay buffer (FIFO + priority)
   - Async trainer / rollout

代表实现: **verl** (字节) 已支持 reasoning + agent；**OpenRLHF** 部分支持。

只列 PPO 不考虑 agent rollout infra；或不知道 verl / OpenRLHF 的当前支持范围。

</details>

<details>

<summary>Q24. Anthropic Computer-Use 训练方法 — 已知 vs 推测的清晰边界</summary>

**官方公开（system card / blog）**：

- action space = 屏幕截图 (vision observation) + 鼠标 + 键盘 events
- 能力持续迭代 Claude 3.5 (new) 2024-10-22 → 3.7 / 4.0 / 4.5 / Opus 4.x
- 安全机制：constitutional AI 风格的护栏 + 红队 + prompt-injection 防御
- 训练涉及人工演示 + 合成数据（system card 一般性陈述）

**未公开 / 完全保密**：

- 具体 RL 算法（PPO? GRPO? Critic-free? 都没说）
- Reward signal 形式（task completion grader? Pair-wise preference? Safety classifier 权重?）
- Train data 规模 / 来源 / 演示 vs 合成比例
- 是否有专门的 screenshot RM / VLM-as-judge

**社区合理推测**（**仅推测，不要在面试中说成事实**）：

- 可能是 RLHF on screenshot trajectories（pair-wise 偏好 + task success outcome 混合）
- 可能 critic-free（呼应 DeepSeek-R1 GRPO 等开源趋势）
- 可能用 VLM-as-judge for screenshot 理解
- 可能 curriculum 简单 → 复杂

**面试时务必区分 "公开能力" vs "推测内部"**：说"Anthropic 用 GRPO + screenshot RM"是错的（无证据）；说"我推测可能用了 critic-free RL，因为 Anthropic 在其他场景倾向 GRPO/RLHF 风格"才是诚实的表述。这种区分能力是高级面试的加分项。

</details>

<details>

<summary>Q25. 如果让你设计 next-gen Agentic RL 算法，会怎么改进？</summary>

可能方向（任答 3-4 个，每个要有 trade-off 讨论）：

1. **Lightweight critic for long-horizon**: 不用 full-size value model，但用小型 step-level critic 缓解 trace-level credit dilution。VAPO 已尝试。Trade-off: 加显存 vs 缓解 long-trajectory 信号稀释

2. **Hierarchical reward**: subgoal-level reward + outcome reward 组合。Trade-off: 需要 subgoal definition（人工 or planner LLM），boundary 错画风险

3. **Off-policy correction with V-trace / Retrace**: rollout 慢，让 stale samples 也能用。Trade-off: IS bias vs sample efficiency

4. **Trajectory hindsight relabeling + RL**: 失败 trajectory 自动改造为 alternative task 的成功 trajectory，扩 data。Trade-off: 适用 open-ended task，不适用 closed-form answer

5. **Multi-task reward normalization**: 每个 task domain (math/code/web) 独立归一化，避免 reward scale 不平衡

6. **Reward model uncertainty**: 多 RM ensemble，min/mean-std 防 over-optimization。Trade-off: 算力

7. **Async distributed rollout**: rollout 与 train 完全异步，trajectory queue + worker pool。已是 industry default (verl, OpenRLHF v0.5+)

8. **Self-curriculum + adaptive difficulty**: WebRL 思路 + R-Zero 的 learnability reward 结合，自动找 model success rate ~50% 的任务

9. **Multi-objective Pareto optimization**: 不再单一 scalar reward，task success + safety + efficiency 同时优化，输出 Pareto front

只罗列 "加 attention / 加更多模型" 没 trade-off；或不知道 DAPO / VAPO / CISPO 等近期工作；或忽略 infra 层面 (async rollout) 的重要性。

</details>

## §A 附录：参考文献清单

按方向分组，论文经 web 检索 + arXiv 验证作者 / 年份 / 会议。少数 2025-2026 会议归属未定的论文以 arXiv 记。

**Agent SFT / 基础**

- Zeng et al. 2023 arXiv 2310.12823 *AgentTuning: Enabling Generalized Agent Abilities for LLMs* (THU)
- Chen et al. 2024 ACL Findings arXiv 2403.12881 *Agent-FLAN: Designing Data and Methods of Effective Agent Tuning for LLMs*
- Luong et al. 2024 ACL arXiv 2401.08967 *ReFT: Reasoning with Reinforced Fine-Tuning*

**RL on agent / reasoning（基础算法）**

- Schulman et al. 2017 arXiv 1707.06347 *Proximal Policy Optimization Algorithms*
- Schulman et al. 2016 ICLR *High-Dimensional Continuous Control Using GAE*
- Shao et al. 2024 arXiv 2402.03300 *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*（GRPO 提出）
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

**RLHF / DPO 基础（cross-reference）**

- Ouyang et al. 2022 NeurIPS *Training Language Models to Follow Instructions with Human Feedback*
- Rafailov et al. 2023 NeurIPS *Direct Preference Optimization*
- Bai et al. 2022 Anthropic arXiv 2212.08073 *Constitutional AI*
- Lee et al. 2023 Google arXiv 2309.00267 *RLAIF: Scaling RLHF with AI Feedback*

**Reward model / verification**

- Lightman et al. 2024 ICLR arXiv 2305.20050 (OpenAI 2023) *Let's Verify Step by Step* (PRM800K)
- Wang et al. 2024 ACL arXiv 2312.08935 *Math-Shepherd: Verify and Reinforce LLMs Step-by-Step without Human Annotations*
- Coste et al. 2024 ICLR *Reward Model Ensembles Help Mitigate Overoptimization*

**Infrastructure / framework**

- TRL (HuggingFace): https://github.com/huggingface/trl —— 标准 PPO / DPO / GRPO trainer
- OpenRLHF: https://github.com/OpenRLHF/OpenRLHF —— PPO / GRPO / RLOO 工业化实现
- verl (ByteDance): https://github.com/volcengine/verl —— GRPO / DAPO / agent RL 主流框架
- ReaLHF / AReaL (Ant Group + Tsinghua, async RL system, arXiv 2505.24298)

**SOTA benchmarks (2024-2026)**

- Jimenez et al. 2024 ICLR arXiv 2310.06770 *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?*
- Yao et al. 2024 arXiv 2406.12045 *τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains*
- Xie et al. 2024 NeurIPS arXiv 2404.07972 *OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments*
- Zhou et al. 2024 ICLR arXiv 2307.13854 *WebArena: A Realistic Web Environment for Building Autonomous Agents*
- Mialon et al. 2024 ICLR *GAIA: A Benchmark for General AI Assistants*
- Chan et al. 2024 arXiv 2410.07095 *MLE-bench: Evaluating Machine Learning Agents on Machine Learning Engineering* (OpenAI)

**Anthropic Computer-Use（公开知识）**

- Claude 3.5 Sonnet (new) 2024-10-22 Computer Use beta launch (Anthropic blog + system card)
- Claude 3.7 / 4.0 / 4.5 / Opus 4.x system cards (Anthropic 公开)

代码框架建议：

- 起步用 TRL（HF）的 GRPOTrainer + 自写 verifier
- 工业化用 verl（GRPO/RLOO/DAPO 都支持，含 agent rollout）
- 自研 multi-turn 用 OpenRLHF v0.5+ 的 agent example + 加 tool sandbox
