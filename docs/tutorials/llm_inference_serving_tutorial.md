## §0 请求状态机心智模型 + TL;DR Cheat Sheet

**LLM 推理服务不是"跑一次 forward"，而是一台横跨网关、调度器、显存管理器和采样器的状态机。** 面试里被问烂的"prefill 和 decode 有什么区别"，其实只是这台状态机里的两个格子——不先把整条路径立起来，就说不清 TTFT 卡在哪一步、KV OOM 是哪个资源在报警、p99 恶化该从哪里查起。本教程的主线就是这台状态机；量化、KV 容量公式、speculative decoding 的推导已经在其他教程讲过，这里只讲它们在 serving 这条流水线上**插在哪、改变了什么**。

> 💡 **8 条 TL;DR** — 30 秒抓住主线，细节在 §0.1–§0.2 与 §1–§7 展开。

1. **请求路径**：ingress gate → tokenize/validate → prefix lookup → scheduler admission + KV **预留** → queued/runnable → prefill → 首 token（直接来自最后一次 prefill forward 的 logits，不是"再单独跑一次 decode"）→ 逐轮 decode（采样 + streaming）→ finished → KV 回收（§0.1、§1）。
2. **两级准入**：ingress gate 在 tokenize 之前只做粗粒度检查；精确的 KV/token-budget 判断要等 tokenize、prefix lookup 之后才能做，通过也只是**预留**（记账层面），物理分配发生在被某次 tick 选中时（§1.2）。
3. **指标**：$TTFT=t_{\text{first emit}}-t_{\text{arrival}}$、$ITL_j=t_{\text{emit},j}-t_{\text{emit},j-1}$、$TPOT=\frac{t_{\text{last emit}}-t_{\text{first emit}}}{N_{\text{out}}-1}$，每个量先声明系统边界；throughput 是速率、SLO attainment 是比例、goodput 是"满足 SLO 的速率"，三者不能混用（§2）。
4. **典型瓶颈**："prefill compute-bound、decode bandwidth-bound"只是常见工作区间的经验结论，不是定理——batch 变大、量化、超长 context 都会移动瓶颈（§2.3）。
5. **采样**：top-p 是排序后累计概率首次越过阈值的最小前缀（边界 token 必须保留）；min-p 支持集随温度改变；多个过滤器不总能交换顺序；constrained decoding 本质是把非法 token 的 logit 置 $-\infty$ 再照常采样（§3）。
6. **KV**：PagedAttention 是 block table 间接寻址，不是"缺页自动从磁盘加载"；prefix caching 复用**精确 token 前缀**而非语义相似的 prompt，多租户下还要防 cache 侧信道（§4）。
7. **调度**：static batching 的真正限制是空 slot 在整批结束前不能被新请求替换；continuous batching 靠 iteration 级 refill 省空槽，但不保证对所有负载都降延迟；chunked prefill 改善 decode ITL 却可能拉长长 prompt 自己的 TTFT（§5）。
8. **扩展与诊断**：disaggregation 不是无条件加速（收益要过 KV 传输/协调/故障恢复的 break-even）；优化手段不能线性叠加；同一种子不保证逐 token 复现；p99 恶化先拆排队时间和服务时间，再按状态机分阶段打点（§6、§7、§1.6）。

### 0.1　Happy path：一个正常完成的请求

```text
客户端请求
  → ① ingress gate（粗粒度快速拒绝，不需要 tokenize）
  → ② tokenize / validate（此时才知道精确 prompt 长度）
  → ③ prefix lookup（精确 token 前缀匹配，命中部分复用 KV）
  → ④ scheduler admission + KV 预留（记账层面，非物理分配）
  → ⑤ queued / runnable（等待被某次 scheduler tick 选中）
  → ⑥ prefill（未命中区间做一次或多次 forward）
  → ⑦ 首 token 采样 → ⑧ 逐轮 decode（采样管线 §3 + detokenization/streaming）
  → ⑨ FINISHED → ⑩ KV 回收（reclaim）+ 计费/日志
```

旁路：REJECTED 可能发生在①④任一准入检查；PREEMPTED/SWAPPED/RESUMED、BACKPRESSURED、TIMED_OUT/FAILED/CANCELLED 是任何运行阶段都可能进入的分支——完整状态图见 §0.2。图后三条注释：

- **首 token 直接来自最后一次 prefill forward 的 logits/hidden state**，不需要先起一轮"独立"的 decode iteration；prefix 全命中时的例外见 §1.1。
- **KV 预留 ≠ 物理分配**：④ 通过只代表配额被记账预留，物理 block 在被某次 tick 选中时才分配（§1.2、§5.1）。
- **stop string 可能跨 token 边界**，服务器需缓冲一小段未提交文本才能安全吐出（§3.4）。

### 0.2　完整状态机：聚合视图

| 类别 | 状态 | 一句话 |
| --- | --- | --- |
| **准入** | REJECTED / QUEUED / DEGRADED | 两阶段准入决定拒绝、排队（KV 只是**预留**记账）还是降级塞入（§1.2） |
| **执行** | RUNNABLE / PREFILLING / DECODING | 被 tick 选中后预留转为物理分配，KV 随 decode 持续增长（§5.1） |
| **暂停** | PREEMPTED / SWAPPED / RESUMED / BACKPRESSURED | 资源被调度器回收、KV 迁到 host、或客户端消费跟不上；三种 preemption 释放的资源各不相同（§4.4） |
| **终态** | FINISHED / CANCELLED / TIMED_OUT / FAILED | 四条路径共享同一条 reclaim 例程——容量/调度/记账三类资源动作 + 幂等要求，完整清理不变量见 §1.7 |

<details>
<summary>完整状态、guard 与资源持有表（14 态）</summary>

| 状态 | 进入条件（guard） | 持有的资源 | 主要退出转移 |
| --- | --- | --- | --- |
| **REJECTED**（终态） | ingress gate 判定 body/速率/配额超限；或 tokenize/validate 失败；或 scheduler admission 判定 KV/token-budget 即使排队也无法满足 | 无（未曾持有 KV 或调度槽） | 终态，向客户端返回错误 |
| **QUEUED** | 通过 ingress gate + tokenize/validate + prefix lookup + scheduler admission，KV **预留**成功但尚未被某次 tick 选中 | 请求元数据 + 预留的 KV 配额（记账层面，非物理 block）+ 排队 deadline 计时器 | → RUNNABLE（被 tick 选中）/ → TIMED_OUT（超过排队 deadline）/ → CANCELLED（客户端取消）/ → DEGRADED |
| **DEGRADED** | 容量紧张时，调度器选择降低这条请求允许的 best-of-n/最大并发,而不是直接拒绝 | 同 QUEUED，但预留已按降级后的参数重新计算 | → RUNNABLE |
| **RUNNABLE** | 被某次 scheduler tick 选中，本次 tick 有可用 batch slot 和至少覆盖这次工作量的 KV 物理 block | 预留转为已分配的物理 KV block + 本次迭代的 batch slot + token-budget 份额 | → PREFILLING/DECODING（正常执行）/ → PREEMPTED（调度器为更高优先级请求回收资源） |
| **PREFILLING → DECODING** | 分别对应未命中 prompt 区间尚未算完 / 已产出至少一个 token 且未终止 | 同 RUNNABLE，KV 持续增长 | → FINISHED / → BACKPRESSURED / → PREEMPTED / → CANCELLED / → TIMED_OUT / → FAILED |
| **BACKPRESSURED** | 已生成但客户端消费跟不上（慢连接/不读），输出缓冲区达到上限 | 同 DECODING + 一段有界的已生成未发送 token 缓冲区 | → DECODING（客户端恢复消费）/ → PREEMPTED（缓冲区触顶）/ → CANCELLED（最终断连） |
| **PREEMPTED** | 调度器判定需要为其他请求腾出资源；具体动作是**丢弃重算**/**迁移 host（SWAPPED）**/**暂停但保留 KV（descheduled）**三者之一（§4.4，非穷尽分类） | 因动作而异：重算变体释放全部物理 KV；SWAPPED 释放 GPU 显存但持有 host 内存拷贝；descheduled 变体继续持有全部物理 KV block，只释放 batch slot/计算份额 | → RESUMED（被重新调度）/ → CANCELLED/TIMED_OUT（等待期间被取消或超时） |
| **SWAPPED** | PREEMPTED 的迁移子情形，KV 已完成向 host 内存的搬运 | host 内存 footprint（不占 GPU 显存）+ 请求元数据 | → RESUMED（swap-in 完成，重新占用 GPU 显存） |
| **RESUMED** | 调度器决定把 PREEMPTED/SWAPPED 的请求重新纳入某次 tick | 视来源重新分配/搬回物理 KV block + batch slot | → RUNNABLE（正常并入后续 tick） |
| **TIMED_OUT**（终态） | 在 QUEUED/PREEMPTED 等待期间超过约定 deadline | 无（转入 cleanup） | 终态 |
| **CANCELLED**（终态） | 客户端显式取消或断连（心跳/半关闭检测） | 无 | 终态，须**立即**标记为最高优先级 KV 回收对象，不能等到下次 tick 才处理 |
| **FAILED**（终态） | worker 崩溃/未捕获异常 | 无 | 终态，通常需要触发 retry（若幂等） |
| **FINISHED**（终态） | EOS/stop-string/stop-token 序列/`max_new_tokens`/context 上限/约束解码终态之一先触发 | 无 | 终态 |

</details>

## §1 请求状态机：从 HTTP 到 token stream

### 1.1　全景：一个请求真正经历了什么

按 §0 的图展开，几个最容易被面试者省略的细节：

- **准入分两个阶段**：ingress gate 先做粗粒度快速拒绝，精确的 KV/token-budget 判断要等 tokenize、prefix lookup 之后——完整机制（含预留 vs 物理分配、超限三种响应）见 §1.2。
- **prefix lookup 在 prefill 之前**：调度器按 token-ID 精确匹配已有 cache（通常是 hash(prefix token 序列)），命中的部分跳过，只对未命中的后缀区间做 prefill（§4.3）。
- **未命中区间的 prefill 是一次或多次 forward**：没有 chunked prefill 时通常一次算完；chunked prefill 下会被拆成多次迭代内完成（§5.4）。
- **首 token 不一定是独立的一轮 decode**：最后一次 prefill forward 本身就会算出"prompt 最后一个位置"的 logits，采样直接发生在这里——但前提是这次 forward 真的算过这个位置。**如果 prefix cache 全命中**（所有 prompt token 都跳过了 forward），系统手上只有 K/V，没有最终 hidden state/logits，必须额外缓存这份 logits，或者至少重算最后一个 prompt token，否则无从采样首 token——这是"全命中看似最快"场景里容易被忽略的一步。很多人以为"prefill 完了、再单独起一次 decode iteration 才产出第一个 token"，这会让人对 TTFT 的构成产生错误直觉。
- **逐轮 decode**：每轮只对新 token 做 1 次 forward，但 attention 要读取全部历史 KV；本轮采样出的 token 立刻变成下一轮的输入（§3 展开采样细节）。
- **detokenization/streaming**：每产出一个 token 就尝试增量 detokenize 并推给客户端；stop string 跨 token 边界的缓冲机制见 §3.4、§8 [D03]。
- **结束条件**是一整个集合，不是"EOS 就完了"：完整清单见 §3.4，终止/失败/清理协议见 §1.7。

### 1.2　两阶段准入（admission）与 overload handling

请求能不能被接收，不是"有空闲 GPU 就收"，而是分两个阶段联合判断（对应 §0.2 状态表里 REJECTED/QUEUED/DEGRADED 的 guard）：

**阶段一：ingress gate（tokenize 之前）**——只看粗粒度信号：请求体大小上限、速率限制、租户配额是否已耗尽。这一层用最低成本挡掉明显超限的流量，不需要等 tokenizer 跑完。

**阶段二：scheduler admission + KV 预留（tokenize/validate、prefix lookup 之后）**——此时才知道精确 prompt 长度和未命中长度，联合看：

- **当前 KV 容量余量**（已分配 block 数 vs 总 block 数，§4）
- **max batched tokens / 并发序列数上限**（本次迭代的 token 工作量预算，§5.1、§5.5）
- **预测输出长度**（未知输出长度是接收risk 的主要来源——大多数系统只能用启发式/历史统计估计，预测偏差本身就是过载的根源之一）
- **租户配额**（多租户环境下防止单租户耗尽全局资源）

通过阶段二只代表 KV 配额被**预留**（记账层面扣减剩余容量），不代表物理 block 已经分配——物理分配发生在某次 scheduler tick 真正选中这条请求的时候（RUNNABLE，§5.1）。

超限时的三种响应：**立即拒绝**（REJECTED，快速失败，客户端可重试别的副本）、**排队**（QUEUED，占用请求元数据和预留配额但不占物理 GPU 资源，需设置排队 deadline）、**降级**（DEGRADED，如降低这条请求允许的最大并发/best-of-n 数，用更小的预留把它塞进去而不是直接拒绝）。

### 1.3　fairness 与 starvation

调度策略的选择直接决定谁被饿死：

| 策略 | 平均延迟 | 尾延迟 | 长请求 | 短请求 |
| --- | --- | --- | --- | --- |
| 纯 FCFS | 中等 | 可能很差（队首长请求阻塞） | 正常 | 可能被卡住 |
| 短作业优先（需预测长度） | 较优 | 依赖预测准确性 | 可能长期饥饿 | 优先 |
| decode-priority | decode 侧好 | decode ITL 稳 | prefill 被推迟，TTFT 变差 | 受益 |
| prefill-priority | prefill 侧好 | decode ITL 抖动 | TTFT 好 | decode 被打断 |

**没有免费的调度策略**——利用率最大化和公平性/尾延迟通常是要主动权衡的两个目标，不能只汇报"吞吐提升了多少"。

### 1.4　cancellation、client backpressure、disconnect

客户端断开连接或显式取消后，如果服务器不及时感知，会继续为一个**没有人在听**的请求跑 decode——纯粹浪费 GPU 且挤占其他请求的调度/KV 资源。工程上需要：连接层的心跳/半关闭检测、调度器把"已取消"标记为最高优先级的 KV 回收对象、流式响应的背压——客户端消费跟不上时不能无限缓冲已生成但未发送的 token（会话内存增长，进而拖慢其他请求的 streaming 缓冲区），需要一个有界队列 + 背压信号（BACKPRESSURED，§0.2），触顶后要么等客户端追上，要么主动降级/抢占这条请求。完整的终止/失败/清理协议（含 deadline、worker failure、retry/幂等）见 §1.7。

### 1.5　模型加载与 warm-up：冷启动不是稳态 TTFT

副本刚启动时，权重加载（从磁盘/对象存储读取到显存）、CUDA kernel autotuning、CUDA Graph capture、显存池初始化都会造成远高于稳态的首批请求延迟。**这条冷启动延迟不能混进"稳态 TTFT" benchmark**——评测报告里必须分别报告冷启动尾部和稳态分布，否则会把扩容抖动误判成常态问题。完整的评测协议（load sweep、饱和点、p99 样本量等）见 §2.4。

### 1.6　reproducibility 的真实限制

> ⚠️ **同一随机种子 + 同一份权重 ≠ 逐 token 复现** — continuous batching 改变 batch 组成与 kernel 形状（不同 batch size 触发不同 GEMM/attention kernel 路径）；浮点加法不满足结合律，reduce/all-reduce 顺序变了数值就变；多请求共享同一个 RNG 流时，谁先消费流的哪一段取决于调度顺序。多数情况下这是"batching + 并行 + 浮点"组合系统的正常行为，**但不能当成万能挡箭牌**：batch 组成、并行拓扑、kernel 路径都固定了还对不上，就应该怀疑真实实现 bug（未初始化内存、竞态条件），而不是不假思索地归因于"系统固有"。

<details>
<summary>逐 token 位级复现需要同时固定的完整清单</summary>

**batch 组成和到达顺序**、**并行拓扑**（TP/PP/EP degree 与 rank 映射）、**硬件型号**（不同 GPU 架构的归约顺序和 kernel 实现可能不同）、**软件/kernel 版本**（同一硬件上库版本升级也可能换 kernel 实现）、**数值精度**（fp16/bf16/fp32 的舍入行为不同）、**collective 通信算法**（ring/tree all-reduce 顺序不同）、**框架的 determinism 开关**（很多框架默认关闭确定性 kernel 以换取速度）、以及**每请求独立的 RNG 流**——只固定随机种子只是必要条件之一，远不充分。面试版与 MoE 路由的精确条件见 Q17。

</details>

### 1.7　终止、失败与资源清理协议

一个请求可能通过四条不同路径进入终态——**正常完成**（FINISHED）、**客户端取消**（CANCELLED）、**超时**（TIMED_OUT）、**worker 失败**（FAILED）——但无论走哪条路径，都必须收敛到**同一条 reclaim 例程**，否则会出现资源泄漏或重复释放。慢客户端的有界缓冲与背压完整版见 §1.4。

**worker failure 与 retry**：一个 worker 进程崩溃时，它正在服务的所有请求都要标记 FAILED；上游是否重试取决于这条请求是否**幂等**——如果客户端已经收到部分 token（partial stream），简单重试会产生重复前缀，需要要么让客户端去重，要么服务器明确声明"失败后从头重试，任何已发送的 partial 输出作废"。

**清理 = 容量/调度/记账三类资源动作**：**容量**——物理 KV block 按引用计数递减，共享前缀/beam 分支只有最后一个持有者释放时才真正归还 allocator（QUEUED/DEGRADED 阶段就终止的请求只需回滚预留记账，没有物理 block 可释放）；**调度**——释放 batch slot 与 active-sequence 配额；**记账**——恰好产生一条终态计费/日志记录。取消和超时可能在竞态条件下几乎同时触发——重复调用 reclaim 例程必须是**幂等**的，不能释放两次同一个 block（double-free），也不能互相"以为对方会处理"而一次都不释放（leak）。

> 💡 **计费口径**：generated / emitted / committed 三个计数不总相等——生成了 N 个 token 可能只有 M ≤ N 个真正发给客户端（stop string 截断丢弃、speculative decoding 被拒候选）；计费和监控必须显式声明按哪个数字算，否则"生成量"和"计费量"对不上。

如果面试官问"取消一个请求之后系统具体做了什么"，答案应覆盖四条路径共享的清理不变量，而不是只回答"停止 decode"。

## §2 Serving 指标、SLO 与 Prefill/Decode Roofline

### 2.1　精确指标定义（必须先声明系统边界）

$$TTFT = t_{\text{first emit}} - t_{\text{arrival}}, \qquad ITL_j = t_{\text{emit},j} - t_{\text{emit},j-1}$$

$$TPOT = \frac{t_{\text{last emit}} - t_{\text{first emit}}}{N_{\text{out}} - 1}\quad (N_{\text{out}} \ge 2)$$

若 E2E 延迟的终点就定义为"最后一个 token 的 emit 时刻"、且所有时间戳共用同一系统边界，则 $t_{\text{last emit}} - t_{\text{arrival}} = TTFT + \sum_{j=2}^{N_{\text{out}}} ITL_j$ 是**恒等式**而非近似；若报告口径还包含 emit 之后的 flush/连接关闭/网络尾部传输，右侧需显式另加这一项尾项，这时才写成 $\approx$。

每个量都必须先声明**系统边界**：TTFT 是否包含排队时间、tokenization 耗时、网关/网络传输？"平均"还是"p95/p99"？

> ⚠️ **不要把 TPOT 和"总延迟除以 token 数"混为一谈** —— TPOT 是 decode 阶段稳态每 token 耗时（分母排除首 token），如果把排队时间和 prefill 时间也摊进"总延迟/token 数"，会把一个受排队影响的请求错误地报告成"这个模型 decode 很慢"。

### 2.2　throughput、SLO attainment 与 goodput

系统吞吐（tokens/s 或 requests/s）高，不代表用户体验好——如果为了堆吞吐把 batch 塞得很满、TTFT/TPOT 全面恶化，用户侧感知到的是变慢。这里要拆开三个经常被混用的概念：

- **throughput（吞吐）**：单位时间处理的工作量，是一个**速率**——但要先声明维度是哪一种：request throughput（requests/s）、output-token throughput（output tokens/s），还是 total-token throughput（含 prompt token 的 total tokens/s，常用于估算 GPU 时间占用/成本）。三种 throughput 在长 prompt/短输出或短 prompt/长输出的负载下可能给出完全不同的排名。
- **SLO attainment（达标率）**：满足 TTFT/TPOT/deadline SLO 的请求（或 token）**占总数的比例**，是一个无量纲的**比例**——$A = N_{\text{SLO}} / N_{\text{total}}$。
- **goodput（有效吞吐）**：满足同样 SLO 约束的请求（或 token）在单位时间内的**速率**——$G = N_{\text{SLO}} / \Delta t$，单位仍然是 requests/s 或 tokens/s，**不是比例**。把 goodput 说成"满足 SLO 的请求占比"是常见但不准确的说法：达标率和 goodput 是两个维度不同的量，可能达标率很高但因为系统整体很慢导致 goodput 很低，也可能达标率一般但因为系统吞吐极高导致 goodput 数值上仍然可观。

报告系统能力时应该同时给吞吐-延迟曲线（在不同并发/到达率下扫出来，见 §2.4），而不是单点吞吐数字；声明是三种 throughput 里的哪一种、SLO attainment 还是 goodput，同样是必须先声明的"系统边界"的一部分。

### 2.3　Prefill/Decode 的经验区间——不是定理

标准 Transformer 一层的 FLOPs 展开（完整推导见 [`kv_cache_speculative_decoding_tutorial.md`](kv_cache_speculative_decoding_tutorial.md) §3.1，此处不重复）大致是：prefill 阶段 $L$ 个 token 一次算完，权重被 $L$ 次复用，通常有较高算术强度，容易跑满 GPU 算力；decode 阶段每步只算 1 个新 token，却要把整层权重和随历史增长的 KV 都重新读一遍，小 batch 下算术强度低，往往落在 HBM 带宽瓶颈区间。这条"prefill compute-bound、decode bandwidth-bound"是**常见工作区间下的经验结论**，两句限定：decode batch 增大后算术强度上升（同一份权重摊给更多 token），部分线性层可能重新变成 compute-bound，量化会改变这个转折点的 batch size；而超长 context 下 prefill attention 也可能受 IO/HBM/workspace/kernel 实现限制——用了 FlashAttention 不自动意味着整个 prefill 就一定 compute-bound。另外输入/输出长度的作用不对称：输入长度直接决定 prefill 工作量和每步 decode 的 KV 读取量，输出长度不改变初始 prefill 的算术强度，但会增加 decode 轮数、KV 驻留时间与后期 context 长度。

> 🎯 **Roofline 是上界思想实验，不是延迟预测器** — Ridge point $I^{*}=\text{peak FLOP/s}/\text{HBM byte/s}$ 只是拿来跟某个 kernel 自己的算术强度对比、判断它理论上更靠近哪一侧；真实延迟还受 kernel 实现、调度开销、访存模式影响，见 §8 的 [D09]。

FlashAttention 一类工作优化的是**attention kernel 的 IO 效率**（省显存读写）；PagedAttention（§4）管理的是**serving 阶段 KV 的地址空间与分配策略**——两者解决的是不同层面的问题，不要混成同一技术。

再往下的纯 kernel/assembly 级因素——算子融合、CUDA Graph（形状变化需重新 capture）、prefill 宽 GEMM vs decode 窄 GEMM/GEMV（$L_q=1$）、paged attention 按 block table 间接寻址的额外开销——本教程只点名不展开，它们都是"实现细节会显著影响实测数字"的例子。

### 2.4　可信 Benchmark 与容量规划（概览）

把吞吐、goodput、TTFT/TPOT 报成一条可信的数字，比看起来更容易出错；下面是一个不完整但覆盖了最常见坑的检查清单，完整的压测脚本设计不在本教程范围内。

- **open-loop vs. closed-loop**：open-loop 按固定 arrival rate（如 Poisson 到达）持续发请求，closed-loop 固定并发数（完成一个才发下一个）。**两者衡量的是不同的东西**：closed-loop 天然会在系统变慢时自动降低有效到达率，容易把一个已经过载的系统压出"虚假稳定"的延迟数字；open-loop 更贴近真实线上流量，但要求压测客户端本身不能成为瓶颈。
- **coordinated omission**：closed-loop 或简单重试逻辑下，如果某个请求卡住了，压测工具往往会"忘记"继续按原定节奏发送后续请求，导致统计样本系统性地漏掉了最慢的那些情况——测出来的 p99 会比真实值乐观得多。修正方法是按**计划到达时间**而不是**实际发送时间**来计算延迟。
- **真实输入/输出长度分布**：用真实 trace（或至少匹配真实分布的合成分布）而不是固定长度，因为 prefill/decode 的比例、KV 占用、batch 组成都强烈依赖长度分布；固定长度的压测会系统性偏离生产行为。
- **load sweep 与饱和点**：单点并发/到达率测出来的数字没有意义，要扫一条从空载到饱和的曲线，找到吞吐-延迟曲线开始陡峭上升的拐点（饱和点），并明确报告"这个吞吐数字对应哪个延迟"。
- **冷启动分报与 p99 样本量**：冷启动尾部必须与稳态分布分开报告（§1.5），否则扩容抖动会被误判成常态问题；p99 这类尾部分位数在小样本下噪声很大，需要足够样本量并说明置信区间。

replica sizing（goodput-based 副本数计算）的完整输入清单见 Q19。

## §3 Logits 到 Token：采样、约束解码与流式输出

### 3.1　采样管线与速记表

**Logits processors 的执行顺序属于 API 语义的一部分，不同框架并不统一。**本教程固定为下面这条教学管线（仅为讲解方便，不代表任何框架的强制顺序；符号：原始 logits $z_i$，某 token 已出现次数 $c_i$）：

$$\text{硬约束/logit bias} \to \text{repetition/frequency/presence 惩罚} \to \text{温度} \to \text{截断过滤器} \to \text{重新归一化} \to \text{采样}$$

四个最常用算子的速记表（$p_i=\text{softmax}(z_i/T)$，各过滤器均在保留集合内重新归一化）：

| 算子 | 精确定义 | 关键坑 |
| --- | --- | --- |
| **Temperature** | $p_i(T)=\exp(z_i/T)/\sum_j \exp(z_j/T)$，仅 $T>0$ 时定义；logits 有限且无额外 mask 时支持集是全词表 | `T=0` 必须实现为独立的 greedy 分支（$y_t=\arg\max_i z_i$，并列最大值须声明确定性 tie-break，本教程取最低 index），绝不能真的去算 $z_i/0$；$T\to0^+$ 的并列极限见 §A [D01] |
| **Top-k** | 支持集 = logits 最大的 $k$ 个 token（$1\le k\le \vert E\vert$，$E$ 为本步有限 logit 的合法 token 集合） | 正温度缩放不改变排序，单独 top-k 支持集不随 $T>0$ 改变；第 $k$ 位并列时支持集不唯一，须声明 tie-break（本教程按 $(-z_i,\text{token\_id})$ 排序取前 $k$ 个） |
| **Top-p** | 按概率降序，取累计概率**首次 $\ge p_0$ 的最小前缀**，边界 token 必须保留 | "保留所有 $p_i\ge p_0$"与"累计严格 $<p_0$"两种直觉定义都误删边界 token——[D01] 用 $p=[0.4,0.3,0.2,0.1]$、阈值 $0.65$ 钉死：正确支持集是前两个 token（累计 $0.7\ge0.65$），"严格小于"在第一个 token（累计 $0.4$）就误停 |
| **Min-p** | $S=\{i:\ p_i \ge \alpha \max_j p_j\}$，等价于 $z_i \ge z_{\max} + T\log\alpha$ | 支持集**随温度改变**，不是固定 logit 差阈值——$\alpha<1$ 时 $T$ 越大阈值越低、支持集越大（[D01] 扫多个 $T$ 验证单调不减） |

多个截断过滤器叠加**不总能交换顺序**——每经过一个过滤器都在其支持集内重新归一化，下一步看到的已不是原始分布：先 `top_k(3)` 再 `top_p(0.8)` 与反过来会给出不同支持集，数值反例见 Q11 与 §8 [D02]。

<details>
<summary>其他 sampler：typical sampling 与 beam search</summary>

**Typical sampling**：$H(p)=-\sum_i p_i\log p_i$，$d_i=|-\log p_i - H(p)|$；按 $d_i$ 从小到大排序（"信息量最接近整体熵"的 token 优先），取累计概率首次达到 `typical_p` 的最小前缀，重新归一化。**不等价于 top-p**——支持集不必是按概率排序的连续前缀：[D01] 构造 $p=[0.02,0.35,0.33,0.30]$，`typical_p=0.63` 时支持集是第 2、3 高概率的两个 token，反而排除了概率最高（0.35）的那个。

**Beam search**：维护宽度 $B$ 的候选集合，近似最大化序列级 $\log P(y_{1:T})=\sum_t \log p(y_t\mid y_{<t},x)$；逐步剪枝**不保证全局最优**；长度归一化不唯一（常见形式 $S(y)=\log P(y)/|y|^{\alpha}$，不要当成普适定义）。它解决的是序列级搜索问题，不是专治"缺乏多样性"——对话生成里反而可能偏向通用、重复或过早结束的文本。

</details>

### 3.2　惩罚类：penalty 不是"减同一个常数"

$$\text{Frequency/presence penalty}:\quad z_i' = z_i - \alpha_f c_i - \alpha_p\mathbf{1}[c_i>0]$$

frequency 项随出现次数线性增长，presence 项只关心"出现过没有"；`counts` 是否含 prompt token（而非仅已生成部分）由具体 API 语义决定，教程/框架应显式声明。

$$\text{Sign-aware repetition penalty}\ (r>1):\quad z_i' = \begin{cases} z_i/r & z_i>0 \\ r\,z_i & z_i<0\end{cases}\qquad\text{仅作用于已出现的 token}$$

**不是**对所有已出现 token 减同一个常数——符号不同，缩放方向不同（正 logit 被压低，负 logit 被压得更低）。

### 3.3　约束解码与 best-of-n

JSON schema、正则、FSM 驱动的约束解码，本质是**每一步把当前状态下非法的 token logit 置为 $-\infty$，再照常做 softmax + 采样**（§8 [D01] 的 `apply_hard_mask` 演示了这个原语）。**bad-word/banned-sequence**（禁止某些 token 序列再次出现）、**forced tokens**（只给指定 token 留下有限 logit）、**no-repeat n-gram** 都是同一个 mask 原语的特例——真正的 mask 时机取决于**当前已生成后缀的状态**，永久禁用单个 token 只是 banned sequence 长度为 1 的特例。自动机状态转移（FSM/PDA）、tokenizer-aware 匹配、dead-end 处理这些真正麻烦的部分见 Q9。

**best-of-n / parallel sampling** 不能当成普通采样参数：$n$ 条候选共享同一段 prompt 的 KV（copy-on-write，§4.2），只有生成后缀近似按 $n$ 倍增长——KV 占用与调度记账的完整版本见 Q24。

### 3.4　Stop conditions 与流式输出边界

Stop conditions 至少要区分：**EOS token**、**stop-token 序列**、**stop string**（字符串级，可能跨多个 token 边界）、`max_new_tokens`、**总 context 上限**、**约束解码的终态**（如 JSON 语法闭合）、**deadline**、**客户端取消**。

stop string 跨 token 边界时，服务器必须在推给客户端前**缓冲一段"可能仍是 stop string 前缀"的未提交文本**，确认不会再匹配到 stop string 才能安全吐出（§8 [D03] 给出最小实现）。这与 tokenization 教程的 **prefix instability** 同属"边界不稳定"问题但发生在不同层面——一个在 tokenizer 的 encode 边界，一个在流式输出的字符串匹配边界；前者与 token healing 的完整机制见 [`tokenization_tutorial.md`](tokenization_tutorial.md) §5.4。

## §4 KV 生命周期：分配、分页、共享与路由

### 4.1　容量核算：把 sibling 教程的已知量接到 serving 侧

每 token 的 KV 字节数直接复用 [`kv_cache_speculative_decoding_tutorial.md`](kv_cache_speculative_decoding_tutorial.md) §2.1 的结论：

$$m_{\text{token}} = 2\, N_{\text{layer}}\, N_{\text{kv\_head}}\, d_{\text{head}}\, b$$

（因子 2 对应 K/V，$b$ 为每元素字节数；这是**理想 payload**，不含 allocator 对齐、block table 元数据、量化 scale 等开销——MQA/GQA/MLA 的推导见那篇教程）。本教程把 $m_{\text{token}}$ 当成外部已知参数，只关心它进入 serving 生命周期之后的三件事：block 分配与内部碎片、跨请求共享与 copy-on-write（§4.2）、抢占/淘汰时的回收（§4.4）——[D07]–[D08] 把这条已知量接到这三件事上，不重复验证公式本身。

### 4.2　PagedAttention：block table，不是自动换出到磁盘

核心机制（完整设计见 [`kv_cache_speculative_decoding_tutorial.md`](kv_cache_speculative_decoding_tutorial.md) §5，本节只讲 serving 视角需要的那一半）：每个序列的逻辑 KV block 通过一张 **block table** 映射到显存里不连续的物理 block，attention kernel 按这张表间接寻址读取 KV。固定大小 block 消除了"每个请求必须预留整段连续最大长度 KV 空间"的要求，降低外部碎片与过度预留；但**最后一个未填满的 block 仍有内部碎片**（§8 [D07] 的分配器演示）。

> ⚠️ **PagedAttention 不等于"缺页时自动从磁盘加载 KV"** — OS 分页只是设计灵感上的类比；PagedAttention 的核心是 **GPU attention kernel 与 block-table 间接寻址机制**，管理的是显存内不连续物理 block 的映射与共享。这不代表整套 serving 系统就此与磁盘/host 内存绝缘——真正涉及"移出显存"的是另一套可以叠加在它之上的机制：host swap / KV offload（把 KV 挪到 CPU 内存甚至更慢的存储），这是 §4.4 preemption 里的一种代价选项，只是**不属于 PagedAttention 本身的默认行为**。

物理 block 可以通过引用计数在共同前缀或 beam 分支间共享；共享的可写尾 block 需要分叉时，通常触发 **copy-on-write**（复制该 block 并改写映射，而不是原地修改共享内容）。

### 4.3　Prefix caching 与多租户正确性

Prefix caching 复用的是**精确 token 前缀**的 KV，判定主干是"**token ID 序列逐位相等 + 产生 KV 的一切配置兼容**"——例如同一模型版本/权重、同一 adapter（LoRA 等）、同一位置编码与 attention 配置、同一 KV dtype/内存布局、同一并行分片方式。

> ⚠️ **"语义相似的 prompt"不能共享 KV** — prefix caching 只减少**重复 prefill** 的开销，不减少未命中后缀的 prefill 工作量，也不减少后续任何 decode 工作量；它不是语义缓存，也不是检索增强。

多租户场景下，prefix cache 是一个真实的安全边界问题：cache 的 **hash key 设计、访问权限、淘汰策略**都必须防止不同租户通过"cache 命中与否""响应延迟差异"这类侧信道，推断出彼此的私有前缀内容，或者更糟——直接复用到不该访问的缓存内容。

### 4.4　Eviction 与 preemption

对标准全因果 attention，**任意淘汰旧 KV 都会让输出失去与"保留完整历史 KV 的精确计算路径"的等价保证**——被淘汰位置原本应该参与每个后续 query 的 attention 计算，丢弃后就引入了一个未经训练验证的近似。安全的 sliding-window eviction 需要模型本身就采用窗口化注意力训练/推理，或者明确接受 attention sink、压缩等**近似方法**带来的语义变化——eviction 是有代价的工程选择，不是无损优化。

Preemption（抢占）常见的几种做法，代价和是否真的能缓解显存压力都不一样——这是一个常见 taxonomy，不是穷尽分类：

- **丢弃 KV 后重新计算**（recompute）：真正释放物理 KV block，代价是额外的重算算力开销；
- **迁移到 host 内存**（swap）：也真正释放 GPU 显存，代价是额外的传输开销（随 KV 大小增长）；
- **暂停但保留 KV（descheduling）**：只是把这条请求从本轮调度里移出去，不参与本轮迭代，但它**继续占用全部物理 KV block**。这一种**不应该和前两种并列成同一类"内存回收手段"**：它能腾出的是算力/调度份额，缓解不了 KV 显存压力；如果目标是给别的请求腾 KV 空间，descheduling 单独用没有帮助，必须配合 recompute 或 swap 才行。

选哪种（或哪几种组合）取决于重算成本 vs 传输带宽 vs 显存压力的具体权衡，没有普适最优解。

长 generation 请求的三重影响——拉长混合 batch 的 iteration、长期占用 KV 容量、抬高 preemption 触发概率——在 Q21 完整展开。

### 4.5　Cache-aware routing

多副本部署下，路由到哪个副本不能只看"prefix cache 命中率最高"：

> 🎯 **Cache-aware routing 的正确目标是最小化预计完成时间或 SLO 违约概率，不是最大化命中率** — 命中率最高的副本如果排队已经很深，仍然可能比一个无命中但空闲的副本更慢。路由决策需要联合考虑：当前排队长度、预计 KV 占用、prefix-cache 命中带来的 prefill 节省、以及故障域隔离，而不是单一指标贪心。

## §5 调度：Scheduler Tick、Batching 与 Admission

### 5.1　Scheduler iteration 全景：一次 tick 做了什么

调度器每次 tick（一次迭代）是连接 admission、KV、prefill、decode、采样、streaming 的枢纽，核心四步：

1. **回收与补位**：按 §1.7 的清理协议回收上一轮终态请求（释放 batch slot 和 KV block），再按调度策略（§1.3）从 QUEUED/DEGRADED 里选请求补进 active 集合——前提是 `max_num_seqs`、KV 物理容量和已预留份额匹配得上，被选中者从"预留"转为物理分配（QUEUED → RUNNABLE）。
2. **分配预算、组 mixed batch**：`max_batched_tokens` 是**这一次迭代**的 token 工作量预算（不是整条请求生命周期的预留，后者在准入阶段已算过，§1.2），在 prefill chunk（§5.4）和每个 active 请求的 1 个 decode token 之间分配。
3. **执行与输出**：跑一次同时含 prefill chunk 和 decode token 的 mixed forward（kernel 层面差异见 §2.3 结尾），对 logits 应用 §3 的采样管线产出下一个 token，交给 streaming/stop-string 检测（§3.4）；客户端消费跟不上则进 BACKPRESSURED（§0.2）而不是无限缓冲。
4. **更新状态**：检查终止条件（EOS/stop/长度/deadline/取消）并标记终态，留给下一轮回收；KV 压力过大或要给高优先级腾资源时，按 §4.4 执行 preemption（recompute/swap/descheduling，只有前两种真正释放显存）。

speculative decoding 会在第 3 步引入"验证后可变数量接受 + 被拒候选的 KV rollback"，对这套 tick 的记账影响见 §7。

### 5.2　Static batching 的真正限制

> ⚠️ **Static batching 的关键限制不是"必须等齐所有请求才输出"** — 已经完成的序列会立刻产出自己的 token（不需要等别人）；真正的限制是**它留下的 batch slot，通常不能在整批结束前被新请求替换**——空着也要等到这批里最长的那个请求跑完，调度器才会组下一批。

### 5.3　Continuous（in-flight）batching

Continuous batching 在**迭代边界**移除已完成的请求、接纳新请求进入空出的槽位，主要收益来自**减少 decode padding、空槽和队首阻塞**。

> ⚠️ **Continuous batching 不保证在每种负载下都降低每个请求的延迟** — 当所有请求长度接近时，continuous 和 static 几乎没有空槽可省，两者的 makespan 可能完全相同（§8 [D04] 的等长请求反例）。

### 5.4　Chunked prefill

把长 prompt 的 prefill 拆成多个固定大小的 chunk，让调度器可以在 chunk 之间插入 decode 工作，从而限制单次迭代的 token 工作量、避免一个长 prefill 独占整个迭代（head-of-line blocking）。

> ⚠️ **Chunked prefill 通常改善 decode 的 ITL 和混合负载下的公平性，但可能增加这个长 prompt 自身的 TTFT** — 因为它自己的 prefill 工作被分摊到了更多次迭代里，每次迭代能分到的 prefill 预算变小了。具体结果取决于 decode 优先级策略、chunk size、kernel batching 效率——§8 的 [D05] 用一个简化的 token-budget 模型同时算出这两个方向相反的效果。这里的"decode gap"衡量的是 **decode service gap**——decode 请求两次被服务之间间隔了多少个 iteration，是 ITL 的一个**代理量**，不是 ITL 本身；只有在"每次 iteration 耗时大致相等"这个简化假设下，service gap 从 3 个 iteration 降到 1 个才对应更低的 ITL——真实系统里不同 iteration 的实际耗时会因 batch 组成、prefill/decode 混合比例而变化，代理量和真实 ITL 之间不是严格的线性换算。

### 5.5　Admission control 与 KV/token-budget 联合约束

调度器实现层面，admission 联合维护 `max_num_seqs`、`max_batched_tokens`（iteration 级预算，§5.1）、KV block 剩余容量、租户配额四个约束——任何一个触顶都要拒绝或排队，且判断只影响"预留"是否成功，物理分配等到被 tick 选中的那一刻；完整两阶段准入见 §1.2。

## §6 单机到集群：副本、并行与 Prefill/Decode Disaggregation

### 6.1　Replica routing

多副本部署下的路由决策要在**当前排队长度、预计 KV 占用、prefix-cache locality、故障域**之间做权衡——cache-aware 的完整版见 §4.5。

### 6.2　TP/PP/EP：forward-only 的推理视角

> ✅ **模型单卡可放、且目标是聚合吞吐（aggregate throughput）时，data-parallel 副本通常是优先考虑的基线** — 避免了单请求跨 GPU 通信，还能基于负载和 prefix locality 自由路由（§4.5、§6.1）。TP/PP/EP 是否有益需要按延迟目标、batch size 和互联带宽实测判断——**模型放不下是最常见的硬约束，但不是切分并行的唯一理由**。

推理阶段没有 backward、没有 optimizer state，但这不代表并行通信消失了——通信量分析、backward/ZeRO/FSDP/训练 pipeline schedule 的完整推导见 [`distributed_training_tutorial.md`](distributed_training_tutorial.md)，这里只讲推理特有的三条：

- **TP**：每层 forward 仍需要 collective communication（如 row-parallel 后的 all-reduce）；**小 decode batch 下，逐 token 的通信延迟很容易压过计算节省**。
- **EP（MoE）**：推理阶段的 expert parallel 仍需要 token dispatch/combine 通信，面对专家负载不均衡；**没有 backward 不会消除 all-to-all 延迟、热点专家、小消息通信开销这些问题**；EP 的使用主要由 MoE 的专家布局驱动。
- **PP**：单请求、低并发场景下 PP **不能自动降低延迟**，因为端到端仍要经过全部 stage；需要足够多并发序列或 micro-batch 才能把流水线填满，否则大部分时间都在等 bubble。

### 6.3　Prefill/Decode Disaggregation

把 prefill 和 decode 拆到不同节点/资源池的动机：两阶段资源特征差异大（prefill 算力密集、decode 带宽密集）、可以独立扩容、减少长 prefill 对同节点 decode ITL 的干扰。

> ⚠️ **Disaggregation 不是无条件加速** — 收益必须超过：KV 跨节点传输开销、额外的排队与元数据协调、负载不平衡、以及故障恢复的复杂度。短 prompt（KV 传输相对总收益占比高）或弱互联环境下，disaggregation 反而可能更慢——这是一个需要实测 break-even 点的工程决策，不是默认更优的架构。Prefix cache 命中会缩短有效 prefill、改变 prefill/decode 两侧的资源配比，进一步影响这个 break-even 点该怎么算。

## §7 优化组合与瓶颈迁移

单独看每个优化手段都有清晰的收益故事，但**组合在一起时收益不能直接相乘或相加**——它们共同改变的是 batch size、显存余量、kernel 形状和瓶颈的位置本身：

- **量化**：权重量化主要降低权重的带宽与容量压力；**KV-cache 量化是独立的设计点**，不能因为模型权重用了低比特就假定 KV 也理所当然用相同比特数（数值敏感度、per-channel/per-token 粒度选择完全不同）。数值细节（scale/zero-point、GPTQ/AWQ/FP8 具体误差分析）见 [`quantization_tutorial.md`](quantization_tutorial.md)，本节只给决策维度：

  | 量化对象 | 主要影响 | 与 serving 的耦合点 |
  | --- | --- | --- |
  | 权重 | 带宽、显存容量 | 改变 decode 算术强度的转折点（§2.3） |
  | KV cache | 显存容量、允许的 batch/context 上限 | 影响 §4 的容量核算与 block 分配 |
  | Activation | kernel 支持、数值稳定性 | 影响能用哪些融合 kernel、prefill 的实际吞吐 |

- **Speculative decoding**：接受率、$E[\tau]$ 期望加速比的完整推导见 [`kv_cache_speculative_decoding_tutorial.md`](kv_cache_speculative_decoding_tutorial.md) §7；本教程只强调 serving 记账：每次 target verification 会处理多个候选 token，产生**可变数量**的已接受 token，调度器必须按"验证 token 数"“输出 token 数”“KV 增量”分别记账，不能当成固定步长的 decode。Target 或 draft 的量化可能通过数值误差改变 draft/target 的分布差异与接受率——**speculative decoding 与量化不严格正交**。
- **Prefix caching × continuous batching × disaggregation**：prefix 命中缩短的是有效 prefill 长度，这会改变 disaggregated 部署里 prefill pool 的实际工作量、进而改变两个资源池该按什么比例配置——三者互相耦合，孤立评估任何一个都会得出误导性结论。

> 🎯 **线上现象 → 定位路径（诊断表）**

| 现象 | 优先检查 |
| --- | --- |
| TTFT 高 | 排队时长、admission 拒绝率、prefix cache 命中率、prefill 是否被 chunk 挤占 |
| TPOT/ITL 抖动 | 是否有长 prefill 插队、preemption 频率、batch 组成是否频繁变化 |
| KV OOM | block 分配碎片率、长 generation 请求占用、admission 的 KV 预留策略 |
| cache 命中低 | hash key 设计、租户隔离策略是否过度分区、路由是否破坏了 locality |
| GPU 利用率低但延迟也不好 | 先看 batch 是否太小/太碎，而不是急着加 GPU——多半是调度或 admission 问题 |
| p99 突然恶化 | 先拆 queueing time 和 service time；再按 queue → cache hit → prefill → decode → KV pressure → preemption → kernel/collective → 网络 → streaming 分阶段打点（cache hit 影响是否/要做多少 prefill，排在 prefill 之前），不要只盯 GPU 利用率 |

## §8 从零实现：demo 索引

完整可跑脚本见 [`code/inference_serving.py`](code/inference_serving.py)（纯标准库、无第三方依赖、CPU 秒级跑完 [D01]–[D09]），每个 demo 验证一个孤立机制：

| Demo | 验证什么 | 对应正文 |
| --- | --- | --- |
| [D01]–[D02] | 采样过滤器精确支持集、$T\to0^+$ 极限、约束 mask；过滤顺序不可交换 | §3.1–§3.3 |
| [D03] | stop-string 跨 token 的流式缓冲 | §3.4 |
| [D04]–[D05] | continuous batching（含等长请求反例）与 chunked prefill 权衡 | §5.2–§5.4 |
| [D06]–[D08] | KV per-token 字节数落地、block 分配内部碎片、prefix 共享 | §4.1–§4.3 |
| [D09] | roofline ridge point（上界参考，非延迟预测器） | §2.3 |

关键断言片段折叠如下；完整不变量清单与脚本真实输出见 §A。

<details>
<summary>(a) 采样：关键断言片段</summary>

拆成"计算过滤后概率"（`top_k_filter`/`top_p_filter`/`min_p_filter`/`typical_filter`，纯函数、不碰随机数）和"抽样"（`sample`，唯一调用 RNG 的函数）两类。

```python
# top-p: 排序后累计概率首次越过阈值的最小前缀，边界 token 必须保留
p_ex = [0.4, 0.3, 0.2, 0.1]
filt = top_p_filter(p_ex, 0.65)
support = {i for i, p in enumerate(filt) if p > 0}
assert support == {0, 1}                                   # 累计 0.7 >= 0.65
assert sum([p_ex[0]]) < 0.65                                # "严格小于 p0" 会漏掉边界 token

# min-p 支持集随温度改变（非固定 logit 差阈值）
sizes = [len(min_p_support_from_logits(logits_t, T, alpha=0.3))
         for T in (0.2, 0.5, 1.0, 2.0, 5.0)]
assert sizes == sorted(sizes) and sizes[0] < sizes[-1]      # T 越大支持集越大

# 过滤顺序不可交换
support_a = support_of(top_p_filter(top_k_filter(probs, 3), 0.8))   # top_k -> top_p
support_b = support_of(top_k_filter(top_p_filter(probs, 0.8), 3))   # top_p -> top_k
assert support_a != support_b                               # {0,1} vs {0,1,2}
```

</details>

<details>
<summary>(b) continuous-batching simulation：关键断言片段</summary>

3 个请求 A/B/C 同时到达，capacity=2，所需 decode token 数分别为 1/4/1，每个 active 请求每 tick 产出 1 个 token：

```python
reqs = [Request("A", 0, 1), Request("B", 0, 4), Request("C", 0, 1)]
comp_static, _  = simulate_batching(reqs, capacity=2, policy="static")
comp_cont,   _  = simulate_batching(reqs, capacity=2, policy="continuous")
assert comp_static == {"A": 1, "B": 4, "C": 5}    # C 要等到 t=4 才能开始
assert comp_cont   == {"A": 1, "C": 2, "B": 4}    # A 完成后 t=1 立即插入 C

# 注意：这里的 utilization 算的是"调度器 slot 利用率"（serviced token 总量 /
# (capacity * makespan)），是这个单位成本玩具模型里的容量指标，不是 GPU
# SM/HBM 利用率；真实系统里每次 iteration 的实际耗时会随 batch 组成、
# prefill/decode 混合比例变化，不是这里假设的"每 tick 等时长"。
u_static, _ = utilization(comp_static, reqs, 2)   # slot utilization: 6 / (2*5) = 0.60
u_cont,   _ = utilization(comp_cont,   reqs, 2)   # slot utilization: 6 / (2*4) = 0.75
```

等长请求反例（两个都需要 3 tick 的请求）验证 `comp_static_eq == comp_cont_eq`——continuous batching **不总是**严格占优。调度模拟器对所有策略统一检查：生成 token 总数守恒、到达前不能运行、active 数不超容量、完成后立刻释放（不再占 KV）。

</details>

<details>
<summary>(c) KV block 分配与共享：关键断言片段</summary>

per-token 字节数直接取 §4.1 已经说明的 sibling 已知量（$m_{\text{token}}=64$ bytes 只是这个已知量代入一组示例配置后的结果，不是本教程要验证的新结论），本节真正关心的是这个数字进入 serving 侧之后的三件事——block 分配器同时报告 logical 与 allocated 两套数字（内部碎片）；prefix-sharing 场景验证共享后物理 block 数下降；以及"KV heads 不能被 TP degree 整除时不能直接除"的显式检查：

```python
m_token = kv_bytes_per_token(n_layer=2, n_kv_head=2, d_head=4, bytes_per_elem=2)
assert m_token == 64                                        # 2*2*2*4*2 bytes/token

logical_slots, allocated_slots = block_alloc_slots([1, 5], block_size=4)
assert (logical_slots, allocated_slots) == (6, 12)          # 内部碎片：12 > 6

no_share = physical_blocks_no_sharing([6, 7], block_size=4)  # 4
with_share = no_share - 1                                    # 共享一个完整 block -> 3
```

</details>

## §9 25 高频面试题

按难度分三档，答题时按"框架 → 关键公式/定义 → 常见错误"组织，不再重复正文推导。

### L1 必会题

<details>
<summary>Q1. 一个请求从到达到拿到第一个 token，经历了哪些阶段？</summary>

- ingress gate → tokenize/validate → prefix lookup → scheduler admission + KV 预留 → queued/runnable → prefill → 首 token 采样（直接来自最后一次 prefill forward 的 logits）→ **detokenize + 第一次真正 emit 给客户端**（§1、§0.2）；"拿到第一个 token"指客户端侧观测到的第一次 emit，中间的 detokenize 和网络传输通常应算进 TTFT
- 任何阶段都可能被客户端取消/断连打断，须立即回收 KV（§1.7）
- 副本刚启动后第一批请求的 TTFT 含冷启动尾部（模型加载/kernel autotune/图捕获），benchmark 须与稳态分开分报（§1.5、§2.4）

**误区**：把"prefill 完了再单独跑一次 decode 产出第一个 token"当成默认实现；漏掉两阶段准入和首次 emit 这一步。

</details>

<details>
<summary>Q2. TTFT/ITL/TPOT/throughput/goodput 分别怎么定义？</summary>

- $TTFT=t_{\text{first emit}}-t_{\text{arrival}}$，$ITL_j=t_{\text{emit},j}-t_{\text{emit},j-1}$，$TPOT=\frac{t_{\text{last emit}}-t_{\text{first emit}}}{N_{\text{out}}-1}$（§2.1）
- throughput 是**速率**（request / output-token / total-token 三种要先声明哪一种）；**SLO attainment**（$A=N_{\text{SLO}}/N_{\text{total}}$）是比例；**goodput**（$G=N_{\text{SLO}}/\Delta t$）是速率——维度不同不能混用（§2.2）
- 必须声明系统边界：是否含排队/tokenization/网络传输；报均值还是 p95/p99

**误区**：把 TPOT 和"总延迟/token 数"混为一谈；把 goodput 说成"满足 SLO 的请求占比"（那是 SLO attainment）。

</details>

<details>
<summary>Q3. 为什么 prefill 和 decode 的瓶颈通常不同？一定是这样吗？</summary>

- Prefill：$L$ 个 token 一次算完，权重复用率高，通常 compute-bound；decode：小 batch 下反复读权重+KV，通常 bandwidth-bound（§2.3）
- **不是定理**：decode batch 大了算术强度上升可能变回 compute-bound；量化改变转折点；超长 context 下 prefill 也可能被 IO/kernel 限制

**误区**：把这条经验结论当成放之四海而皆准的定理，答不出任何反例。

</details>

<details>
<summary>Q4. top-k/top-p/min-p/temperature 精确支持集分别是什么？</summary>

- top-k：logits 最大的 $k$ 个（$1\le k\le\vert E\vert$，$E$ 为本步合法有限 logit 集合；第 $k$ 位并列时按 $(-z_i,\text{token\_id})$ tie-break）（§3.1）
- top-p：排序后累计概率**首次达到或超过阈值的最小前缀**（保留边界 token）——"$p_i\ge p_0$"和"累计严格 $<p_0$"两种常见定义都会误删边界 token
- min-p：$p_i\ge\alpha\max_j p_j$，等价 logit 阈值 $z_i\ge z_{\max}+T\log\alpha$，支持集随 $T$ 改变，不是固定 logit 差
- temperature：logits 有限、无额外 mask 时，任意 $T>0$ 的支持集是**全词表**；$T=0$ 不属于这条曲线，而是独立的 greedy 分支（tie-break 后唯一确定，并列极限见 §A [D01]）

**误区**：把"首次达到"误记成"首次严格超过"；以为 min-p 阈值和温度无关；忘答 temperature 本身的支持集。

</details>

<details>
<summary>Q5. PagedAttention 实际解决了什么问题？</summary>

- 逻辑 KV block 通过 block table 映射到不连续的物理 block，消除"预留整段连续最大长度"的要求，降碎片（§4.2）
- 支持共享（前缀/beam 分支）+ copy-on-write
- **不是**降低 attention 计算复杂度，**也不是**"缺页自动从磁盘加载"——那是另一套 host swap/offload 机制

**误区**：答成"降低了 attention 的计算复杂度"，或和磁盘换页混为一谈。

</details>

<details>
<summary>Q6. static batching 的关键限制是什么？</summary>

- **不是**"必须等所有请求跑完才输出"——完成的请求立刻能产出结果
- 真正限制：已完成序列留下的 slot **在整批结束前不能被新请求替换**（§5.2）

**误区**：答成"必须等齐"，说不出"slot 不能被替换"这条更准确的表述。

</details>

<details>
<summary>Q7. continuous batching 为什么更高效？是否总是更快？</summary>

- iteration 级移除完成请求、接纳新请求，主要收益是减少空槽/padding/队首阻塞（§5.3）
- **不保证对所有负载都降延迟**：请求长度接近时，static 和 continuous 的 makespan 可能完全相同（§8 [D04] 反例）

**误区**：只会说"更快"，给不出等长请求下两者等价的反例。

</details>

<details>
<summary>Q8. prefix caching 复用的是什么？有什么限制？</summary>

- 复用**精确 token 前缀**的 KV，需同时匹配 token ID、模型版本、adapter、位置编码状态、attention 配置、KV dtype/布局、并行分片方式（§4.3）
- 只减少重复 prefill，不减少未命中后缀和后续 decode 工作
- 多租户场景下 hash key/访问权限/淘汰策略需防止跨租户侧信道

**误区**：以为"语义相似的 prompt"也能命中；不知道命中只省 prefill、decode 工作量不变。

</details>

### L2 进阶题

<details>
<summary>Q9. Constrained decoding（JSON/schema/正则）是怎么实现的？</summary>

- 本质：每步把当前状态下非法 token 的 logit 置为 $-\infty$，照常 softmax + 采样（§3.3）；"当前状态下非法"由跟着已生成文本走的自动机/解析器决定——FSM（正则/简单 schema）、PDA（嵌套结构如 JSON 的括号匹配）、或增量语法解析器
- 合法 token 集合是 **tokenizer-aware** 的：候选 next token 可能跨越多个字符，需要和当前自动机状态做逐字符匹配，不能只在字符串层面判断
- 需要处理 **dead-end**：某个 token 眼下合法，但选了它自动机会走进再也无法达到接受态的状态——完整实现要么前瞻剪枝，要么显式声明这个边界情况没有处理

**误区**：把约束解码讲成需要全新的采样理论，或只停留在"mask 非法 token"一句、说不出自动机状态/tokenizer-aware 匹配/dead-end 这些真正麻烦的地方。

</details>

<details>
<summary>Q10. 如何做一次可信的 serving benchmark？</summary>

- 先分清 **open-loop**（固定 arrival rate 持续发送）和 **closed-loop**（固定并发数）——closed-loop 会在系统变慢时自动降低有效到达率，容易把过载系统压出"虚假稳定"的数字
- 警惕 **coordinated omission**：按计划到达时间而不是实际发送时间算延迟，否则会系统性丢掉最慢的样本，让 p99 显得比真实情况乐观
- 用真实（或匹配真实分布的）输入/输出长度分布 + **load sweep**（报吞吐-延迟曲线和饱和拐点，而非单点吞吐）；冷启动与稳态分报、p99 要有足够样本量（§2.4）

**误区**：只会"跑个压测脚本量吞吐"，说不出 open/closed loop 的区别和 coordinated omission 的失真机制。

</details>

<details>
<summary>Q11. 为什么采样过滤器的执行顺序很重要？给出一个不可交换的例子</summary>

- top-p 依赖"上一步是否已重新归一化"，不同顺序下累计概率基数不同（§3.1）；每经过一个过滤器都要在其支持集内重新归一化，下一步看到的分布已经不是原始分布
- 具体反例（§8 [D02]，输入分布 $p=[0.5,0.2,0.15,0.1,0.05]$）：先 `top_k(3)` 把支持集截到概率最高的 3 个 token 并重新归一化，再做 `top_p(0.8)` 得到支持集大小 2；反过来先做 `top_p(0.8)`（在原始分布上累计到 0.8）再做 `top_k(3)` 得到支持集大小 3——同一份分布，顺序不同，最终支持集不同

**误区**：只会说"顺序可能有影响"，给不出具体反例，也忘了"每一步都重新归一化"这个导致顺序敏感的根因。

</details>

<details>
<summary>Q12. Chunked prefill 解决了什么问题？代价是什么？</summary>

- 把长 prompt 的 prefill 切 chunk，避免独占迭代、卡住同批 decode（head-of-line blocking）（§5.4）
- 通常改善 decode ITL 和混合负载公平性，但**可能增加这个长 prompt 自身的 TTFT**（§8 [D05] 用 token-budget 玩具模型量化了这个此消彼长）

**误区**：只答"改善延迟"，答不出对长 prompt 自身 TTFT 的负面影响。

</details>

<details>
<summary>Q13. Admission control 应该按哪些信号做决策？</summary>

- 两阶段：ingress gate（body/rate/quota，tokenize 之前）→ scheduler admission + KV 预留（tokenize 之后）——完整机制见 §1.2；超限时三种响应：拒绝（REJECTED）、排队（QUEUED，带 deadline）、降级（DEGRADED）
- **prompt 长度在 tokenize 完成后是确定值**，可以精确算 KV 占用；**输出长度在生成完成前只能预测**（启发式/历史统计），预测偏差是过载的重要来源
- 联合约束：KV 容量余量、`max_batched_tokens`（**iteration 级**工作量预算，§5.1，不等于请求级预留）、并发序列数上限、租户配额

**误区**：只说"看 GPU 利用率"；把 `max_batched_tokens`（iteration 级预算）和请求级 KV 预留混为一谈。

</details>

<details>
<summary>Q14. Preemption 有哪几种实现方式？代价分别是什么？</summary>

- 常见做法（不是穷尽分类，§4.4）：**丢弃 KV 后重算**（额外计算，真正释放显存）、**迁移到 host（swap）**（额外传输，代价随 KV 大小增长，真正释放显存）、**暂停但保留 KV（descheduling）**（继续占显存，只释放算力/调度份额）
- **descheduling 不应该和前两种并列成"内存回收手段"**：它缓解不了 KV 显存压力，只是暂时不占计算——如果目标是腾 KV 空间，必须配合 recompute 或 swap
- 选择依据：重算成本 vs 传输带宽 vs 显存压力的具体权衡，没有普适最优

**误区**：说不出三种常见路径和各自代价；把"暂停保留 KV"也当成能省显存的手段。

</details>

<details>
<summary>Q15. Cache-aware routing 的正确目标是什么？</summary>

- 实际目标是若干维度的组合：最小化预计完成时间/SLO 违约概率、最大化 goodput、控制成本、跨租户公平性（§4.5）——先明确这次路由要优化哪一个（或哪几个加权组合）
- 命中率最高但排队很深的副本，可能比无命中但空闲的副本更慢

**误区**：把"命中率最高"或"完成时间最短"当成唯一排他的路由目标，忽略成本和公平性维度。

</details>

<details>
<summary>Q16. Speculative decoding 在 serving iteration 里怎么记账？</summary>

- draft→verify→accept/reject 的位置在 decode 迭代内部；接受率/$E[\tau]$ 的推导见 sibling 教程，本教程只讲 serving 视角（§7）
- 每次 verification 产生**可变数量**的已接受 token；调度器须按验证 token 数、输出 token 数、KV 增量分别记账，不能当固定步长处理
- **accepted、emitted、committed 的 KV 不总相等**：被拒绝的候选 token 对应的 KV 增量需要 rollback，不能算进这条请求已提交的 KV；输出侧还可能包含 bonus/residual token（比如验证成功后 target 模型额外多出的一个自由 token），这个 token 是否计入"输出 token 数"由具体实现的计费口径决定，必须显式声明（§1.7）

**误区**：把 spec decoding 当成固定倍数的吞吐提升；以为生成的 token、真正提交的 KV、最终计费的输出 token 数三者总是一回事。

</details>

<details>
<summary>Q17. 为什么同一个随机种子不保证跨部署得到逐 token 相同的输出？</summary>

- continuous batching 下 batch 组成变了，kernel 的 batch 形状和 reduce/all-reduce 顺序跟着变——浮点加法不满足结合律，顺序变了数值就变（§1.6）
- **多请求共享同一个 RNG 流时，谁先消费到流的哪一段取决于调度顺序**，"固定种子"因此失去意义——种子只是位级复现的必要条件之一（完整清单见 §1.6 折叠块）

> 注：标准 MoE **不会仅仅因为负载变化就自动换路由**——只有在 capacity limit/drop、或数值扰动改变了 top-k 排序这些具体条件下，负载才可能间接改变某个 token 落到哪个专家。

**误区**：以为固定种子就能跨环境逐 token 复现；把"系统固有"当万能挡箭牌——batch 组成、并行拓扑、kernel 路径都固定了还对不上，就要怀疑真实实现 bug。

</details>

### L3 顶级 lab 题

<details>
<summary>Q18. 什么情况下值得做 prefill/decode disaggregation？</summary>

- 动机：两阶段资源特征差异大、可独立扩容、减少长 prefill 对同节点 decode ITL 的干扰（§6.3）
- **不是无条件加速**：收益须超过 KV 跨节点传输、排队协调、故障恢复的额外成本；短 prompt/弱互联环境可能更慢
- prefix cache 命中会改变有效 prefill 长度，进而改变两个资源池的最优配比

**误区**：把 disaggregation 当成默认更优的架构，不谈 break-even 条件。

</details>

<details>
<summary>Q19. 如何选择 TP/PP/EP 和 replica 数量？</summary>

- **并行方式**：模型单卡可放、目标是聚合吞吐时，data-parallel 副本通常是优先基线，避免跨 GPU 通信（§6.2）；TP/PP/EP 是否有益要按延迟目标、batch size、互联带宽实测，"模型放不下"是最常见但不是唯一动机——TP 小 decode batch 下逐 token 通信延迟容易压过计算节省；PP 低并发时不能自动降延迟，需要足够 micro-batch 填流水线；EP 的 no-backward 不会消除 all-to-all/热点专家延迟
- **replica 数量该怎么定**（这题真正的重点，只答并行方式只答了一半）：
  1. 先确定单副本在给定 SLO 约束下能达到的 **goodput**（不是裸吞吐，§2.2），这需要在目标延迟约束下做 §2.4 的 load sweep 实测出来，不能只靠理论估算
  2. 用目标 arrival rate 除以单副本 goodput，得到理论所需副本数下界
  3. 再叠加**峰值余量**——不能按平均到达率配置，要按可预期的峰值/突发流量留出余量
  4. 加上 **N+1 冗余**，保证单副本故障/滚动升级时不会导致整体过载
  5. 结合 **topology**：副本是否分散在不同故障域、是否共享同一批交换机，会影响路由（§4.5、§6.1）和实际可用的聚合 goodput

**误区**：只会"模型大就上 TP/PP"，答不出小 batch/低并发下这些并行反而可能变慢；不给 goodput-based sizing 的具体方法。

</details>

<details>
<summary>Q20. 服务 p99 突然恶化，你怎么定位？</summary>

- 先把延迟拆成 **queueing time**（排队等 admission/调度）和 **service time**（真正在跑 prefill/decode），分位数恶化经常是排队侧的问题，而不是单个请求本身变慢了
- 再按状态机顺序分阶段打点：queue → cache hit → prefill → decode → KV pressure → preemption → kernel/collective → 网络 → streaming（§7 诊断表；cache hit 在 prefill 之前，因为它决定这条请求到底要不要做/做多少 prefill）
- 看 **workload mix** 是否变了（输入/输出长度分布突然变化、某个租户突发大量长 prompt）；如果有分阶段 trace，直接对照哪个阶段的耗时占比上升了
- 检查是否踩了 **cold-start/autoscaling**（新副本刚上线还在冷启动、或者扩容期间副本数一过性下降）
- 排查压测/监控本身是否有 **coordinated omission**（按实际处理时间而不是计划到达时间统计，会系统性漏掉最慢的请求）
- 不能只看 GPU 利用率——利用率低也可能是调度/admission 问题，而不是需要加卡

**误区**：只看 GPU 利用率、没有分阶段方法论；把 cache hit 排在 prefill 之后；不区分排队时间和服务时间。

</details>

<details>
<summary>Q21. 一个长 generation 请求会怎样影响其他请求的 SLO？</summary>

- 在 continuous batching 下**通常不会独占整个 batch**，但影响不只是"占着 KV 配额"这一个容量层面：
  1. 它的 context 越长，每一步 decode 要读的历史 KV、做的 attention 计算就越多，**直接拉长这次混合 batch 的 iteration 时长**，同批里其他请求的 ITL 会跟着变差——这是直接效应，不是"挤占配额"的间接效应
  2. 它**长期占用 KV block 和 active-sequence 配额**，推高显存压力
  3. 显存压力上升会**提高触发 preemption 的概率**，进而影响更多请求（§4.4）
- 更准确的说法是"直接拉长 iteration + 占用容量 + 抬高抢占风险"这三重影响的叠加，不是笼统的"容量层面的干扰"，更不是"阻塞所有请求"

**误区**：用"一个长请求会阻塞所有请求"这种过强表述，或反过来窄化成只是"占容量"、漏掉直接拉长其他请求 ITL 这条机制。

</details>

<details>
<summary>Q22. 多租户场景下 prefix cache 有哪些安全隐患？</summary>

- hash key 设计不当可能让不同租户的前缀发生碰撞或被推断存在性；命中/未命中造成的响应延迟差异本身就是侧信道（§4.3）
- 命中判定应该做 **tenant namespace 隔离** + **精确 token 校验**（hash 碰撞不代表真的是同一段前缀）；淘汰策略同样需要租户隔离，否则租户 A 可能把租户 B 的敏感前缀挤出或窥探
- "给 KV 缓存加密存储"防的是**静态存储泄漏**，解决不了核心问题——跨租户的**授权隔离**和**命中/延迟侧信道**是访问控制和信息泄漏层面的问题，不是数据有没有加密的问题

**误区**：只答"加密存储"并当成全解；答不出命中/延迟侧信道这个具体机制。

</details>

<details>
<summary>Q23. 为什么不能对标准全因果 attention 随便做 sliding-window eviction？</summary>

- 任意淘汰旧 KV 都会让输出**失去与"保留完整历史 KV 的精确计算路径"的等价保证**——标准 attention 的每个 query 理论上都可能依赖任意历史位置，丢弃某个位置的 KV 后就不再是精确计算，而是未经训练验证的近似
- 安全做法：模型本身训练/推理时就用窗口化注意力，或明确接受 attention sink/压缩类方法带来的近似语义变化，把 eviction 当成一个有代价的工程选择

**误区**：以为 eviction 是可以无损叠加在任何模型上的"免费优化"。

</details>

<details>
<summary>Q24. best-of-n/parallel sampling 对 KV 和调度有什么额外成本？</summary>

- 并行生成 $n$ 条候选，但它们**共享同一段 prompt**——prompt 的 KV 可以通过 copy-on-write 共享一份，不必整体放大 $n$ 倍；更准确的近似是 $M_{KV}\approx M_{\text{prompt}}+\sum_j M_{\text{branch},j}$，只有各分支**生成后缀**的 KV 才近似按分支数增长（§3.3、§4.2）
- 只有实现里**没有做 CoW 共享**时，才会退化到接近 $n(M_{\text{prompt}}+M_{\text{generated}})$ 这个更悲观的上界；wall time 也不必然是单条候选的 $n$ 倍，取决于这些候选能否塞进同一批 decode 迭代
- admission/KV 容量核算必须把这份增量记进去，否则会在运行时才发现容量估计错误

**误区**：把 best-of-n 当成"几乎不花额外代价"的轻量参数，或错记成"KV 整体乘以 n"而忽略 prompt 前缀本可共享。

</details>

<details>
<summary>Q25. 长 prompt 到达时，如何保护正在运行的在线 decode 请求的 SLO？</summary>

- 手段组合：chunked prefill 限制单次迭代的 prefill 预算、decode-priority 调度、独立的 prefill 资源池（进一步演化为 disaggregation）、admission 按预测长度分流（§5.4、§6.3）
- 权衡：在同一份资源里挤时间片的手段（如 chunked prefill）通常会拉长长 prompt 自身的 TTFT；而独立资源池/disaggregation sizing 得当时可能双赢——"保护 decode 必然牺牲 prefill TTFT"不是普遍成立的结论

**误区**：只答"chunked prefill"一招，或断言"这些手段都会拉长长 prompt 的 TTFT"。

</details>

## §A 附录：sanity check

运行 `python3 code/inference_serving.py`（纯 Python 标准库、无第三方依赖、CPU 秒级）应通过 [D01]–[D09] 全部检查；关键不变量清单与真实终端输出折叠如下。

<details>
<summary>关键不变量清单（[D01]–[D09]）</summary>

1. **[D01] 采样支持集**：greedy 对并列最大值取最低 index；`temperature_probs` 对 $T\le0$ 抛异常；唯一最大 logit 时 $T\to0^+$ 集中到 argmax，并列最大 logit 时质量对半分；top-k 支持集大小恰为 $k$；top-p 在 $p=[0.4,0.3,0.2,0.1]$、阈值 $0.65$ 时支持集为前两个 token（累计 $0.7$）；min-p 恒保留最大概率 token，支持集随 $\alpha\downarrow$ 单调不减，随 $T\uparrow$（$\alpha<1$ 时）单调不减；typical sampling 在构造例子里排除最高概率 token；logits 整体加常数不改变 min-p/top-k/top-p 支持集。
2. **[D02] 过滤顺序不可交换**：同一分布下 `top_k->top_p` 与 `top_p->top_k` 得到不同支持集（$\{0,1\}$ vs $\{0,1,2\}$）。
3. **[D03] 流式 stop 检测**：stop string 跨 3 个 token chunk 仍能被检测到，且中途从未提前泄漏任何不安全的部分前缀；无匹配时全部文本可通过 flush 完整恢复。
4. **[D04] 批调度**：static 给出 $A=1,B=4,C=5$（utilization $0.60$），continuous 给出 $A=1,C=2,B=4$（utilization $0.75$）；等长请求反例下两策略 makespan 相同；调度器满足 token 数守恒、到达前不运行、active 数不超容量、完成即释放。
5. **[D05] Chunked prefill 权衡**：max decode gap 从 3（unchunked）降到 1（chunked），但该长 prompt 自身 TTFT 从 3 升到 4。
6. **[D07]–[D08] KV block 分配与共享**（per-token 字节数复用 §4.1 已经给出的 sibling 已知量 $m_{\text{token}}=64$ bytes，$N_{\text{layer}}=2,N_{\text{kv\_head}}=2,d_{\text{head}}=4,b=2$，不在本教程内单独验证）：长度 $[1,5]$、block size $4$ 下 logical/allocated slots $=6/12$（对应 $384/768$ bytes，内部碎片真实存在）；长度 $[6,7]$ 共享 1 个完整 block 后物理 block 数从 4 降到 3；KV heads 不能被 TP degree 整除时拒绝直接相除。
7. **[D09] roofline ridge point** 仅作为上界参考量，不作为延迟预测器使用。

</details>

<details>
<summary>运行 <code>python3 code/inference_serving.py</code> 的真实输出</summary>

```text
[D01] sampling operators: greedy tie-break, T=0 rejected, T->0 limits (unique->one-hot, tied->split), top-k/top-p/min-p/typical support sets (exact boundaries, nested/shift-invariant, domain-checked), penalties, constrained masking, deterministic + empirical draw checks PASS
[D02] filter order is NOT commutative: top_k->top_p support=[0, 1], top_p->top_k support=[0, 1, 2]  PASS
[D03] stop-string streaming: 'STOP' split across 3 chunks still detected (emitted='abc'), no-match case fully recoverable via flush  PASS
[D04] static: A=1,B=4,C=5 (util=0.60); continuous: A=1,C=2,B=4 (util=0.75); equal-length counterexample: both makespan=3 (no strict dominance); delayed-arrival workload respects arrival<=tick<completion; illegal inputs (7 cases) rejected immediately  PASS
[D05] chunked prefill: decode max-gap 3->1 (ITL improves) but long-prompt TTFT 3->4 (worsens) -- not a free lunch  PASS
[D06] KV bytes/token (L=2,kv_head=2,d_head=4,fp16)=64B; [D07] block alloc lengths=[1,5],P=4: logical=6 slots/384B, allocated=12 slots/768B; [D08] prefix-sharing (real token-sequence match) blocks 4->3, sub-block overlap shares 0, 3-way sharing scales as (n-1)*B_shared, TP-divisibility guard enforced  PASS
[D09] roofline ridge point I* = 201.3 FLOPs/byte (upper-bound reference only)  PASS

all inference serving sanity checks passed
```

（以上为重新运行 `code/inference_serving.py` 后截取的真实终端输出，与该脚本当前版本逐字一致。）

</details>

## 📚 参考文献

- **PagedAttention / vLLM** — Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention*, arXiv 2309.06180 (2023), SOSP 2023.
- **Orca（iteration-level / continuous batching）** — Yu et al., *Orca: A Distributed Serving System for Transformer-Based Generative Models*, arXiv 2204.05120 (2022), OSDI 2022.
- **Sarathi-Serve（chunked prefill）** — Agrawal et al., *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve*, arXiv 2403.02310 (2024), OSDI 2024.
- **DistServe（disaggregation, goodput）** — Zhong et al., *DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving*, arXiv 2401.09670 (2024), OSDI 2024.
- **Splitwise（phase splitting）** — Patel et al., *Splitwise: Efficient Generative LLM Inference Using Phase Splitting*, arXiv 2311.18677 (2023), ISCA 2024.
- **SGLang / RadixAttention（前缀树 prefix caching）** — Zheng et al., *SGLang: Efficient Execution of Structured Language Model Programs*, arXiv 2312.07104 (2023).
- **Nucleus sampling（top-p）** — Holtzman et al., *The Curious Case of Neural Text Degeneration*, arXiv 1904.09751 (2019), ICLR 2020.
- **Top-k sampling** — Fan, Lewis & Dauphin, *Hierarchical Neural Story Generation*, arXiv 1805.04833 (2018), ACL 2018.
- **Min-p sampling** — Nguyen et al., *Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs*, arXiv 2407.01082 (2024).
- **Locally typical sampling** — Meister et al., *Locally Typical Sampling*, arXiv 2202.00666 (2022), TACL 2023.
- **Speculative decoding** — Leviathan, Kalman & Matias, *Fast Inference from Transformers via Speculative Decoding*, arXiv 2211.17192 (2022), ICML 2023.
- **Speculative sampling** — Chen et al. (DeepMind), *Accelerating Large Language Model Decoding with Speculative Sampling*, arXiv 2302.01318 (2023).
- **FlashAttention（attention kernel IO 优化，与 PagedAttention 不是同一技术）** — Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*, arXiv 2205.14135 (2022), NeurIPS 2022.
- **Megatron-LM（TP）** — Shoeybi et al., *Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism*, arXiv 1909.08053 (2019).
- **Roofline model** — Williams, Waterman & Patterson, *Roofline: An Insightful Visual Performance Model for Multicore Architectures*, Communications of the ACM (2009)（无 arXiv id）.
- **约束解码 / 语法约束生成** — Willard & Louf, *Efficient Guided Generation for Large Language Models*, arXiv 2307.09702 (2023)。该论文支持 guided/grammar-constrained generation 的具体实现，但不宜用这一篇论文去论证"所有约束解码实现本质上都只有 mask-and-renormalize 这一种机制"——本教程 §3.3 的"本质就是 mask"是对采样原语层面的抽象总结，不是对该论文范围的复述。
