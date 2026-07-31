## §0 The Request-State-Machine Mental Model + TL;DR Cheat Sheet

**LLM inference serving is not "run one forward pass" — it is a state machine spanning the gateway, the scheduler, the GPU-memory manager, and the sampler.** The interview-worn "what's the difference between prefill and decode" covers just two cells of this machine — without standing the full path up first, you cannot say which step TTFT is stuck on, which resource a KV OOM is alarming about, or where to start when p99 degrades. This state machine is the through-line of this tutorial; quantization, the KV capacity formula, and the speculative-decoding derivations are already covered in the other tutorials, so here we only discuss **where they plug into the serving pipeline and what they change**.

> 💡 **8 TL;DR items** — grab the through-line in 30 seconds; details unfold in §0.1–§0.2 and §1–§7.

1. **Request path**: ingress gate → tokenize/validate → prefix lookup → scheduler admission + KV **reservation** → queued/runnable → prefill → first token (taken directly from the logits of the last prefill forward, not "a separate decode run") → per-step decode (sampling + streaming) → finished → KV reclaim (§0.1, §1).
2. **Two-stage admission**: the ingress gate does only coarse-grained checks before tokenization; the precise KV/token-budget decision must wait until after tokenize and prefix lookup, and passing it only **reserves** (bookkeeping level) — physical allocation happens when some tick selects the request (§1.2).
3. **Metrics**: $TTFT=t_{\text{first emit}}-t_{\text{arrival}}$, $ITL_j=t_{\text{emit},j}-t_{\text{emit},j-1}$, $TPOT=\frac{t_{\text{last emit}}-t_{\text{first emit}}}{N_{\text{out}}-1}$ — declare the system boundary for each quantity first; throughput is a rate, SLO attainment is a ratio, goodput is "the rate that meets the SLO" — the three must not be conflated (§2).
4. **Typical bottlenecks**: "prefill compute-bound, decode bandwidth-bound" is an empirical conclusion for common operating regimes, not a theorem — larger batches, quantization, and ultra-long context all move the bottleneck (§2.3).
5. **Sampling**: top-p is the smallest sorted prefix whose cumulative probability first reaches or exceeds the threshold (the boundary token must be kept); the min-p support set changes with temperature; multiple filters do not always commute; constrained decoding is essentially setting illegal tokens' logits to $-\infty$ and then sampling as usual (§3).
6. **KV**: PagedAttention is block-table indirection, not "page faults auto-loading from disk"; prefix caching reuses **exact token prefixes**, not semantically similar prompts, and multi-tenant deployments must also guard against cache side channels (§4).
7. **Scheduling**: static batching's real limitation is that empty slots cannot be refilled by new requests before the whole batch ends; continuous batching saves empty slots via iteration-level refill but does not guarantee lower latency for every workload; chunked prefill improves decode ITL yet may lengthen the long prompt's own TTFT (§5).
8. **Scaling and diagnosis**: disaggregation is not an unconditional speedup (its gains must clear the break-even of KV transfer/coordination/failure recovery); optimizations do not stack linearly; the same seed does not guarantee token-level reproduction; when p99 degrades, split queueing time from service time first, then instrument stage by stage along the state machine (§6, §7, §1.6).

### 0.1　Happy path: a request that completes normally

```text
Client request
  → ① ingress gate (coarse-grained fast rejection, no tokenization needed)
  → ② tokenize / validate (only now is the exact prompt length known)
  → ③ prefix lookup (exact token-prefix match; the hit portion reuses KV)
  → ④ scheduler admission + KV reservation (bookkeeping, not physical allocation)
  → ⑤ queued / runnable (waiting to be selected by some scheduler tick)
  → ⑥ prefill (one or more forwards over the missed span)
  → ⑦ first-token sampling → ⑧ per-step decode (sampling pipeline §3 + detokenization/streaming)
  → ⑨ FINISHED → ⑩ KV reclaim + billing/logging
```

Side paths: REJECTED can occur at either admission check ① or ④; PREEMPTED/SWAPPED/RESUMED, BACKPRESSURED, and TIMED_OUT/FAILED/CANCELLED are branches reachable from any running stage — the full state diagram is in §0.2. Three notes below the diagram:

- **The first token comes directly from the logits/hidden state of the last prefill forward** — no "independent" decode iteration needs to start first; the full-prefix-hit exception is in §1.1.
- **KV reservation ≠ physical allocation**: passing ④ only means the quota is reserved in the books; physical blocks are allocated when some tick selects the request (§1.2, §5.1).
- **Stop strings can straddle token boundaries**, so the server must buffer a small span of uncommitted text before it can emit safely (§3.4).

### 0.2　The full state machine: aggregated view

| Category | States | One-liner |
| --- | --- | --- |
| **Admission** | REJECTED / QUEUED / DEGRADED | Two-stage admission decides between reject, queue (KV is only a **reservation** in the books), and degraded admit (§1.2) |
| **Execution** | RUNNABLE / PREFILLING / DECODING | Once a tick selects the request, the reservation becomes physical allocation, and KV keeps growing through decode (§5.1) |
| **Paused** | PREEMPTED / SWAPPED / RESUMED / BACKPRESSURED | Resources reclaimed by the scheduler, KV migrated to host, or the client cannot keep up; the three preemption flavors release different resources (§4.4) |
| **Terminal** | FINISHED / CANCELLED / TIMED_OUT / FAILED | All four paths share one reclaim routine — capacity/scheduling/accounting resource actions + idempotency; full cleanup invariants in §1.7 |

<details>
<summary>Full table of states, guards, and held resources (14 states)</summary>

| State | Entry condition (guard) | Held resources | Main exit transitions |
| --- | --- | --- | --- |
| **REJECTED** (terminal) | The ingress gate finds body/rate/quota over limit; or tokenize/validate fails; or scheduler admission finds the KV/token budget unsatisfiable even with queueing | None (never held KV or a scheduling slot) | Terminal; returns an error to the client |
| **QUEUED** | Passed ingress gate + tokenize/validate + prefix lookup + scheduler admission; the KV **reservation** succeeded but no tick has selected it yet | Request metadata + reserved KV quota (bookkeeping, not physical blocks) + a queueing-deadline timer | → RUNNABLE (selected by a tick) / → TIMED_OUT (queueing deadline exceeded) / → CANCELLED (client cancels) / → DEGRADED |
| **DEGRADED** | Under capacity pressure, the scheduler chooses to lower this request's allowed best-of-n/max concurrency instead of rejecting outright | Same as QUEUED, but the reservation is recomputed under the degraded parameters | → RUNNABLE |
| **RUNNABLE** | Selected by a scheduler tick that has a free batch slot and physical KV blocks covering at least this round of work | The reservation converts to allocated physical KV blocks + this iteration's batch slot + a token-budget share | → PREFILLING/DECODING (normal execution) / → PREEMPTED (scheduler reclaims resources for a higher-priority request) |
| **PREFILLING → DECODING** | Respectively: the missed prompt span is not yet fully computed / at least one token produced and not yet terminated | Same as RUNNABLE, with KV growing continuously | → FINISHED / → BACKPRESSURED / → PREEMPTED / → CANCELLED / → TIMED_OUT / → FAILED |
| **BACKPRESSURED** | Tokens generated but the client cannot keep up (slow connection/not reading); the output buffer hits its cap | Same as DECODING + a bounded buffer of generated-but-unsent tokens | → DECODING (client resumes consuming) / → PREEMPTED (buffer tops out) / → CANCELLED (final disconnect) |
| **PREEMPTED** | The scheduler decides to free resources for other requests; the concrete action is one of **discard-and-recompute** / **migrate to host (SWAPPED)** / **pause but keep KV (descheduled)** (§4.4, not an exhaustive taxonomy) | Depends on the action: the recompute variant releases all physical KV; SWAPPED releases GPU memory but holds a host-memory copy; the descheduled variant keeps holding all physical KV blocks and releases only the batch slot/compute share | → RESUMED (rescheduled) / → CANCELLED/TIMED_OUT (cancelled or timed out while waiting) |
| **SWAPPED** | The migration sub-case of PREEMPTED; KV has finished moving to host memory | Host-memory footprint (no GPU memory held) + request metadata | → RESUMED (swap-in completes, GPU memory reoccupied) |
| **RESUMED** | The scheduler decides to bring a PREEMPTED/SWAPPED request back into some tick | Physical KV blocks re-allocated/moved back depending on origin + a batch slot | → RUNNABLE (merges into subsequent ticks normally) |
| **TIMED_OUT** (terminal) | The agreed deadline expires while waiting in QUEUED/PREEMPTED | None (moves into cleanup) | Terminal |
| **CANCELLED** (terminal) | The client explicitly cancels or disconnects (heartbeat/half-close detection) | None | Terminal; must be marked **immediately** as the highest-priority KV-reclaim target, not deferred to the next tick |
| **FAILED** (terminal) | Worker crash/uncaught exception | None | Terminal; usually needs to trigger a retry (if idempotent) |
| **FINISHED** (terminal) | Whichever fires first among EOS/stop string/stop-token sequence/`max_new_tokens`/context limit/constrained-decoding accept state | None | Terminal |

</details>

## §1 The Request State Machine: From HTTP to Token Stream

### 1.1　Panorama: what a request actually goes through

Expanding §0's diagram, the details interviewees most often omit:

- **Admission has two stages**: the ingress gate does coarse fast rejection first; the precise KV/token-budget decision waits until after tokenize and prefix lookup — full mechanism (including reservation vs. physical allocation and the three over-limit responses) in §1.2.
- **Prefix lookup happens before prefill**: the scheduler matches existing cache exactly by token IDs (typically hash of the prefix token sequence); the hit portion is skipped, and prefill runs only over the missed suffix span (§4.3).
- **Prefill over the missed span is one or more forwards**: without chunked prefill it usually completes in one pass; with chunked prefill it is split across multiple iterations (§5.4).
- **The first token is not necessarily a separate decode round**: the last prefill forward itself computes the logits at "the last prompt position", and sampling happens right there — provided that forward actually computed this position. **If the prefix cache fully hits** (every prompt token skipped the forward), the system holds only K/V, no final hidden state/logits; it must have cached those logits separately, or at minimum recompute the last prompt token, otherwise there is nothing to sample the first token from — an easily missed step in the "full hit looks fastest" scenario. Many assume "prefill finishes, then a separate decode iteration produces the first token", which breeds the wrong intuition about what TTFT is made of.
- **Per-step decode**: each step runs 1 forward over only the new token, but attention reads the entire KV history; the token sampled this step immediately becomes next step's input (§3 unfolds the sampling details).
- **Detokenization/streaming**: each produced token is incrementally detokenized and pushed to the client; the buffering mechanism for stop strings straddling token boundaries is in §3.4 and §8 [D03].
- **Termination conditions are a whole set, not "EOS and done"**: full list in §3.4; the termination/failure/cleanup protocol in §1.7.

### 1.2　Two-stage admission and overload handling

Whether a request gets accepted is not "admit if a GPU is free" — it is a joint decision in two stages (matching the guards of REJECTED/QUEUED/DEGRADED in §0.2's state table):

**Stage one: ingress gate (before tokenization)** — looks only at coarse signals: request-body size limit, rate limits, whether the tenant quota is exhausted. This layer blocks obviously over-limit traffic at minimal cost, without waiting for the tokenizer to finish.

**Stage two: scheduler admission + KV reservation (after tokenize/validate and prefix lookup)** — only now are the exact prompt length and miss length known; it jointly considers:

- **Current KV headroom** (allocated blocks vs. total blocks, §4)
- **Max batched tokens / max concurrent sequences** (this iteration's token-work budget, §5.1, §5.5)
- **Predicted output length** (unknown output length is the main source of admission risk — most systems can only estimate via heuristics/history, and prediction error is itself one root cause of overload)
- **Tenant quotas** (prevent a single tenant from exhausting global resources in multi-tenant environments)

Passing stage two only means the KV quota is **reserved** (remaining capacity decremented in the books), not that physical blocks are allocated — physical allocation happens when a scheduler tick actually selects this request (RUNNABLE, §5.1).

Three responses when over limit: **immediate rejection** (REJECTED — fail fast so the client can retry another replica), **queueing** (QUEUED — holds request metadata and a reservation but no physical GPU resources; needs a queueing deadline), and **degradation** (DEGRADED — e.g. lower this request's allowed max concurrency/best-of-n, squeezing it in with a smaller reservation instead of rejecting).

### 1.3　Fairness and starvation

The choice of scheduling policy directly decides who starves:

| Policy | Mean latency | Tail latency | Long requests | Short requests |
| --- | --- | --- | --- | --- |
| Pure FCFS | Medium | Can be very bad (long request blocks the queue head) | Fine | Can get stuck |
| Shortest-job-first (needs length prediction) | Better | Depends on prediction accuracy | May starve indefinitely | Prioritized |
| decode-priority | Good on the decode side | Stable decode ITL | Prefill deferred, TTFT worsens | Benefits |
| prefill-priority | Good on the prefill side | Decode ITL jitters | Good TTFT | Decode gets interrupted |

**There is no free scheduling policy** — utilization maximization and fairness/tail latency are usually two goals to trade off deliberately; do not report only "how much throughput improved".

### 1.4　Cancellation, client backpressure, disconnect

If the server does not notice promptly after a client disconnects or explicitly cancels, it keeps running decode for a request **nobody is listening to** — pure GPU waste that also crowds out other requests' scheduling/KV resources. Engineering-wise this requires: heartbeat/half-close detection at the connection layer; the scheduler marking "cancelled" as the highest-priority KV-reclaim target; and backpressure on streaming responses — when the client cannot keep up, generated-but-unsent tokens must not buffer without bound (session memory grows, which then drags down other requests' streaming buffers); a bounded queue + backpressure signal is needed (BACKPRESSURED, §0.2), and once it tops out, either wait for the client to catch up or actively degrade/preempt the request. The full termination/failure/cleanup protocol (deadlines, worker failure, retry/idempotency) is in §1.7.

### 1.5　Model loading and warm-up: cold start is not steady-state TTFT

Right after a replica starts, weight loading (disk/object store into GPU memory), CUDA kernel autotuning, CUDA Graph capture, and memory-pool initialization all make the first batch of requests far slower than steady state. **This cold-start latency must not be mixed into "steady-state TTFT" benchmarks** — evaluation reports must report the cold-start tail and the steady-state distribution separately, or scale-up jitter gets misdiagnosed as a chronic problem. The full evaluation protocol (load sweep, saturation point, p99 sample size, etc.) is in §2.4.

### 1.6　The real limits of reproducibility

> ⚠️ **Same random seed + same weights ≠ token-level reproduction** — continuous batching changes batch composition and kernel shapes (different batch sizes trigger different GEMM/attention kernel paths); floating-point addition is not associative, so changed reduce/all-reduce orders change the numbers; and when multiple requests share one RNG stream, who consumes which segment first depends on scheduling order. Most of the time this is normal behavior for a "batching + parallelism + floating point" system, **but it must not become a universal shield**: if batch composition, parallel topology, and kernel paths are all fixed and outputs still differ, suspect a real implementation bug (uninitialized memory, race conditions) instead of reflexively blaming "inherent system behavior".

<details>
<summary>The full list of what bit-exact token-level reproduction must fix simultaneously</summary>

**Batch composition and arrival order**, **parallel topology** (TP/PP/EP degrees and rank mapping), **hardware model** (different GPU architectures may differ in reduction order and kernel implementations), **software/kernel versions** (library upgrades on the same hardware can also swap kernel implementations), **numeric precision** (fp16/bf16/fp32 round differently), **collective-communication algorithms** (ring vs. tree all-reduce orders differ), **the framework's determinism switches** (many frameworks disable deterministic kernels by default for speed), and **a per-request independent RNG stream** — fixing the random seed is just one necessary condition, far from sufficient. The interview version and the precise MoE-routing conditions are in Q17.

</details>

### 1.7　Termination, failure, and the resource-cleanup protocol

A request can reach a terminal state via four different paths — **normal completion** (FINISHED), **client cancellation** (CANCELLED), **timeout** (TIMED_OUT), **worker failure** (FAILED) — but every path must converge to **the same reclaim routine**, or you get resource leaks or double frees. The full bounded-buffer and backpressure story for slow clients is in §1.4.

**Worker failure and retry**: when a worker process crashes, every request it was serving must be marked FAILED; whether upstream retries depends on whether the request is **idempotent** — if the client already received partial tokens (a partial stream), naive retry produces a duplicated prefix, so either the client deduplicates or the server explicitly declares "on failure, retry from scratch; any partial output already sent is void".

**Cleanup = three classes of resource actions — capacity/scheduling/accounting**: **capacity** — physical KV blocks decrement by reference count, and shared prefixes/beam branches only truly return blocks to the allocator when the last holder releases (requests terminated while still QUEUED/DEGRADED only roll back reservation bookkeeping — there are no physical blocks to free); **scheduling** — release the batch slot and the active-sequence quota; **accounting** — produce exactly one terminal billing/log record. Cancellation and timeout can race and fire almost simultaneously — repeated calls into the reclaim routine must be **idempotent**: never free the same block twice (double-free), and never both "assume the other will handle it" so nothing gets freed (leak).

> 💡 **Billing semantics**: the generated / emitted / committed counts are not always equal — N generated tokens may yield only M ≤ N actually sent to the client (stop-string truncation discards, speculative-decoding rejected candidates); billing and monitoring must state explicitly which number they use, or "generated volume" and "billed volume" will not reconcile.

If the interviewer asks "what exactly does the system do after a request is cancelled", the answer should cover the cleanup invariants shared by all four paths, not just "stop decoding".

## §2 Serving Metrics, SLOs, and the Prefill/Decode Roofline

### 2.1　Precise metric definitions (declare the system boundary first)

$$TTFT = t_{\text{first emit}} - t_{\text{arrival}}, \qquad ITL_j = t_{\text{emit},j} - t_{\text{emit},j-1}$$

$$TPOT = \frac{t_{\text{last emit}} - t_{\text{first emit}}}{N_{\text{out}} - 1}\quad (N_{\text{out}} \ge 2)$$

If the endpoint of E2E latency is defined as "the emit time of the last token" and all timestamps share the same system boundary, then $t_{\text{last emit}} - t_{\text{arrival}} = TTFT + \sum_{j=2}^{N_{\text{out}}} ITL_j$ is an **identity**, not an approximation; if the reported scope also includes post-emit flush/connection close/trailing network transfer, the right side needs that tail term added explicitly — only then write $\approx$.

Every quantity must first declare its **system boundary**: does TTFT include queueing time, tokenization time, gateway/network transfer? "Mean" or "p95/p99"?

> ⚠️ **Do not conflate TPOT with "total latency divided by token count"** — TPOT is the steady-state per-token time of the decode phase (the denominator excludes the first token); folding queueing and prefill time into "total latency / token count" misreports a queueing-affected request as "this model decodes slowly".

### 2.2　Throughput, SLO attainment, and goodput

High system throughput (tokens/s or requests/s) does not mean good user experience — if the batch is packed full to pump throughput and TTFT/TPOT degrade across the board, users perceive a slowdown. Three routinely conflated concepts to pull apart:

- **Throughput**: work processed per unit time, a **rate** — but first declare which dimension: request throughput (requests/s), output-token throughput (output tokens/s), or total-token throughput (total tokens/s including prompt tokens, often used to estimate GPU-time occupancy/cost). The three can rank systems completely differently under long-prompt/short-output or short-prompt/long-output workloads.
- **SLO attainment**: the requests (or tokens) meeting the TTFT/TPOT/deadline SLO **as a fraction of the total**, a dimensionless **ratio** — $A = N_{\text{SLO}} / N_{\text{total}}$.
- **Goodput**: the requests (or tokens) meeting the same SLO constraints **as a rate per unit time** — $G = N_{\text{SLO}} / \Delta t$, still in requests/s or tokens/s, **not a ratio**. Describing goodput as "the fraction of requests meeting the SLO" is common but inaccurate: attainment and goodput are quantities of different dimensions — attainment can be high while goodput is low because the whole system is slow, and attainment can be mediocre while goodput is still numerically sizable because raw throughput is enormous.

When reporting system capability, give the throughput–latency curve (swept across concurrency/arrival rates, see §2.4) rather than a single-point throughput number; declaring which of the three throughputs, and whether it is SLO attainment or goodput, is likewise part of the "system boundary" that must be stated upfront.

### 2.3　The prefill/decode empirical regime — not a theorem

Expanding one standard Transformer layer's FLOPs (full derivation in [`kv_cache_speculative_decoding_tutorial.md`](kv_cache_speculative_decoding_tutorial.md) §3.1, not repeated here) gives roughly: prefill computes $L$ tokens in one pass, reusing the weights $L$ times, so arithmetic intensity is usually high and GPU compute easy to saturate; decode computes only 1 new token per step yet re-reads the whole layer's weights plus the history-growing KV, so at small batch its arithmetic intensity is low and it often lands in the HBM-bandwidth-bound regime. This "prefill compute-bound, decode bandwidth-bound" is an **empirical conclusion for common operating regimes**, with two qualifiers: as decode batch grows, arithmetic intensity rises (the same weights amortize over more tokens), some linear layers can turn compute-bound again, and quantization shifts the batch size at which that flip happens; and at ultra-long context, prefill attention can itself be limited by IO/HBM/workspace/kernel implementation — using FlashAttention does not automatically make the whole prefill compute-bound. Input and output lengths also act asymmetrically: input length directly sets prefill work and per-step decode KV reads, while output length does not change the initial prefill's arithmetic intensity but adds decode rounds, KV residency time, and late-stage context length.

> 🎯 **Roofline is an upper-bound thought experiment, not a latency predictor** — the ridge point $I^{*}=\text{peak FLOP/s}/\text{HBM byte/s}$ exists only to compare against a kernel's own arithmetic intensity and judge which side it theoretically leans toward; real latency also depends on kernel implementation, scheduling overhead, and memory-access patterns — see [D09] in §8.

FlashAttention-style work optimizes the **attention kernel's IO efficiency** (saving GPU-memory reads/writes); PagedAttention (§4) manages **the address space and allocation policy of KV during serving** — they solve problems at different layers; do not merge them into one technique.

Lower still, the pure kernel/assembly-level factors — operator fusion, CUDA Graphs (shape changes force re-capture), prefill's wide GEMM vs. decode's narrow GEMM/GEMV ($L_q=1$), paged attention's extra block-table-indirection cost — are only name-checked here, not expanded; they are all examples of "implementation details visibly moving measured numbers".

### 2.4　Trustworthy benchmarks and capacity planning (overview)

Reporting throughput, goodput, and TTFT/TPOT as one trustworthy set of numbers is easier to get wrong than it looks; below is an incomplete checklist covering the most common pitfalls — full load-test script design is out of scope for this tutorial.

- **Open-loop vs. closed-loop**: open-loop keeps sending at a fixed arrival rate (e.g. Poisson arrivals); closed-loop fixes concurrency (send the next only after one completes). **They measure different things**: closed-loop naturally lowers the effective arrival rate as the system slows, easily squeezing "falsely stable" latency numbers out of an already-overloaded system; open-loop is closer to real production traffic but requires the load-test client itself not to become the bottleneck.
- **Coordinated omission**: under closed-loop or naive retry logic, when a request gets stuck the load tool tends to "forget" to keep sending follow-ups on the original cadence, so the sample systematically misses the slowest cases — the measured p99 comes out far more optimistic than reality. The fix is computing latency against **scheduled arrival times** rather than **actual send times**.
- **Realistic input/output length distributions**: use real traces (or at least synthetic distributions matching the real ones) instead of fixed lengths, because the prefill/decode ratio, KV occupancy, and batch composition all depend strongly on the length distribution; fixed-length load tests deviate systematically from production behavior.
- **Load sweep and the saturation point**: numbers measured at a single concurrency/arrival rate are meaningless; sweep a curve from idle to saturation, find the knee where the throughput–latency curve turns steep (the saturation point), and state explicitly "which latency this throughput number corresponds to".
- **Report cold start separately, and mind p99 sample size**: the cold-start tail must be reported apart from the steady-state distribution (§1.5), or scale-up jitter gets misread as a chronic problem; tail quantiles like p99 are very noisy at small sample sizes, so use enough samples and state confidence intervals.

The full input list for replica sizing (goodput-based replica-count calculation) is in Q19.

## §3 Logits to Tokens: Sampling, Constrained Decoding, and Streaming

### 3.1　The sampling pipeline and a quick-reference table

**The execution order of logits processors is part of the API semantics, and frameworks do not agree on it.** This tutorial fixes the teaching pipeline below (for exposition only — it mandates no framework's order; notation: raw logits $z_i$, occurrence count $c_i$ of a token):

$$\text{hard constraints/logit bias} \to \text{repetition/frequency/presence penalties} \to \text{temperature} \to \text{truncation filters} \to \text{renormalize} \to \text{sample}$$

Quick-reference table for the four most common operators ($p_i=\text{softmax}(z_i/T)$; every filter renormalizes within its kept set):

| Operator | Precise definition | Key pitfall |
| --- | --- | --- |
| **Temperature** | $p_i(T)=\exp(z_i/T)/\sum_j \exp(z_j/T)$, defined only for $T>0$; with finite logits and no extra mask the support is the full vocabulary | `T=0` must be implemented as a separate greedy branch ($y_t=\arg\max_i z_i$; tied maxima need a declared deterministic tie-break — this tutorial takes the lowest index); never actually compute $z_i/0$; the tied $T\to0^+$ limit is in §A [D01] |
| **Top-k** | Support = the $k$ tokens with the largest logits ($1\le k\le \vert E\vert$, $E$ the set of legal tokens with finite logits this step) | Positive temperature scaling preserves ranking, so top-k alone has a support unchanged for any $T>0$; ties at rank $k$ make the support non-unique — declare a tie-break (this tutorial sorts by $(-z_i,\text{token\_id})$ and takes the first $k$) |
| **Top-p** | Sort by probability descending; take the smallest prefix whose cumulative probability **first reaches $\ge p_0$**; the boundary token must be kept | Both intuitive definitions — "keep all $p_i\ge p_0$" and "cumulative strictly $<p_0$" — wrongly drop the boundary token; [D01] nails it with $p=[0.4,0.3,0.2,0.1]$ and threshold $0.65$: the correct support is the first two tokens (cumulative $0.7\ge0.65$), while "strictly less than" stops wrongly at the first token (cumulative $0.4$) |
| **Min-p** | $S=\{i:\ p_i \ge \alpha \max_j p_j\}$, equivalent to $z_i \ge z_{\max} + T\log\alpha$ | The support **changes with temperature** — not a fixed logit-gap threshold: for $\alpha<1$, larger $T$ lowers the threshold and enlarges the support ([D01] sweeps several $T$ to verify monotone non-decrease) |

Stacked truncation filters **do not always commute** — each filter renormalizes within its kept set, so the next one no longer sees the original distribution: `top_k(3)` then `top_p(0.8)` gives a different support from the reverse order; numerical counterexample in Q11 and §8 [D02].

<details>
<summary>Other samplers: typical sampling and beam search</summary>

**Typical sampling**: $H(p)=-\sum_i p_i\log p_i$, $d_i=|-\log p_i - H(p)|$; sort by $d_i$ ascending (tokens whose information content is closest to the overall entropy come first), take the smallest prefix whose cumulative probability reaches `typical_p`, renormalize. **Not equivalent to top-p** — the support need not be a contiguous prefix of the probability ranking: [D01] constructs $p=[0.02,0.35,0.33,0.30]$, where `typical_p=0.63` selects the 2nd- and 3rd-highest-probability tokens while excluding the highest one (0.35).

**Beam search**: maintains a candidate set of width $B$, approximately maximizing the sequence-level $\log P(y_{1:T})=\sum_t \log p(y_t\mid y_{<t},x)$; stepwise pruning **does not guarantee global optimality**; length normalization is not unique (a common form is $S(y)=\log P(y)/|y|^{\alpha}$ — do not treat it as universal). It solves a sequence-level search problem, not "lack of diversity" — in dialogue generation it can actually favor generic, repetitive, or prematurely ended text.

</details>

### 3.2　Penalties: a penalty is not "subtract the same constant"

$$\text{Frequency/presence penalty}:\quad z_i' = z_i - \alpha_f c_i - \alpha_p\mathbf{1}[c_i>0]$$

The frequency term grows linearly with occurrence count; the presence term only cares "has it appeared or not"; whether `counts` includes prompt tokens (versus only the generated part) is decided by each API's semantics — tutorials/frameworks should state it explicitly.

$$\text{Sign-aware repetition penalty}\ (r>1):\quad z_i' = \begin{cases} z_i/r & z_i>0 \\ r\,z_i & z_i<0\end{cases}\qquad\text{applies only to tokens that have appeared}$$

It is **not** subtracting one constant from every seen token — the scaling direction depends on sign (positive logits get pushed down, negative logits get pushed further down).

### 3.3　Constrained decoding and best-of-n

JSON-schema, regex, and FSM-driven constrained decoding is essentially **setting the logits of currently illegal tokens to $-\infty$ at every step, then doing softmax + sampling as usual** (the `apply_hard_mask` in §8 [D01] demonstrates this primitive). **Bad-word/banned-sequence** (forbidding certain token sequences from reappearing), **forced tokens** (leaving finite logits only on designated tokens), and **no-repeat n-gram** are all special cases of the same mask primitive — the actual mask timing depends on **the state of the generated suffix so far**, and permanently banning a single token is just a banned sequence of length 1. The genuinely messy parts — automaton state transitions (FSM/PDA), tokenizer-aware matching, dead-end handling — are in Q9.

**Best-of-n / parallel sampling** cannot be treated as an ordinary sampling parameter: the $n$ candidates share the KV of the same prompt (copy-on-write, §4.2), and only the generated suffixes grow roughly $n$-fold — the full KV-occupancy and scheduling-accounting version is in Q24.

### 3.4　Stop conditions and streaming boundaries

Stop conditions must at least distinguish: **the EOS token**, **stop-token sequences**, **stop strings** (string-level, possibly straddling multiple token boundaries), `max_new_tokens`, **the total context limit**, **the constrained decoder's accept state** (e.g. JSON syntactically closed), **deadlines**, and **client cancellation**.

When a stop string straddles token boundaries, the server must **buffer a span of uncommitted text that "may still be a stop-string prefix"** before pushing to the client, and emit safely only once no further stop-string match is possible (§8 [D03] gives a minimal implementation). This belongs to the same "unstable boundary" family as the tokenization tutorial's **prefix instability**, but at a different layer — one at the tokenizer's encode boundary, the other at streaming's string-match boundary; the former, plus the full token-healing mechanism, is in [`tokenization_tutorial.md`](tokenization_tutorial.md) §5.4.

## §4 The KV Lifecycle: Allocation, Paging, Sharing, and Routing

### 4.1　Capacity accounting: plugging the sibling tutorial's known quantity into serving

Per-token KV bytes directly reuse the conclusion of [`kv_cache_speculative_decoding_tutorial.md`](kv_cache_speculative_decoding_tutorial.md) §2.1:

$$m_{\text{token}} = 2\, N_{\text{layer}}\, N_{\text{kv\_head}}\, d_{\text{head}}\, b$$

(the factor 2 is K/V; $b$ is bytes per element; this is the **ideal payload**, excluding allocator alignment, block-table metadata, quantization scales, and other overheads — the MQA/GQA/MLA derivations are in that tutorial). This tutorial treats $m_{\text{token}}$ as an externally known parameter and cares only about three things once it enters the serving lifecycle: block allocation and internal fragmentation, cross-request sharing and copy-on-write (§4.2), and reclamation under preemption/eviction (§4.4) — [D07]–[D08] hook this known quantity into those three things without re-verifying the formula itself.

### 4.2　PagedAttention: a block table, not automatic swap-out to disk

The core mechanism (full design in [`kv_cache_speculative_decoding_tutorial.md`](kv_cache_speculative_decoding_tutorial.md) §5; this section covers only the serving half): each sequence's logical KV blocks map through a **block table** to non-contiguous physical blocks in GPU memory, and the attention kernel reads KV via this table's indirection. Fixed-size blocks remove the requirement that "each request must reserve one contiguous max-length KV span", cutting external fragmentation and over-reservation; but **the last partially filled block still carries internal fragmentation** (the allocator demo in §8 [D07]).

> ⚠️ **PagedAttention does not mean "KV auto-loads from disk on a page fault"** — OS paging is only a design-inspiration analogy; PagedAttention's core is the **GPU attention kernel plus the block-table indirection mechanism**, managing the mapping and sharing of non-contiguous physical blocks within GPU memory. That does not seal the serving system off from disk/host memory — actually "moving out of GPU memory" is a separate mechanism that can stack on top of it: host swap / KV offload (moving KV to CPU memory or even slower storage), one of the cost options under §4.4's preemption — it is just **not PagedAttention's own default behavior**.

Physical blocks can be shared across common prefixes or beam branches via reference counting; when a shared writable tail block needs to diverge, the usual trigger is **copy-on-write** (copy the block and rewrite the mapping, rather than mutating the shared content in place).

### 4.3　Prefix caching and multi-tenant correctness

Prefix caching reuses the KV of an **exact token prefix**; the core criterion is "**token-ID sequences equal position by position + every configuration that produced the KV is compatible**" — e.g. same model version/weights, same adapter (LoRA etc.), same position encoding and attention configuration, same KV dtype/memory layout, same parallel sharding.

> ⚠️ **"Semantically similar prompts" cannot share KV** — prefix caching only cuts the cost of **repeated prefill**; it does not reduce prefill work on the missed suffix, nor any subsequent decode work; it is not a semantic cache, nor retrieval augmentation.

In multi-tenant settings, the prefix cache is a genuine security-boundary problem: the cache's **hash-key design, access permissions, and eviction policy** must all prevent tenants from using side channels like "did the cache hit" or "response-latency differences" to infer each other's private prefix contents — or worse, directly reusing cached content they should not access.

### 4.4　Eviction and preemption

For standard full causal attention, **evicting any old KV forfeits the guarantee of equivalence with "the exact computation path that keeps the full history KV"** — the evicted position should have participated in every subsequent query's attention, so dropping it introduces an approximation never validated in training. Safe sliding-window eviction requires the model itself to train/infer with windowed attention, or an explicit acceptance of the semantic shift brought by **approximate methods** like attention sinks or compression — eviction is a priced engineering choice, not a lossless optimization.

Common preemption approaches differ both in cost and in whether they actually relieve GPU-memory pressure — this is a common taxonomy, not an exhaustive one:

- **Discard KV and recompute** (recompute): truly frees physical KV blocks, at the cost of extra recomputation compute;
- **Migrate to host memory** (swap): also truly frees GPU memory, at the cost of extra transfer (growing with KV size);
- **Pause but keep KV (descheduling)**: merely removes this request from the current round of scheduling; it skips this iteration but **keeps holding all its physical KV blocks**. This one **should not be lumped with the other two as the same class of "memory reclamation"**: what it frees is compute/scheduling share, and it cannot relieve KV memory pressure; if the goal is to free KV space for other requests, descheduling alone does not help — it must pair with recompute or swap.

Which one (or which combination) to pick depends on the concrete trade-off of recompute cost vs. transfer bandwidth vs. memory pressure — there is no universally optimal answer.

The triple impact of long-generation requests — stretching mixed-batch iterations, occupying KV capacity long-term, raising the probability of triggering preemption — unfolds fully in Q21.

### 4.5　Cache-aware routing

With multiple replicas, choosing which replica to route to cannot just chase "the highest prefix-cache hit rate":

> 🎯 **Cache-aware routing's correct objective is minimizing expected completion time or SLO-violation probability, not maximizing hit rate** — the highest-hit replica can still be slower than a no-hit but idle replica if its queue is already deep. The routing decision must jointly weigh current queue depth, expected KV occupancy, the prefill savings from prefix-cache hits, and failure-domain isolation — not greedily optimize a single metric.

## §5 Scheduling: Scheduler Ticks, Batching, and Admission

### 5.1　A scheduler iteration in full: what one tick does

Each scheduler tick (one iteration) is the hub connecting admission, KV, prefill, decode, sampling, and streaming — four core steps:

1. **Reclaim and refill**: reclaim last round's terminal requests per §1.7's cleanup protocol (freeing batch slots and KV blocks), then refill the active set from QUEUED/DEGRADED per the scheduling policy (§1.3) — provided `max_num_seqs`, physical KV capacity, and the already-reserved shares line up; the selected requests convert from "reserved" to physically allocated (QUEUED → RUNNABLE).
2. **Allocate the budget, form the mixed batch**: `max_batched_tokens` is **this iteration's** token-work budget (not a whole-lifetime reservation for the request — that was computed at admission, §1.2), split between prefill chunks (§5.4) and 1 decode token per active request.
3. **Execute and emit**: run one mixed forward containing both prefill chunks and decode tokens (kernel-level differences at the end of §2.3), apply §3's sampling pipeline to the logits to produce the next token, and hand it to streaming/stop-string detection (§3.4); if the client cannot keep up, the request goes BACKPRESSURED (§0.2) instead of buffering without bound.
4. **Update states**: check termination conditions (EOS/stop/length/deadline/cancel) and mark terminal states for next round's reclaim; under excessive KV pressure or to make room for higher priority, run preemption per §4.4 (recompute/swap/descheduling — only the first two truly free GPU memory).

Speculative decoding adds "a variable number of accepted tokens after verification + KV rollback of rejected candidates" to step 3; its accounting impact on this tick is in §7.

### 5.2　Static batching's real limitation

> ⚠️ **Static batching's key limitation is not "all requests must finish before any output"** — completed sequences emit their own tokens immediately (no waiting for others); the real limitation is that **the batch slots they leave behind usually cannot be refilled by new requests before the whole batch ends** — the slot sits empty until the longest request in the batch finishes and the scheduler forms the next batch.

### 5.3　Continuous (in-flight) batching

Continuous batching removes finished requests and admits new ones into the vacated slots at **iteration boundaries**; the main gains come from **less decode padding, fewer empty slots, and less head-of-line blocking**.

> ⚠️ **Continuous batching does not guarantee lower per-request latency under every workload** — when all requests are similar in length, continuous and static have almost no empty slots to save, and their makespans can be identical (the equal-length counterexample in §8 [D04]).

### 5.4　Chunked prefill

Splitting a long prompt's prefill into multiple fixed-size chunks lets the scheduler interleave decode work between chunks, capping the token work of a single iteration and preventing one long prefill from monopolizing a whole iteration (head-of-line blocking).

> ⚠️ **Chunked prefill usually improves decode ITL and fairness under mixed load, but can increase the long prompt's own TTFT** — its own prefill work is spread over more iterations, each getting a smaller prefill budget. The concrete outcome depends on the decode-priority policy, chunk size, and kernel batching efficiency — [D05] in §8 computes both opposing effects at once with a simplified token-budget model. The "decode gap" there measures the **decode service gap** — how many iterations pass between two servings of a decode request — a **proxy** for ITL, not ITL itself; only under the simplifying assumption "every iteration takes roughly equal time" does a service gap dropping from 3 iterations to 1 correspond to lower ITL — in real systems each iteration's actual duration varies with batch composition and the prefill/decode mix, so the proxy and true ITL are not a strict linear conversion.

### 5.5　Admission control and the joint KV/token-budget constraint

At the scheduler-implementation level, admission jointly maintains four constraints — `max_num_seqs`, `max_batched_tokens` (the iteration-level budget, §5.1), remaining KV block capacity, and tenant quotas — hitting any one means reject or queue, and the decision only affects whether the "reservation" succeeds; physical allocation waits for the moment a tick selects the request. The full two-stage admission is in §1.2.

## §6 From One Machine to a Cluster: Replicas, Parallelism, and Prefill/Decode Disaggregation

### 6.1　Replica routing

Routing decisions across multiple replicas must trade off **current queue depth, expected KV occupancy, prefix-cache locality, and failure domains** — the cache-aware full version is in §4.5.

### 6.2　TP/PP/EP: the forward-only inference view

> ✅ **When the model fits on one GPU and the goal is aggregate throughput, data-parallel replicas are usually the first-choice baseline** — no per-request cross-GPU communication, plus free routing on load and prefix locality (§4.5, §6.1). Whether TP/PP/EP helps must be measured against latency targets, batch size, and interconnect bandwidth — **"the model doesn't fit" is the most common hard constraint, but not the only reason to shard**.

Inference has no backward pass and no optimizer state, but that does not make parallel communication disappear — communication-volume analysis and the full derivations of backward/ZeRO/FSDP/training pipeline schedules are in [`distributed_training_tutorial.md`](distributed_training_tutorial.md); here are only the three inference-specific points:

- **TP**: every layer's forward still needs collective communication (e.g. the all-reduce after row-parallel); **at small decode batches, per-token communication latency easily outweighs the compute savings**.
- **EP (MoE)**: expert parallelism at inference still needs token dispatch/combine communication and faces expert load imbalance; **no backward does not remove all-to-all latency, hot experts, or small-message communication overhead**; EP usage is driven mainly by the MoE's expert placement.
- **PP**: for single requests and low concurrency, PP **cannot automatically lower latency**, since end-to-end still traverses all stages; it takes enough concurrent sequences or micro-batches to fill the pipeline, otherwise most time is spent waiting on bubbles.

### 6.3　Prefill/Decode Disaggregation

The motivation for splitting prefill and decode onto different nodes/resource pools: the two phases have very different resource profiles (prefill compute-heavy, decode bandwidth-heavy), they can scale independently, and it reduces interference from long prefills with same-node decode ITL.

> ⚠️ **Disaggregation is not an unconditional speedup** — the gains must exceed: cross-node KV-transfer overhead, extra queueing and metadata coordination, load imbalance, and failure-recovery complexity. With short prompts (KV transfer large relative to total gain) or weak interconnects, disaggregation can actually be slower — it is an engineering decision requiring a measured break-even point, not a default-better architecture. Prefix-cache hits shorten effective prefill and shift the resource ratio between the prefill and decode sides, further changing how that break-even should be computed.

## §7 Combining Optimizations and Bottleneck Migration

Each optimization alone has a clean win story, but **combined, the gains neither multiply nor add directly** — together they change batch size, memory headroom, kernel shapes, and the location of the bottleneck itself:

- **Quantization**: weight quantization mainly relieves weight bandwidth and capacity pressure; **KV-cache quantization is an independent design point** — the model using low-bit weights does not entitle the KV to the same bit width as a matter of course (numeric sensitivity and per-channel/per-token granularity choices differ completely). Numeric details (scale/zero-point, concrete GPTQ/AWQ/FP8 error analysis) are in [`quantization_tutorial.md`](quantization_tutorial.md); this section gives only the decision dimensions:

  | Quantized object | Main effect | Coupling point with serving |
  | --- | --- | --- |
  | Weights | Bandwidth, memory capacity | Moves the flip point of decode arithmetic intensity (§2.3) |
  | KV cache | Memory capacity, allowed batch/context ceiling | Affects §4's capacity accounting and block allocation |
  | Activations | Kernel support, numeric stability | Determines which fused kernels are usable and prefill's real throughput |

- **Speculative decoding**: acceptance rate and the $E[\tau]$ expected-speedup derivation are in [`kv_cache_speculative_decoding_tutorial.md`](kv_cache_speculative_decoding_tutorial.md) §7; this tutorial stresses only the serving accounting: each target verification processes multiple candidate tokens and yields a **variable number** of accepted tokens, so the scheduler must account separately for "verified tokens", "output tokens", and "KV delta" — never treat it as fixed-stride decode. Quantizing the target or the draft can shift the draft/target distribution gap and the acceptance rate through numeric error — **speculative decoding and quantization are not strictly orthogonal**.
- **Prefix caching × continuous batching × disaggregation**: a prefix hit shortens the effective prefill length, which changes the prefill pool's actual workload in a disaggregated deployment and thus the ratio at which the two resource pools should be provisioned — the three are mutually coupled, and evaluating any one in isolation yields misleading conclusions.

> 🎯 **Online symptom → localization path (diagnostic table)**

| Symptom | Check first |
| --- | --- |
| High TTFT | Queueing time, admission rejection rate, prefix-cache hit rate, whether prefill is being squeezed by chunking |
| TPOT/ITL jitter | Long prefills cutting in, preemption frequency, whether batch composition churns frequently |
| KV OOM | Block-allocation fragmentation, long-generation requests' occupancy, admission's KV reservation policy |
| Low cache hit rate | Hash-key design, whether tenant-isolation policy over-partitions, whether routing destroys locality |
| Low GPU utilization but latency still bad | First check whether batches are too small/fragmented rather than rushing to add GPUs — usually a scheduling or admission problem |
| p99 suddenly degrades | First split queueing time vs. service time; then instrument stage by stage: queue → cache hit → prefill → decode → KV pressure → preemption → kernel/collective → network → streaming (cache hit affects whether/how much prefill runs, so it precedes prefill); do not stare at GPU utilization alone |

## §8 From-Scratch Implementations: Demo Index

The full runnable script is [`code/inference_serving.py`](code/inference_serving.py) (pure standard library, no third-party dependencies, [D01]–[D09] finish in seconds on CPU); each demo verifies one isolated mechanism:

| Demo | What it verifies | Section |
| --- | --- | --- |
| [D01]–[D02] | Exact support sets of sampling filters, the $T\to0^+$ limit, constraint masks; filter-order non-commutativity | §3.1–§3.3 |
| [D03] | Streamed buffering of stop strings across tokens | §3.4 |
| [D04]–[D05] | Continuous batching (with the equal-length counterexample) and the chunked-prefill trade-off | §5.2–§5.4 |
| [D06]–[D08] | KV per-token bytes in practice, block-allocation internal fragmentation, prefix sharing | §4.1–§4.3 |
| [D09] | Roofline ridge point (upper-bound reference, not a latency predictor) | §2.3 |

Key assertion snippets are folded below; the full invariant list and the script's real output are in §A.

<details>
<summary>(a) Sampling: key assertion snippets</summary>

Split into "compute filtered probabilities" (`top_k_filter`/`top_p_filter`/`min_p_filter`/`typical_filter` — pure functions that never touch randomness) and "draw" (`sample`, the only function that calls the RNG).

```python
# top-p: smallest sorted prefix whose cumulative probability first reaches or exceeds the threshold; keep the boundary token
p_ex = [0.4, 0.3, 0.2, 0.1]
filt = top_p_filter(p_ex, 0.65)
support = {i for i, p in enumerate(filt) if p > 0}
assert support == {0, 1}                                   # cumulative 0.7 >= 0.65
assert sum([p_ex[0]]) < 0.65                                # "strictly below p0" would drop the boundary token

# min-p support changes with temperature (not a fixed logit-gap threshold)
sizes = [len(min_p_support_from_logits(logits_t, T, alpha=0.3))
         for T in (0.2, 0.5, 1.0, 2.0, 5.0)]
assert sizes == sorted(sizes) and sizes[0] < sizes[-1]      # larger T, larger support

# filter order is not commutative
support_a = support_of(top_p_filter(top_k_filter(probs, 3), 0.8))   # top_k -> top_p
support_b = support_of(top_k_filter(top_p_filter(probs, 0.8), 3))   # top_p -> top_k
assert support_a != support_b                               # {0,1} vs {0,1,2}
```

</details>

<details>
<summary>(b) Continuous-batching simulation: key assertion snippets</summary>

Three requests A/B/C arrive simultaneously, capacity=2, needing 1/4/1 decode tokens respectively; each active request emits 1 token per tick:

```python
reqs = [Request("A", 0, 1), Request("B", 0, 4), Request("C", 0, 1)]
comp_static, _  = simulate_batching(reqs, capacity=2, policy="static")
comp_cont,   _  = simulate_batching(reqs, capacity=2, policy="continuous")
assert comp_static == {"A": 1, "B": 4, "C": 5}    # C cannot start until t=4
assert comp_cont   == {"A": 1, "C": 2, "B": 4}    # C slots in at t=1 right after A finishes

# Note: "utilization" here is scheduler-slot utilization (total serviced tokens /
# (capacity * makespan)) — a capacity metric of this unit-cost toy model, not GPU
# SM/HBM utilization; in real systems each iteration's actual duration varies with
# batch composition and the prefill/decode mix, unlike the equal-time ticks assumed here.
u_static, _ = utilization(comp_static, reqs, 2)   # slot utilization: 6 / (2*5) = 0.60
u_cont,   _ = utilization(comp_cont,   reqs, 2)   # slot utilization: 6 / (2*4) = 0.75
```

The equal-length counterexample (two requests both needing 3 ticks) verifies `comp_static_eq == comp_cont_eq` — continuous batching does **not always** strictly dominate. The scheduling simulator checks uniformly across all policies: total generated tokens conserved, no running before arrival, active count within capacity, immediate release on completion (no lingering KV).

</details>

<details>
<summary>(c) KV block allocation and sharing: key assertion snippets</summary>

Per-token bytes directly take the sibling known quantity already stated in §4.1 ($m_{\text{token}}=64$ bytes is just that quantity instantiated with one example configuration, not a new conclusion this tutorial verifies); what this section actually cares about is three things after that number enters the serving side — the block allocator reports both logical and allocated figures (internal fragmentation); the prefix-sharing scenario verifies the physical block count drops after sharing; and the explicit check that "KV heads not divisible by the TP degree must not be naively divided":

```python
m_token = kv_bytes_per_token(n_layer=2, n_kv_head=2, d_head=4, bytes_per_elem=2)
assert m_token == 64                                        # 2*2*2*4*2 bytes/token

logical_slots, allocated_slots = block_alloc_slots([1, 5], block_size=4)
assert (logical_slots, allocated_slots) == (6, 12)          # internal fragmentation: 12 > 6

no_share = physical_blocks_no_sharing([6, 7], block_size=4)  # 4
with_share = no_share - 1                                    # sharing one full block -> 3
```

</details>

## §9 25 High-Frequency Interview Questions

Three difficulty tiers; organize answers as "framework → key formulas/definitions → common mistakes", without repeating the main text's derivations.

### L1 Must-know

<details>
<summary>Q1. What stages does a request go through from arrival to its first token?</summary>

- ingress gate → tokenize/validate → prefix lookup → scheduler admission + KV reservation → queued/runnable → prefill → first-token sampling (directly from the last prefill forward's logits) → **detokenize + the first real emit to the client** (§1, §0.2); "getting the first token" means the first emit observed client-side, and the detokenization and network transfer in between usually belong inside TTFT
- Any stage can be interrupted by client cancel/disconnect, and KV must be reclaimed immediately (§1.7)
- The first requests after a replica boots have TTFT with a cold-start tail (model loading/kernel autotune/graph capture); benchmarks must report it separately from steady state (§1.5, §2.4)

**Mistakes**: assuming "prefill finishes, then a separate decode run produces the first token" is the default implementation; omitting two-stage admission and the first-emit step.

</details>

<details>
<summary>Q2. How are TTFT/ITL/TPOT/throughput/goodput each defined?</summary>

- $TTFT=t_{\text{first emit}}-t_{\text{arrival}}$, $ITL_j=t_{\text{emit},j}-t_{\text{emit},j-1}$, $TPOT=\frac{t_{\text{last emit}}-t_{\text{first emit}}}{N_{\text{out}}-1}$ (§2.1)
- Throughput is a **rate** (declare which of request / output-token / total-token first); **SLO attainment** ($A=N_{\text{SLO}}/N_{\text{total}}$) is a ratio; **goodput** ($G=N_{\text{SLO}}/\Delta t$) is a rate — different dimensions, never conflate (§2.2)
- Declare the system boundary: whether queueing/tokenization/network transfer is included; report mean or p95/p99

**Mistakes**: conflating TPOT with "total latency / token count"; calling goodput "the fraction of requests meeting the SLO" (that is SLO attainment).

</details>

<details>
<summary>Q3. Why do prefill and decode usually have different bottlenecks? Is that always the case?</summary>

- Prefill: $L$ tokens computed in one pass with high weight reuse, usually compute-bound; decode: repeatedly re-reads weights + KV at small batch, usually bandwidth-bound (§2.3)
- **Not a theorem**: at large decode batch, arithmetic intensity rises and it can turn compute-bound again; quantization moves the flip point; at ultra-long context, prefill can also be IO/kernel-limited

**Mistakes**: treating the empirical rule as a universal theorem, with no counterexample to offer.

</details>

<details>
<summary>Q4. What exactly are the support sets of top-k/top-p/min-p/temperature?</summary>

- top-k: the $k$ largest logits ($1\le k\le\vert E\vert$, $E$ the legal finite-logit set this step; ties at rank $k$ broken by $(-z_i,\text{token\_id})$) (§3.1)
- top-p: the smallest sorted prefix whose cumulative probability **first reaches or exceeds the threshold** (keep the boundary token) — both common definitions "$p_i\ge p_0$" and "cumulative strictly $<p_0$" wrongly drop the boundary token
- min-p: $p_i\ge\alpha\max_j p_j$, equivalent logit threshold $z_i\ge z_{\max}+T\log\alpha$; the support changes with $T$, not a fixed logit gap
- temperature: with finite logits and no extra mask, the support for any $T>0$ is the **full vocabulary**; $T=0$ is not on this curve but a separate greedy branch (unique after tie-break; the tied limit is in §A [D01])

**Mistakes**: misremembering "first reaches" as "first strictly exceeds"; thinking the min-p threshold is temperature-independent; forgetting to answer temperature's own support set.

</details>

<details>
<summary>Q5. What problem does PagedAttention actually solve?</summary>

- Logical KV blocks map through a block table to non-contiguous physical blocks, removing the "reserve one contiguous max-length span" requirement and cutting fragmentation (§4.2)
- Supports sharing (prefixes/beam branches) + copy-on-write
- It does **not** lower attention's computational complexity, and it is **not** "page faults auto-loading from disk" — that is a separate host swap/offload mechanism

**Mistakes**: answering "it lowered attention's computational complexity", or conflating it with disk paging.

</details>

<details>
<summary>Q6. What is static batching's key limitation?</summary>

- It is **not** "all requests must finish before any output" — finished requests emit their results immediately
- The real limitation: slots left by finished sequences **cannot be refilled by new requests before the whole batch ends** (§5.2)

**Mistakes**: answering "must wait for everyone", unable to state the sharper "slots cannot be refilled" formulation.

</details>

<details>
<summary>Q7. Why is continuous batching more efficient? Is it always faster?</summary>

- Removes finished requests and admits new ones at iteration level; the main gains are fewer empty slots/less padding/less head-of-line blocking (§5.3)
- **No guarantee of lower latency for every workload**: with similar request lengths, static and continuous can have identical makespans (§8 [D04] counterexample)

**Mistakes**: only saying "faster", with no equal-length counterexample where the two tie.

</details>

<details>
<summary>Q8. What does prefix caching reuse? What are its limits?</summary>

- Reuses the KV of an **exact token prefix**; must simultaneously match token IDs, model version, adapter, position-encoding state, attention configuration, KV dtype/layout, and parallel sharding (§4.3)
- Only cuts repeated prefill; does not reduce the missed suffix or any subsequent decode work
- In multi-tenant settings, the hash key/access permissions/eviction policy must prevent cross-tenant side channels

**Mistakes**: believing "semantically similar prompts" can also hit; not knowing a hit saves only prefill while decode work stays unchanged.

</details>

### L2 Advanced

<details>
<summary>Q9. How is constrained decoding (JSON/schema/regex) implemented?</summary>

- Essence: at every step, set currently illegal tokens' logits to $-\infty$, then softmax + sample as usual (§3.3); "currently illegal" is decided by an automaton/parser that tracks the generated text — an FSM (regex/simple schemas), a PDA (nested structures like JSON's bracket matching), or an incremental grammar parser
- The legal-token set is **tokenizer-aware**: a candidate next token may span multiple characters and must be matched character by character against the current automaton state — string-level checks alone do not suffice
- **Dead-ends** must be handled: a token can be legal right now yet drive the automaton into a state from which no accept state is ever reachable — a full implementation either prunes with lookahead or explicitly declares this edge case unhandled

**Mistakes**: presenting constrained decoding as needing brand-new sampling theory, or stopping at the one-liner "mask illegal tokens" without naming automaton state/tokenizer-aware matching/dead-ends — the genuinely messy parts.

</details>

<details>
<summary>Q10. How do you run a trustworthy serving benchmark?</summary>

- First separate **open-loop** (keep sending at a fixed arrival rate) from **closed-loop** (fixed concurrency) — closed-loop automatically lowers the effective arrival rate as the system slows, easily squeezing "falsely stable" numbers out of an overloaded system
- Beware **coordinated omission**: compute latency against scheduled arrival times instead of actual send times, or the slowest samples get systematically dropped and p99 looks more optimistic than reality
- Use real (or realistically matched) input/output length distributions + a **load sweep** (report the throughput–latency curve and the saturation knee, not single-point throughput); report cold start separately from steady state, and give p99 enough samples (§2.4)

**Mistakes**: just "run a load script and measure throughput", unable to explain the open/closed-loop difference or coordinated omission's distortion mechanism.

</details>

<details>
<summary>Q11. Why does the execution order of sampling filters matter? Give a non-commuting example</summary>

- top-p depends on "whether the previous step already renormalized" — the cumulative-probability base differs across orders (§3.1); every filter renormalizes within its kept set, so the next one no longer sees the original distribution
- Concrete counterexample (§8 [D02], input distribution $p=[0.5,0.2,0.15,0.1,0.05]$): `top_k(3)` first truncates the support to the 3 highest-probability tokens and renormalizes, then `top_p(0.8)` gives support size 2; the reverse — `top_p(0.8)` first (accumulating to 0.8 on the original distribution) then `top_k(3)` — gives support size 3. Same distribution, different order, different final support

**Mistakes**: only saying "order can matter" without a concrete counterexample, and forgetting "renormalization at every step" — the root cause of the order sensitivity.

</details>

<details>
<summary>Q12. What does chunked prefill solve? What does it cost?</summary>

- Chunks a long prompt's prefill so it cannot monopolize iterations and stall same-batch decode (head-of-line blocking) (§5.4)
- Usually improves decode ITL and mixed-load fairness, but **can increase the long prompt's own TTFT** (§8 [D05] quantifies this trade-off with a token-budget toy model)

**Mistakes**: answering only "improves latency", missing the negative effect on the long prompt's own TTFT.

</details>

<details>
<summary>Q13. What signals should admission control decide on?</summary>

- Two stages: ingress gate (body/rate/quota, before tokenization) → scheduler admission + KV reservation (after tokenization) — full mechanism in §1.2; three over-limit responses: reject (REJECTED), queue (QUEUED, with a deadline), degrade (DEGRADED)
- **Prompt length is a definite value once tokenization completes**, so KV occupancy can be computed exactly; **output length can only be predicted until generation finishes** (heuristics/history), and prediction error is a major source of overload
- Joint constraints: KV headroom, `max_batched_tokens` (an **iteration-level** work budget, §5.1 — not a per-request reservation), max concurrent sequences, tenant quotas

**Mistakes**: only "look at GPU utilization"; conflating `max_batched_tokens` (the iteration-level budget) with per-request KV reservation.

</details>

<details>
<summary>Q14. What are the preemption implementations, and what does each cost?</summary>

- Common approaches (not an exhaustive taxonomy, §4.4): **discard KV and recompute** (extra compute; truly frees memory), **migrate to host (swap)** (extra transfer, cost growing with KV size; truly frees memory), **pause but keep KV (descheduling)** (keeps occupying memory; frees only compute/scheduling share)
- **Descheduling should not be lumped with the other two as "memory reclamation"**: it cannot relieve KV memory pressure, it just temporarily stops using compute — if the goal is freeing KV space, it must pair with recompute or swap
- Selection basis: the concrete trade-off of recompute cost vs. transfer bandwidth vs. memory pressure — no universal optimum

**Mistakes**: unable to name the three common paths and their costs; counting "pause but keep KV" as a memory-saving technique too.

</details>

<details>
<summary>Q15. What is the correct objective of cache-aware routing?</summary>

- The real objective is a combination of dimensions: minimize expected completion time/SLO-violation probability, maximize goodput, control cost, cross-tenant fairness (§4.5) — first make explicit which one (or which weighted mix) this routing decision optimizes
- The highest-hit replica with a deep queue can be slower than a no-hit but idle replica

**Mistakes**: treating "highest hit rate" or "shortest completion time" as the single exclusive routing objective, ignoring the cost and fairness dimensions.

</details>

<details>
<summary>Q16. How does speculative decoding account inside a serving iteration?</summary>

- draft→verify→accept/reject sits inside the decode iteration; the acceptance-rate/$E[\tau]$ derivations are in the sibling tutorial — this one covers only the serving view (§7)
- Each verification yields a **variable number** of accepted tokens; the scheduler must account separately for verified tokens, output tokens, and the KV delta — never as fixed-stride
- **Accepted, emitted, and committed KV are not always equal**: the KV deltas of rejected candidate tokens must be rolled back and cannot count toward this request's committed KV; the output side may also include a bonus/residual token (e.g. the extra free token from the target model after a successful verification), and whether it counts toward the "output token count" is decided by each implementation's billing semantics — it must be declared explicitly (§1.7)

**Mistakes**: treating spec decoding as a fixed-multiplier throughput gain; assuming generated tokens, truly committed KV, and finally billed output tokens are always the same thing.

</details>

<details>
<summary>Q17. Why doesn't the same random seed guarantee token-identical output across deployments?</summary>

- Under continuous batching the batch composition changes, so the kernels' batch shapes and reduce/all-reduce orders change with it — floating-point addition is not associative, so changed order changes the numbers (§1.6)
- **When multiple requests share one RNG stream, who consumes which segment of the stream first depends on scheduling order**, so "fixed seed" loses its meaning — the seed is only one necessary condition for bit-level reproduction (full list in §1.6's fold)

> Note: standard MoE **does not reroute merely because load changed** — only under concrete conditions like capacity limits/drops, or numeric perturbations flipping the top-k ranking, can load indirectly change which expert a token lands on.

**Mistakes**: believing a fixed seed reproduces token-for-token across environments; using "inherent to the system" as a universal shield — if batch composition, parallel topology, and kernel paths are all fixed and outputs still differ, suspect a real implementation bug.

</details>

### L3 Top-lab

<details>
<summary>Q18. When is prefill/decode disaggregation worth doing?</summary>

- Motivation: very different resource profiles across the two phases, independent scaling, less interference from long prefills on same-node decode ITL (§6.3)
- **Not an unconditional speedup**: gains must exceed the extra costs of cross-node KV transfer, queueing coordination, and failure recovery; short prompts/weak interconnects can make it slower
- Prefix-cache hits change the effective prefill length and thus the optimal ratio between the two pools

**Mistakes**: treating disaggregation as a default-better architecture without discussing break-even conditions.

</details>

<details>
<summary>Q19. How do you choose TP/PP/EP and the replica count?</summary>

- **Parallelism**: when the model fits on one GPU and the goal is aggregate throughput, data-parallel replicas are usually the first baseline, avoiding cross-GPU communication (§6.2); whether TP/PP/EP helps must be measured against latency targets, batch size, and interconnect bandwidth — "the model doesn't fit" is the most common but not the only motive; TP's per-token communication latency easily outweighs compute savings at small decode batches; PP cannot automatically lower latency at low concurrency and needs enough micro-batches to fill the pipeline; EP's no-backward does not remove all-to-all/hot-expert latency
- **How to size the replica count** (the real point of this question — parallelism alone is only half an answer):
  1. First determine a single replica's achievable **goodput** under the given SLO constraints (not raw throughput, §2.2), measured via §2.4's load sweep under the target latency constraint — never estimated from theory alone
  2. Divide the target arrival rate by single-replica goodput to get a theoretical lower bound on the replica count
  3. Layer on **peak headroom** — provision for expected peaks/bursts, not the average arrival rate
  4. Add **N+1 redundancy** so a single replica's failure/rolling upgrade does not overload the whole system
  5. Factor in **topology**: whether replicas spread across failure domains or share the same switches affects routing (§4.5, §6.1) and the actually usable aggregate goodput

**Mistakes**: only "big model, use TP/PP", unable to say these parallelisms can be slower at small batch/low concurrency; giving no concrete goodput-based sizing method.

</details>

<details>
<summary>Q20. Service p99 suddenly degrades — how do you localize it?</summary>

- First split latency into **queueing time** (waiting for admission/scheduling) and **service time** (actually running prefill/decode); quantile degradation is often a queueing-side problem, not individual requests getting slower
- Then instrument stage by stage in state-machine order: queue → cache hit → prefill → decode → KV pressure → preemption → kernel/collective → network → streaming (§7 diagnostic table; cache hit precedes prefill because it decides whether/how much prefill this request runs at all)
- Check whether the **workload mix** changed (input/output length distributions shifted suddenly, one tenant bursting lots of long prompts); with per-stage traces, directly compare which stage's share of time rose
- Check for **cold-start/autoscaling** effects (new replicas still cold after coming online, or a transient replica-count dip during scale-out)
- Audit whether the load test/monitoring itself has **coordinated omission** (measuring by actual processing time instead of scheduled arrival time systematically drops the slowest requests)
- Do not look only at GPU utilization — low utilization can also be a scheduling/admission problem, not a signal to add GPUs

**Mistakes**: GPU-utilization-only debugging with no staged methodology; placing cache hit after prefill; not separating queueing time from service time.

</details>

<details>
<summary>Q21. How does one long-generation request affect other requests' SLOs?</summary>

- Under continuous batching it **usually does not monopolize the whole batch**, but the impact is more than the single capacity dimension of "holding KV quota":
  1. The longer its context, the more history KV each of its decode steps reads and the more attention it computes, **directly stretching this mixed batch's iteration time** — other requests' ITL in the same batch degrades with it; this is a direct effect, not the indirect "quota crowding" one
  2. It **occupies KV blocks and active-sequence quota long-term**, pushing up memory pressure
  3. Rising memory pressure **raises the probability of triggering preemption**, which then affects even more requests (§4.4)
- The accurate statement is the superposition of these three — "directly stretching iterations + occupying capacity + raising preemption risk" — not a vague "capacity-level interference", and certainly not "blocks all requests"

**Mistakes**: over-strong claims like "one long request blocks all requests", or the opposite narrowing to just "occupies capacity", missing the mechanism of directly stretching other requests' ITL.

</details>

<details>
<summary>Q22. What are the security risks of a prefix cache under multi-tenancy?</summary>

- Bad hash-key design can let different tenants' prefixes collide or have their existence inferred; the response-latency difference between hit and miss is itself a side channel (§4.3)
- Hit checks need **tenant-namespace isolation** + **exact token verification** (a hash collision does not prove it is really the same prefix); the eviction policy likewise needs tenant isolation, or tenant A can evict or probe tenant B's sensitive prefixes
- "Encrypting the KV cache at rest" guards against **storage leakage at rest** and does not solve the core problem — cross-tenant **authorization isolation** and the **hit/latency side channel** are access-control and information-leakage issues, not a question of whether the data is encrypted

**Mistakes**: answering only "encrypted storage" as the full solution; unable to name the hit/latency side channel as the concrete mechanism.

</details>

<details>
<summary>Q23. Why can't you casually apply sliding-window eviction to standard full causal attention?</summary>

- Evicting any old KV makes the output **lose the guarantee of equivalence with "the exact computation path that keeps the full history KV"** — every query in standard attention may in principle depend on any historical position, so after dropping a position's KV the computation is no longer exact but an approximation never validated in training
- The safe versions: the model itself uses windowed attention in training/inference, or you explicitly accept the approximate semantic shift of attention-sink/compression methods, treating eviction as a priced engineering choice

**Mistakes**: assuming eviction is a "free optimization" losslessly stackable on any model.

</details>

<details>
<summary>Q24. What extra KV and scheduling costs do best-of-n/parallel sampling add?</summary>

- The $n$ candidates generate in parallel but **share the same prompt** — the prompt's KV can be shared as one copy via copy-on-write, so the whole thing need not scale by $n$; the better approximation is $M_{KV}\approx M_{\text{prompt}}+\sum_j M_{\text{branch},j}$ — only each branch's **generated-suffix** KV grows roughly with the branch count (§3.3, §4.2)
- Only when the implementation **lacks CoW sharing** does it degrade toward the more pessimistic bound $n(M_{\text{prompt}}+M_{\text{generated}})$; wall time also need not be $n$ times that of a single candidate — it depends on whether the candidates fit into the same decode iterations
- Admission/KV capacity accounting must include this increment, or the capacity misestimate surfaces only at runtime

**Mistakes**: treating best-of-n as a nearly free lightweight parameter, or misremembering "KV multiplies by n" while forgetting the prompt prefix can be shared.

</details>

<details>
<summary>Q25. When a long prompt arrives, how do you protect running online decode requests' SLOs?</summary>

- The toolbox: chunked prefill capping the per-iteration prefill budget, decode-priority scheduling, a dedicated prefill resource pool (evolving further into disaggregation), admission splitting traffic by predicted length (§5.4, §6.3)
- Trade-off: techniques that carve time slices within one resource pool (like chunked prefill) usually stretch the long prompt's own TTFT; a dedicated pool/disaggregation, sized well, can win on both — "protecting decode necessarily sacrifices prefill TTFT" is not a universally valid conclusion

**Mistakes**: answering only the single move "chunked prefill", or asserting "all these techniques stretch the long prompt's TTFT".

</details>

## §A Appendix: Sanity Check

Running `python3 code/inference_serving.py` (pure Python standard library, no third-party dependencies, seconds on CPU) should pass all checks [D01]–[D09]; the key invariant list and the real terminal output are folded below.

<details>
<summary>Key invariant list ([D01]–[D09])</summary>

1. **[D01] Sampling support sets**: greedy takes the lowest index on tied maxima; `temperature_probs` raises on $T\le0$; with a unique max logit, $T\to0^+$ concentrates on the argmax, and with tied max logits the mass splits evenly; the top-k support size is exactly $k$; top-p on $p=[0.4,0.3,0.2,0.1]$ with threshold $0.65$ gives the first two tokens as support (cumulative $0.7$); min-p always keeps the max-probability token, with support monotone non-decreasing as $\alpha\downarrow$ and as $T\uparrow$ (for $\alpha<1$); typical sampling excludes the highest-probability token in the constructed example; adding a constant to all logits leaves the min-p/top-k/top-p supports unchanged.
2. **[D02] Filter-order non-commutativity**: on the same distribution, `top_k->top_p` and `top_p->top_k` yield different supports ($\{0,1\}$ vs $\{0,1,2\}$).
3. **[D03] Streamed stop detection**: a stop string split across 3 token chunks is still detected, with no unsafe partial prefix ever leaked along the way; in the no-match case, all text is fully recoverable via flush.
4. **[D04] Batch scheduling**: static gives $A=1,B=4,C=5$ (utilization $0.60$), continuous gives $A=1,C=2,B=4$ (utilization $0.75$); in the equal-length counterexample both policies have the same makespan; the scheduler satisfies token-count conservation, no running before arrival, active count within capacity, release upon completion.
5. **[D05] The chunked-prefill trade-off**: max decode gap drops from 3 (unchunked) to 1 (chunked), but the long prompt's own TTFT rises from 3 to 4.
6. **[D07]–[D08] KV block allocation and sharing** (per-token bytes reuse the sibling known quantity already given in §4.1, $m_{\text{token}}=64$ bytes with $N_{\text{layer}}=2,N_{\text{kv\_head}}=2,d_{\text{head}}=4,b=2$, not separately verified in this tutorial): lengths $[1,5]$ with block size $4$ give logical/allocated slots $=6/12$ (i.e. $384/768$ bytes — internal fragmentation is real); lengths $[6,7]$ sharing 1 full block drop physical blocks from 4 to 3; KV heads not divisible by the TP degree refuse naive division.
7. **[D09] The roofline ridge point** serves only as an upper-bound reference, never as a latency predictor.

</details>

<details>
<summary>Real output of running <code>python3 code/inference_serving.py</code></summary>

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

(The above is real terminal output captured from re-running `code/inference_serving.py`, verbatim-identical to the current version of that script.)

</details>

## 📚 References

- **PagedAttention / vLLM** — Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention*, arXiv 2309.06180 (2023), SOSP 2023.
- **Orca (iteration-level / continuous batching)** — Yu et al., *Orca: A Distributed Serving System for Transformer-Based Generative Models*, arXiv 2204.05120 (2022), OSDI 2022.
- **Sarathi-Serve (chunked prefill)** — Agrawal et al., *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve*, arXiv 2403.02310 (2024), OSDI 2024.
- **DistServe (disaggregation, goodput)** — Zhong et al., *DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving*, arXiv 2401.09670 (2024), OSDI 2024.
- **Splitwise (phase splitting)** — Patel et al., *Splitwise: Efficient Generative LLM Inference Using Phase Splitting*, arXiv 2311.18677 (2023), ISCA 2024.
- **SGLang / RadixAttention (radix-tree prefix caching)** — Zheng et al., *SGLang: Efficient Execution of Structured Language Model Programs*, arXiv 2312.07104 (2023).
- **Nucleus sampling (top-p)** — Holtzman et al., *The Curious Case of Neural Text Degeneration*, arXiv 1904.09751 (2019), ICLR 2020.
- **Top-k sampling** — Fan, Lewis & Dauphin, *Hierarchical Neural Story Generation*, arXiv 1805.04833 (2018), ACL 2018.
- **Min-p sampling** — Nguyen et al., *Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs*, arXiv 2407.01082 (2024).
- **Locally typical sampling** — Meister et al., *Locally Typical Sampling*, arXiv 2202.00666 (2022), TACL 2023.
- **Speculative decoding** — Leviathan, Kalman & Matias, *Fast Inference from Transformers via Speculative Decoding*, arXiv 2211.17192 (2022), ICML 2023.
- **Speculative sampling** — Chen et al. (DeepMind), *Accelerating Large Language Model Decoding with Speculative Sampling*, arXiv 2302.01318 (2023).
- **FlashAttention (attention-kernel IO optimization — not the same technique as PagedAttention)** — Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*, arXiv 2205.14135 (2022), NeurIPS 2022.
- **Megatron-LM (TP)** — Shoeybi et al., *Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism*, arXiv 1909.08053 (2019).
- **Roofline model** — Williams, Waterman & Patterson, *Roofline: An Insightful Visual Performance Model for Multicore Architectures*, Communications of the ACM (2009) (no arXiv id).
- **Constrained decoding / grammar-constrained generation** — Willard & Louf, *Efficient Guided Generation for Large Language Models*, arXiv 2307.09702 (2023). This paper backs the concrete implementation of guided/grammar-constrained generation, but it should not be cited to argue that "every constrained-decoding implementation is essentially nothing but mask-and-renormalize" — §3.3's "essentially a mask" is an abstraction at the sampling-primitive level, not a restatement of that paper's scope.
