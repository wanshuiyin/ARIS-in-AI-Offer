## §0 Request State-Machine Mental Model + TL;DR Cheat Sheet

**LLM inference serving is not "run one forward pass" — it is a state machine spanning the gateway, scheduler, memory manager, and sampler.** The interview question everyone is sick of — "what's the difference between prefill and decode" — is really just two cells in this state machine. Without first building a mental model of the entire path, you can't say precisely where TTFT is getting stuck, which resource is alarming on a KV OOM, or where to start digging when p99 degrades. This tutorial's throughline is exactly this state machine; quantization, the KV-capacity formula, and the speculative-decoding derivation are already covered in other tutorials — here we only cover **where they plug into the serving pipeline and what they change about it**.

### 0.1　Happy path: a request that completes normally

```text
client request
  → ① ingress gate            (coarse body-size/rate/tenant-quota checks — fast rejection, no tokenize needed)
  → ② tokenize / validate      (see tokenization tutorial; chat template already applied upstream; exact prompt length is only known now)
  → ③ prefix lookup            (exact token-prefix match; hit reuses KV, miss range goes through normal prefill)
  → ④ scheduler admission + KV reservation
                                (jointly checks KV headroom/max_batched_tokens/concurrency/predicted output length;
                                 on pass, **reserves** (bookkeeping-level, not physical allocation) the KV quota)
  → ⑤ queued / runnable        (once reserved, enters the scheduling queue, waiting to be picked by some scheduler tick)
  → ⑥ prefill                  (one or more forward passes over the miss-range prompt tokens — may be split across multiple passes under chunked prefill)
  → ⑦ first-token sampling     (comes directly from the logits/hidden state of the last prefill forward pass —
                                 if the prefix is a full cache hit, the final logits must be cached separately, or at least the last prompt token must be recomputed;
                                 you cannot sample directly off K/V alone; there is no need to first run a "separate" decode iteration)
  → ⑧ per-step decode          (each step produces 1 new token, attention reads the full historical KV)
       ⑧a logits processing/sampling  (hard constraints → penalties → temperature → truncation filters → renormalize → sample, see §3)
       ⑧b detokenization/streaming (incremental emission; a stop string may span tokens, so the server must buffer)
  → ⑨ FINISHED                 (EOS / stop-string / max_new_tokens / context limit / deadline / client cancellation)
  → ⑩ KV reclaim + billing/logging
(side branches: REJECTED can occur at either admission check ①④; PREEMPTED/SWAPPED/RESUMED, BACKPRESSURED,
  TIMED_OUT/FAILED/CANCELLED are branches reachable from any running stage — see §0.2 for the full state diagram)
```

This happy path is only one trunk of the full state machine; what actually separates "understood it" from "memorized the diagram" in an interview is whether you can articulate the guard, held resources, and mandatory cleanup-on-exit for each of the side-branch states below.

### 0.2　The full state machine: states, guards, resource ownership, cleanup invariants

| State | Entry condition (guard) | Resources held | Main exit transitions |
| --- | --- | --- | --- |
| **REJECTED** (terminal) | ingress gate determines body/rate/quota limit exceeded; or tokenize/validate fails; or scheduler admission determines KV/token-budget cannot be satisfied even by queuing | None (never held KV or a scheduling slot) | Terminal, returns an error to the client |
| **QUEUED** | Passed ingress gate + tokenize/validate + prefix lookup + scheduler admission, KV **reserved** successfully but not yet picked by any tick | Request metadata + reserved KV quota (bookkeeping-level, not a physical block) + queuing deadline timer | → RUNNABLE (picked by a tick) / → TIMED_OUT (queuing deadline exceeded) / → CANCELLED (client cancels) / → DEGRADED |
| **DEGRADED** | Under capacity pressure, the scheduler chooses to lower this request's allowed best-of-n/max concurrency instead of rejecting outright | Same as QUEUED, but the reservation has been recomputed with the downgraded parameters | → RUNNABLE |
| **RUNNABLE** | Picked by some scheduler tick, with the tick having an available batch slot and physical KV blocks covering at least this workload | Reservation converted to allocated physical KV blocks + this iteration's batch slot + a token-budget share | → PREFILLING/DECODING (normal execution) / → PREEMPTED (scheduler reclaims resources for a higher-priority request) |
| **PREFILLING → DECODING** | Correspond respectively to: the miss-range prompt not yet fully computed / at least one token produced and not yet terminated | Same as RUNNABLE, with KV continuing to grow | → FINISHED / → BACKPRESSURED / → PREEMPTED / → CANCELLED / → TIMED_OUT / → FAILED |
| **BACKPRESSURED** | Tokens have been generated but the client can't consume them fast enough (slow connection/not reading), output buffer hits its cap | Same as DECODING + a bounded buffer of generated-but-unsent tokens | → DECODING (client consumption resumes) / → PREEMPTED (buffer hits the cap) / → CANCELLED (eventual disconnect) |
| **PREEMPTED** | Scheduler determines resources need to be freed for other requests; the concrete action is one of **discard-and-recompute** / **migrate to host (SWAPPED)** / **pause but retain KV (descheduled)** (§4.4, a non-exhaustive taxonomy) | Depends on the action: the recompute variant frees all physical KV; SWAPPED frees GPU memory but holds a host-memory copy; the descheduled variant continues to hold all physical KV blocks, releasing only the batch slot/compute share | → RESUMED (rescheduled) / → CANCELLED/TIMED_OUT (cancelled or timed out while waiting) |
| **SWAPPED** | A sub-case of PREEMPTED's migration path, where KV has finished being moved to host memory | Host-memory footprint (no GPU memory) + request metadata | → RESUMED (swap-in complete, GPU memory reoccupied) |
| **RESUMED** | Scheduler decides to fold a PREEMPTED/SWAPPED request back into some tick | Physical KV blocks re-allocated/moved back + batch slot, depending on origin | → RUNNABLE (folded normally into a subsequent tick) |
| **TIMED_OUT** (terminal) | Deadline exceeded while waiting in QUEUED/PREEMPTED | None (proceeds to cleanup) | Terminal |
| **CANCELLED** (terminal) | Client explicitly cancels or disconnects (heartbeat/half-close detection) | None | Terminal, must be marked **immediately** as the highest-priority KV reclaim target — cannot wait for the next tick to be processed |
| **FAILED** (terminal) | Worker crash/uncaught exception | None | Terminal, usually needs to trigger a retry (if idempotent) |
| **FINISHED** (terminal) | One of EOS/stop-string/stop-token sequence/`max_new_tokens`/context limit/constrained-decoding terminal state fires first | None | Terminal |

**Cleanup invariant (the four terminal states TIMED_OUT/CANCELLED/FAILED/FINISHED share the same reclaim routine, detailed in §1.7)**: (1) decrement the reference count of each physical KV block — only when the last holder of a shared prefix/beam branch releases it is the block truly returned to the allocator; (2) if the request terminates while still in QUEUED/DEGRADED, only the **reservation** bookkeeping needs to be rolled back — there are no physical blocks to release; (3) release the batch slot and the scheduler-side active-sequence quota; (4) produce exactly one terminal billing/log record — repeated invocations of this reclaim routine (e.g., cancellation and timeout firing almost simultaneously) must be **idempotent**: no double billing, no double-freeing the same block, and no leaks.

Two-phase admission further breaks down the claim that "admission happens before tokenization": the ingress gate (coarse-grained, before tokenize) can only block requests that obviously exceed limits; the truly precise KV/token-budget judgment must happen after tokenize/validate, or even after prefix lookup (only then is the miss-length known) — which is also why the KV quota in the QUEUED state is a "reservation," not a blanket "approval."

> 💡 **The serving state machine in 13 sentences** — one page to nail the interview core points (expanded in §0.2 and §1–§7 below).

1. **State-machine skeleton**: ingress gate → tokenize/validate → prefix lookup → scheduler admission + KV **reservation** → queued/runnable → prefill → first token → per-step decode (sampling + streaming) → finished → KV reclaim; preempted/swapped/backpressured/timed-out/cancelled/failed are side branches reachable at any time during execution, each with its own resource-ownership and cleanup obligations — see the full state table in §0.2. The first token normally comes directly from the logits of the last prefill forward pass, not from "prefill finishes, then a separate decode iteration runs."
2. **Declare the system boundary before quoting a metric**: $TTFT=t_{\text{first emit}}-t_{\text{arrival}}$, $ITL_j=t_{\text{emit},j}-t_{\text{emit},j-1}$, $TPOT=\frac{t_{\text{last emit}}-t_{\text{first emit}}}{N_{\text{out}}-1}$ — whether TTFT includes queuing/tokenization/network transit, and not conflating TPOT with "total latency / token count," must be stated up front (§2).
3. **High throughput ≠ good service**: throughput (requests/s or tokens/s) is just a rate number; look first at **SLO attainment** — the **fraction** of requests meeting TTFT/TPOT/deadline — then at **goodput** — the **rate** of requests/tokens meeting those SLOs (not a fraction! still in units of requests/s or tokens/s); chasing raw throughput alone can blow up tail latency for a large fraction of requests (§2.2).
4. **"Prefill is compute-bound, decode is bandwidth-bound" is only an empirical rule of thumb for common operating regimes, not a theorem** — a larger decode batch, or a change in weight byte-width from quantization, can push arithmetic intensity back up toward compute-bound; at very long context lengths, prefill too can become IO/kernel-limited (§2).
5. **Sampling filters have exact support sets** — don't wing it from intuition: top-p is the **smallest prefix that first reaches or exceeds the threshold** after sorting (the boundary token must be kept; both the "$p_i\ge p_0$" and "cumulative strictly $<p_0$" intuitions are wrong); min-p's support set **changes with temperature** (equivalent logit threshold $z_i\ge z_{\max}+T\log\alpha$); multiple filters **do not always commute** (§3).
6. **What constrained decoding really is**: set the logits of illegal tokens to $-\infty$ and then run softmax + sampling as usual — no further black magic (§3).
7. **PagedAttention is not "automatically loading KV from disk on a page fault"** — its core is mapping logical KV blocks to non-contiguous physical blocks via a block table; host swap/offload is a separate, stackable mechanism (§4).
8. **Prefix caching reuses exact token prefixes**, not semantically similar prompts; in multi-tenant settings you also need to prevent others from probing your prefix content via cache-hit/latency side channels (§4).
9. **Static batching's real limitation is not "must wait for all requests to finish together"** — it's that **the slot left behind by a finished sequence usually cannot be replaced by a new request before the whole batch ends**; continuous batching reduces such empty slots via iteration-level refill, but **doesn't guarantee lower latency for every workload** (§5).
10. **Chunked prefill is not a free lunch**: it typically improves decode ITL/fairness, but can lengthen that long prompt's own TTFT (§5).
11. **Disaggregation is not an unconditional speedup** — the benefit must exceed the extra cost of cross-node KV transfer, queuing coordination, and failure recovery; it can be slower under short prompts or weak interconnects (§6).
12. **Optimization techniques don't stack linearly**: quantization, speculative decoding, prefix caching, and continuous batching jointly change batch size, memory headroom, and kernel shapes — their gains cannot simply be multiplied or added (§7).
13. **Reproducibility is limited**: the same random seed does not guarantee token-identical results across batching order/GPU kernel/parallel topology/floating-point precision — in most cases this is normal behavior for a "batching + parallelism + floating point" composite system under these conditions, but if results still don't match with identical batch composition, identical parallel topology, and identical kernel path, you should suspect a genuine implementation bug (a race condition, uninitialized memory) rather than reflexively attributing it to "that's just how the system is" (§1.6).

## §1 The Request State Machine: From HTTP to Token Stream

### 1.1　The full picture: what a request actually goes through

Expanding on the diagram in §0, here are a few details interviewees most easily leave out:

- **Admission has two stages**: the ingress gate (body size/rate/tenant quota — a coarse check before tokenize) can only block requests that are obviously over the limit; the truly precise KV/token-budget judgment can only be made once tokenize/validate finishes (exact prompt length is known), or even after prefix lookup finishes (miss length is known) (§1.2, §5.5). On passing, the KV quota is first **reserved** (bookkeeping-level); it only converts to a physical allocation once a scheduler tick actually picks it after it enters the queue.
- **Prefix lookup happens before prefill**: the scheduler matches an existing cache by exact token ID (typically hash(prefix token sequence)); the hit portion is skipped, and prefill is only done on the miss-range suffix (§4.3).
- **Prefill on the miss range is one or more forward passes**: without chunked prefill it's typically computed in one shot; under chunked prefill it gets split across multiple iterations (§5.4).
- **The first token is not necessarily a separate decode round**: the last prefill forward pass itself computes the logits for the "last prompt position," and sampling happens directly there — but only if that forward pass actually computed that position. **If the prefix cache is a full hit** (every prompt token skipped the forward pass), the system only has K/V on hand, not the final hidden state/logits, and must either cache those logits separately or at least recompute the last prompt token — otherwise there's nothing to sample the first token from. This is an easily overlooked step in the "a full hit looks fastest" scenario. Many people assume "prefill finishes, then a separate decode iteration is run to produce the first token," which produces a wrong intuition about what makes up TTFT.
- **Per-step decode**: each step does only 1 forward pass over the new token, but attention must read the entire historical KV; the token sampled this step immediately becomes next step's input (§3 expands on sampling details).
- **Detokenization/streaming**: each time a token is produced, an incremental detokenize is attempted and pushed to the client; but a stop string may span the boundary of multiple tokens, so the server must buffer a small span of "not yet confirmed safe" text and cannot emit it preemptively (§3.7, the [D03] demo in §8).
- **Termination conditions** are a whole set, not "EOS and it's done": EOS token, stop-token sequence, stop string, `max_new_tokens`, total context limit, a constrained-decoding terminal state (e.g., JSON grammar closure), deadline, and client-initiated cancellation — whichever fires first must end the request; see §1.7 for the full termination/failure/cleanup protocol.

### 1.2　Two-phase admission and overload handling

Whether a request can be accepted is not "accept it if a GPU is free" — it's jointly decided across two phases (corresponding to the guards for REJECTED/QUEUED/DEGRADED in the §0.2 state table):

**Phase 1: ingress gate (before tokenize)** — looks only at coarse-grained signals: request-body size limit, rate limiting, whether the tenant quota is already exhausted. This layer blocks obviously over-limit traffic at minimal cost, without waiting for the tokenizer to finish.

**Phase 2: scheduler admission + KV reservation (after tokenize/validate and prefix lookup)** — only now is the exact prompt length and miss length known; jointly checks:

- **Current KV capacity headroom** (allocated block count vs. total block count, §4)
- **max batched tokens / concurrent-sequence cap** (this iteration's token-workload budget, §5.1, §5.5)
- **Predicted output length** (unknown output length is the main source of admission risk — most systems can only estimate it heuristically/from historical statistics, and the prediction error itself is one root cause of overload)
- **Tenant quota** (prevents a single tenant from exhausting global resources in a multi-tenant environment)

Passing phase 2 only means the KV quota has been **reserved** (deducted from remaining capacity at the bookkeeping level), not that a physical block has been allocated — physical allocation happens only when a scheduler tick actually picks this request (RUNNABLE, §5.1).

Three responses to overload: **immediate rejection** (REJECTED, fail fast, client can retry another replica), **queuing** (QUEUED, occupies request metadata and the reservation quota but not physical GPU resources, requires a queuing deadline), and **degradation** (DEGRADED, e.g., lowering this request's allowed max concurrency/best-of-n so it can be fit in with a smaller reservation instead of being rejected outright).

### 1.3　Fairness and starvation

The choice of scheduling policy directly determines who gets starved:

| Policy | Average latency | Tail latency | Long requests | Short requests |
| --- | --- | --- | --- | --- |
| Pure FCFS | Moderate | Can be very poor (head-of-line long request blocks) | Normal | Can get stuck |
| Shortest-job-first (needs length prediction) | Better | Depends on prediction accuracy | Can starve long-term | Prioritized |
| decode-priority | Good on decode side | Stable decode ITL | Prefill deferred, TTFT worsens | Benefits |
| prefill-priority | Good on prefill side | Decode ITL jittery | Good TTFT | Decode gets interrupted |

**There is no free scheduling policy** — maximizing utilization and fairness/tail latency are usually two objectives that must be actively traded off; you cannot just report "how much throughput improved."

### 1.4　Cancellation, client backpressure, disconnect

If a server doesn't promptly detect a client disconnect or explicit cancellation, it will keep running decode for a request **nobody is listening to** — pure waste of GPU that also crowds out other requests' scheduling/KV resources. Engineering-wise this requires: connection-layer heartbeat/half-close detection, the scheduler marking "cancelled" as the highest-priority KV reclaim target, and backpressure on streamed responses (the server cannot buffer generated-but-unsent tokens without bound when the client can't keep up). See §1.7 for the full termination/failure/cleanup protocol (including deadlines, worker failure, retry/idempotency).

### 1.5　Model loading and warm-up: cold start is not steady-state TTFT

Right after a replica starts, weight loading (reading from disk/object storage into GPU memory), CUDA kernel autotuning, CUDA Graph capture, and memory-pool initialization all cause first-batch latency far above steady state. **This cold-start latency must not be mixed into a "steady-state TTFT" benchmark** — evaluation reports must report the cold-start tail and the steady-state distribution separately, otherwise scale-up jitter gets misdiagnosed as a steady-state problem. See §2.4 for the full evaluation protocol (load sweep, saturation point, p99 sample size, etc.).

### 1.6　The real limits of reproducibility

The intuition that "the same random seed + the same weights should reproduce token-for-token" does not hold in a serving context: under continuous batching, the same request's kernel batch shape can change depending on **whether other requests happen to be in the same batch**; different batch sizes trigger different GEMM/attention kernel implementation paths; floating-point addition is not associative, so a changed reduce order changes the numeric result; tensor-parallel all-reduce order and MoE routing outcomes can also shift with load; furthermore, **multiple requests sharing the same RNG rather than each request having its own independent RNG stream** also makes "fixed seed" meaningless — which request consumes which segment of the random-number stream first depends on scheduling order.

Most of the time this is normal behavior for this "batching + parallelism + floating point" system combination under these conditions, **but it cannot be treated as a universal excuse**: if results still don't match even with batch composition, parallel topology, and kernel path all completely fixed, you should suspect a genuine implementation bug (uninitialized memory, race conditions, incorrectly shared mutable state) rather than reflexively attributing it to "that's just how the system is."

If an evaluation/audit process requires bit-for-bit, token-for-token reproducibility, it must simultaneously fix: **batch composition and arrival order**, **parallel topology** (TP/PP/EP degree and rank mapping), **hardware model** (reduction order and kernel implementations can differ across GPU architectures), **software/kernel version** (a library upgrade on the same hardware can also swap kernel implementations), **numeric precision** (fp16/bf16/fp32 have different rounding behavior), **collective communication algorithm** (ring/tree all-reduce have different orders), **the framework's determinism switch** (many frameworks disable deterministic kernels by default in exchange for speed), and **a per-request independent RNG stream** — fixing only the random seed is just one necessary condition among many, and far from sufficient.

### 1.7　Termination, failure, and the resource-cleanup protocol

A request can reach a terminal state via four different paths — **normal completion** (FINISHED: EOS/stop-string/`max_new_tokens`/context limit/constrained-decoding terminal state), **client cancellation** (CANCELLED: explicit cancel or disconnect), **timeout** (TIMED_OUT: deadline exceeded while queuing or executing), **worker failure** (FAILED: process crash/uncaught exception) — but regardless of which path is taken, all must converge on **the same reclaim routine**, otherwise resources get leaked or double-freed.

**Slow clients and partial streams**: when the client can't keep up, the server cannot buffer generated-but-unsent tokens without bound (session memory would grow, in turn slowing down the streaming buffers for other requests) — this needs a bounded queue + a backpressure signal (the BACKPRESSURED state, §0.2); once it hits the cap, either wait for the client to catch up or proactively downgrade/preempt this request.

**Worker failure and retry**: when a worker process crashes, every request it was serving must be marked FAILED; whether upstream retries depends on whether the request is **idempotent** — if the client already received partial tokens (a partial stream), a naive retry produces a duplicated prefix, so either the client must dedupe, or the server must explicitly declare "on failure, retry from scratch; any already-sent partial output is void."

**The generated / emitted / committed counts are not always equal**: a request may internally generate N tokens, but only M ≤ N of them are actually sent to the client (e.g., a few extra tokens computed before a stop string was detected and then truncated/discarded, or candidate tokens rejected under speculative decoding); billing and monitoring must explicitly state which number they use — mixing the three up causes "amount generated" and "amount billed" to disagree.

**KV reference counting and idempotent cleanup**: the cleanup routine must accomplish — (1) decrement the reference count of each physical KV block, with only the last holder of a shared prefix/beam branch truly returning it to the allocator; (2) if the request terminates while still in QUEUED/DEGRADED, only roll back the reservation bookkeeping — there are no physical blocks to release; (3) release the batch slot and the active-sequence quota; (4) produce exactly one terminal billing/log record. **Cancellation and timeout may fire almost simultaneously under a race condition** — repeated invocations of this reclaim routine must be a no-op (idempotent): it must not free the same block twice just because both signals arrived, nor fail to free it at all because each side assumed "the other one will handle it."

This protocol doesn't belong to any single optimization technique — it's the underlying guarantee that keeps the §0.2 state machine self-consistent "outside the normal path." If an interviewer asks "what exactly does the system do after a request is cancelled," the answer should cover the cleanup invariants shared by all four paths, not just "it stops decoding."

## §2 Serving Metrics, SLOs, and the Prefill/Decode Roofline

### 2.1　Precise metric definitions (the system boundary must be stated first)

$$TTFT = t_{\text{first emit}} - t_{\text{arrival}}, \qquad ITL_j = t_{\text{emit},j} - t_{\text{emit},j-1}$$

$$TPOT = \frac{t_{\text{last emit}} - t_{\text{first emit}}}{N_{\text{out}} - 1}\quad (N_{\text{out}} \ge 2)$$

If the endpoint of E2E latency is defined as "the emit instant of the last token," and all timestamps share the same system boundary, then the following is an **identity**, not an approximation:

$$t_{\text{last emit}} - t_{\text{arrival}} = TTFT + \sum_{j=2}^{N_{\text{out}}} ITL_j$$

If the reported "E2E latency" also includes the last flush, connection close, or network tail-transit overhead — costs that occur after the emit — then the right-hand side of the equation misses that tail term, and only then does it need to be written as $T_{E2E}\approx TTFT+\sum ITL_j$, with an explicit statement of what that extra term is.

Every quantity must first state its **system boundary**: does TTFT include queuing time, tokenization time, gateway/network transit? Is it "mean" or "p95/p99"?

> ⚠️ **Do not conflate TPOT with "total latency divided by token count"** — TPOT is the steady-state per-token cost during the decode phase (the denominator excludes the first token); if queuing time and prefill time are also folded into "total latency / token count," a request that was actually delayed by queuing gets incorrectly reported as "this model's decode is slow."

### 2.2　Throughput, SLO attainment, and goodput

High system throughput (tokens/s or requests/s) does not mean good user experience — if batches are packed to chase throughput and TTFT/TPOT worsen across the board, users perceive that as slower. Three concepts that are often conflated need to be separated here:

- **Throughput**: the amount of work processed per unit time, a **rate** — but the dimension must be stated first: request throughput (requests/s), output-token throughput (output tokens/s), or total-token throughput (total tokens/s including prompt tokens, often used to estimate GPU-time occupancy/cost). The three throughputs can produce completely different rankings under long-prompt/short-output vs. short-prompt/long-output workloads.
- **SLO attainment**: the **fraction of the total** of requests (or tokens) meeting the TTFT/TPOT/deadline SLO — a dimensionless **fraction**: $A = N_{\text{SLO}} / N_{\text{total}}$.
- **Goodput**: the **rate** at which requests (or tokens) meeting the same SLO constraints are produced per unit time — $G = N_{\text{SLO}} / \Delta t$, still in units of requests/s or tokens/s, **not a fraction**. Describing goodput as "the fraction of requests meeting the SLO" is a common but inaccurate statement: attainment and goodput are quantities along different dimensions — attainment can be high while goodput is low because the system is overall too slow, or attainment can be middling while goodput is still substantial because overall system throughput is very high.

When reporting system capability, give a throughput-latency curve (swept across different concurrency levels/arrival rates, see §2.4) rather than a single-point throughput number; stating which of the three throughputs, or whether it's SLO attainment or goodput, is likewise part of the "system boundary" that must be declared up front.

### 2.3　The prefill/decode empirical regime — not a theorem

The FLOPs expansion for one standard Transformer layer (full derivation in [`kv_cache_speculative_decoding_tutorial.md`](kv_cache_speculative_decoding_tutorial.md) §3.1, not repeated here) is roughly: in prefill, $L$ tokens are computed in one shot, weights are reused $L$ times, arithmetic intensity is usually high, and it's easy to saturate GPU compute; in decode, each step computes only 1 new token, yet must re-read the entire layer's weights plus KV that grows with history — under small batches, arithmetic intensity is low and it tends to fall into the HBM-bandwidth-bound regime. This "prefill compute-bound, decode bandwidth-bound" statement is an **empirical conclusion for common operating regimes**, and needs three qualifications immediately:

- **Arithmetic intensity rises as decode batch grows**: the same weights are amortized over more tokens, and some linear layers may swing back to compute-bound; quantization (fewer weight bytes) shifts the batch size at which this turning point occurs;
- **At very long context lengths, prefill attention can also be limited by IO/HBM/workspace/kernel implementation** — using IO-efficient attention (e.g., FlashAttention) does not automatically mean the whole prefill is compute-bound;
- **Input length and output length play asymmetric roles**: input length directly determines the prefill workload and also determines how much KV each subsequent decode step must read; output length doesn't change the initial prefill's arithmetic intensity, but it increases the number of decode steps, KV residency time, and the later context length (which in turn affects the KV read volume of each later decode step).

> 🎯 **Roofline is an upper-bound thought experiment, not a latency predictor** — the ridge point $I^{*}=\text{peak FLOP/s}/\text{HBM byte/s}$ is only used to compare against a given kernel's own arithmetic intensity, to judge which side it's theoretically closer to; real latency is also shaped by kernel implementation, scheduling overhead, and memory-access patterns — see [D09] in §8.

FlashAttention-style work optimizes the **IO efficiency of the attention kernel** (saving memory reads/writes); PagedAttention (§4) manages the **address space and allocation policy of KV during serving** — the two solve problems at different layers, and should not be conflated as the same technique.

One level further down are purely kernel/assembly-level factors, which this tutorial only names without deriving: operator fusion reduces the number of kernel launches and the memory round-trips of intermediate results; CUDA Graph records a whole sequence of kernel launches as a single replay, amortizing launch overhead (but usually needs to be re-captured when input shapes change); prefill's core matrix multiply is typically a well-shaped wide GEMM, while decode's core matrix multiply degenerates into a narrow GEMM or even a GEMV ($L_q=1$) — this is one of the underlying reasons the two differ in their sensitivity to kernel implementation; paged KV attention's kernel must do indirect addressing via the block table, adding a layer of overhead beyond contiguous memory access (the magnitude was already mentioned in §4.2). These are all examples of "implementation details significantly affecting measured numbers," not something this tutorial sets out to derive.

### 2.4　Trustworthy benchmarking and capacity planning (overview)

Reporting throughput, goodput, and TTFT/TPOT as one trustworthy number is easier to get wrong than it looks; below is an incomplete checklist that nevertheless covers the most common pitfalls — the full design of a load-testing script is out of scope for this tutorial.

- **Open-loop vs. closed-loop**: open-loop keeps sending requests at a fixed arrival rate (e.g., Poisson arrivals); closed-loop fixes the concurrency level (sends the next request only after one finishes). **The two measure different things**: closed-loop naturally lowers the effective arrival rate as the system slows down, easily producing "falsely stable" latency numbers for an already-overloaded system; open-loop is closer to real production traffic, but requires that the load-testing client itself not become the bottleneck.
- **Coordinated omission**: under closed-loop or naive retry logic, if some request gets stuck, the load-testing tool often "forgets" to keep sending subsequent requests on the original schedule, causing the sample to systematically miss the slowest cases — the measured p99 will look much more optimistic than the true value. The fix is to compute latency against the **planned arrival time** rather than the **actual send time**.
- **Real input/output length distributions**: use real traces (or at least synthetic distributions matching real ones) rather than fixed lengths, because the prefill/decode ratio, KV occupancy, and batch composition all depend strongly on the length distribution; a fixed-length load test will systematically deviate from production behavior.
- **Load sweep and the saturation point**: a single concurrency/arrival-rate data point is meaningless — sweep a curve from no-load to saturation, find the knee where the throughput-latency curve starts rising steeply (the saturation point), and explicitly report which latency corresponds to that throughput number.
- **Warm-up / cold-start separate reporting** (already set up in §1.5): the cold-start tail must be reported separately from the steady-state distribution, otherwise scale-up jitter will be misdiagnosed as a steady-state problem.
- **p99's sample size**: quantiles are very noisy at small sample sizes — reporting a tail quantile like p99 requires a sufficiently large sample, with a confidence interval stated, or at least the sample size.
- **Inputs to replica sizing**: target arrival rate, the goodput a single replica can reach under a given SLO, peak headroom (not sizing to the mean alone), N+1 redundancy (tolerating a single-replica failure), and inter-replica topology (whether replicas share the same switches/network failure domain) — §6.1 and Q19 expand on this in more concrete terms.

If this tutorial's positioning is "production-grade serving," it should also state whether autoscaling, readiness probes, rolling upgrades, and cross-availability-zone failure recovery are in scope — this tutorial covers only the request lifecycle within a single replica/single cluster; these operational topics are not expanded on.

## §3 From Logits to Token: Sampling, Constrained Decoding, and Streaming Output

### 3.1　The sampling pipeline: order matters, and there is no single order across implementations

Fix notation first: raw logits $z_i$, and $c_i$ the count of times a given token has already appeared. **The execution order of logits processors is not standardized across frameworks** — the phrase "turn on top-k, top-p, min-p, and penalties simultaneously" can itself give different results under different implementations. This tutorial adopts one pedagogical pipeline (purely for explanatory convenience — it does not represent the mandated order of any particular framework):

$$\text{hard constraints/logit bias} \to \text{repetition/frequency/presence penalties} \to \text{temperature} \to \text{truncation filters} \to \text{renormalize} \to \text{sample}$$

### 3.2　Greedy and temperature: T=0 must be a separate branch

$$\text{Greedy}:\quad y_t = \arg\max_i z_i$$

No need to compute softmax first; **ties for the maximum must have a deterministic tie-break rule declared** (this tutorial adopts taking the lowest index).

$$p_i(T) = \frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}\qquad\text{defined only for } T>0$$

`temperature=0` **must be implemented as a separate greedy branch**, and must never actually compute $z_i/0$. Let $A=\{i : z_i = z_{\max}\}$ be the set of indices tied for the maximum logit: when $|A|=1$ (the maximum is unique), the mathematical limit as $T\to0^+$ concentrates on this unique argmax; when $|A|>1$ (there are tied maxima), the limit is **uniform over $A$** — $p_i\to 1/|A|\ (i\in A)$, with the probability of every other token going to zero. This is **not "proportional allocation"** (under standard softmax, all logits tied for the max already have equal weight, so there's no "proportion" to speak of), nor does it automatically converge to any one specific token — a real greedy implementation still typically needs to explicitly pick one (a tie-break rule), which is a subtle mismatch between the mathematical limit and the engineering implementation ([D01] in §8 uses tied logits at a very small $T$ to verify that mass splits $1/|A|$ ways rather than collapsing onto one).

### 3.3　Truncation-family filters: exact support sets

**Top-k**: support set $S_k$ = the $k$ tokens with the largest logits:

$$q_i = \frac{p_i \cdot \mathbf{1}[i\in S_k]}{\sum_{j\in S_k} p_j}$$

Positive-temperature scaling doesn't change the ordering, so the top-k support set alone doesn't change with $T>0$ (what changes is the relative mass distribution within the set). If there's a tie in logits at position $k$, the support set is not unique, and a deterministic tie-break rule must be declared (this tutorial adopts sorting by $(-z_i, \text{token\_id})$ and taking the top $k$ — i.e., when logits tie, the smaller token id wins); the legal range for $k$ is $1\le k\le|E|$, where $E$ is the set of legal tokens with a finite logit (not $-\infty$) at this step.

**Top-p (nucleus sampling)**: sort probabilities in descending order $p_{i_1}\ge p_{i_2}\ge\cdots$, and take

$$m=\min\Big\{r:\ \sum_{j=1}^r p_{i_j}\ge p_0\Big\},\qquad S=\{i_1,\dots,i_m\}$$

i.e., **the smallest prefix, after sorting, whose cumulative probability first reaches or exceeds the threshold** — this boundary token must be kept.

> ⚠️ **Two common incorrect definitions both mis-drop the boundary token** — top-p is **not** "keep all tokens with $p_i\ge p_0$" (in practice, often no single token's probability reaches $p_0$, and this definition would yield an empty or erroneously small set); nor is it **"tokens with cumulative probability strictly less than $p_0$"** (this stops before the boundary token appears, so the final cumulative mass never reaches $p_0$). [D01] in §8 nails both errors down with the example $p=[0.4,0.3,0.2,0.1]$, threshold $0.65$: the correct support set is the first two tokens (cumulative $0.7\ge0.65$); the "strictly less than" definition stops erroneously with only the first token (cumulative $0.4$).

**Min-p**:

$$S_{\text{minp}}=\{i:\ p_i \ge \alpha \max_j p_j\},\quad\text{renormalize within the set}$$

If $p_i=\text{softmax}(z_i/T)$, this condition is equivalent to $z_i \ge z_{\max} + T\log\alpha$.

> ⚠️ **Min-p's support set changes with temperature — it is not a fixed logit-gap threshold** — because the right-hand side of the equivalent condition, $z_{\max}+T\log\alpha$, itself contains $T$: with $\alpha<1$, $\log\alpha<0$, so the larger $T$ is, the lower the threshold and the larger the support set ([D01] in §8 sweeps multiple values of $T$ over the same logits and verifies the support-set size is monotonically non-decreasing).

**Typical sampling**:

$$H(p)=-\sum_i p_i\log p_i,\qquad d_i=|-\log p_i - H(p)|$$

Sort by $d_i$ ascending (tokens whose "information content is closest to the overall entropy" go first), take the smallest prefix whose cumulative probability first reaches `typical_p`, and renormalize. **This is not equivalent to top-p** — it selects tokens whose "surprisal is close to average," and **the support set need not be a contiguous prefix in probability order** — [D01] in §8 constructs the example $p=[0.02,0.35,0.33,0.30]$: with `typical_p=0.63`, the support set is the **2nd- and 3rd-highest-probability tokens**, which actually **excludes the highest-probability token (0.35)** — this set is not equal to any "top-$m$ sorted by probability" prefix.

> ⚠️ **Multiple truncation filters do not necessarily commute** — top-p depends on "the filtering from the previous step, and whether renormalization has already happened," so when `top-k + top-p + min-p` are stacked, an explicit execution order must be given. [D02] in §8 verifies with the same distribution: `top_k(3)` then `top_p(0.8)` yields a support set of size 2; `top_p(0.8)` then `top_k(3)` yields a support set of size 3 — **different order, different result**.

### 3.4　Penalty types: a penalty is not "subtract the same constant"

$$\text{Frequency/presence penalty}:\quad z_i' = z_i - \alpha_f c_i - \alpha_p\mathbf{1}[c_i>0]$$

The frequency term grows linearly with occurrence count; the presence term only cares "has it appeared at all." Whether `counts` includes prompt tokens (versus only the generated portion) is determined by the specific API's semantics, and the tutorial/framework should state this explicitly.

$$\text{Sign-aware repetition penalty}\ (r>1):\quad z_i' = \begin{cases} z_i/r & z_i>0 \\ r\,z_i & z_i<0\end{cases}\qquad\text{applies only to tokens already seen}$$

This is **not** subtracting the same constant from every already-seen token — the sign differs, so the scaling direction differs (positive logits get pushed down, negative logits get pushed down further).

### 3.5　Beam search: a sequence search, not a dedicated fix for lack of diversity

Beam search maintains a candidate set of width $B$, approximately maximizing the sequence-level cumulative log-probability

$$\log P(y_{1:T}) = \sum_t \log p(y_t\mid y_{<t}, x)$$

Because it prunes step by step, it **does not guarantee global optimality**. Length normalization is not unique — a common form is $S(y)=\log P(y)/|y|^{\alpha}$ — different implementations use different penalty functions, so don't treat it as a universal definition.

> ⚠️ **Beam search primarily solves a sequence-level search problem, not a dedicated method for "lack of diversity"** — in dialogue generation, it can actually bias toward generic, repetitive, or prematurely terminated text; its footprint in this tutorial is smaller than top-p/temperature/penalties/constrained decoding, consistent with how frequently it's actually used in real conversational LLM serving.

### 3.6　Constrained decoding: at its core, it's just masking

JSON-schema-, regex-, and FSM-driven constrained decoding is at its core: **at each step, set the logits of tokens that are illegal in the current state to $-\infty$, then run softmax + sampling as usual** (the `apply_hard_mask` in [D01] of §8 demonstrates this primitive). The state-transition logic of the constraining automaton can itself be arbitrarily complex (FSM/PDA, incremental grammar parsing), but as far as the sampling pipeline is concerned it's just "one more masking step" — it needs no additional sampling theory. **Bad-word/banned-sequence constraints** (forbidding certain token **sequences** from recurring) and **forced tokens** (forcing the next position to be a specific token, equivalent to leaving only that token a finite logit and setting everything else to $-\infty$) are both special cases of the same masking primitive: the actual masking timing depends on **the state of the already-generated suffix** — a token is only masked when completing it with this next token would make the generated suffix fully match a banned sequence; permanently disabling a single token (context-independent, forbidden everywhere) is merely the special case of a banned sequence of length 1 (a unigram) — "forbidding a single token" should not be treated as the default or only form of a bad-word constraint.

**best-of-n / parallel sampling** cannot be treated as an ordinary sampling parameter: it generates $n$ candidates in parallel for the same prompt. Since these $n$ candidates share the same prompt, **the prompt's KV can be shared across candidates via the copy-on-write mechanism in §4.2**, without needing to be scaled up by $n$ overall — a more accurate approximation of KV usage is

$$M_{KV} \approx M_{\text{prompt,shared}} + n \times M_{\text{generated}}$$

That is, only the KV of the **generated suffix** (and the corresponding decode compute workload) scales approximately by $n$; the shared prompt prefix is only counted once. Only when the specific implementation **does not do CoW sharing** (each candidate independently copies the entire prompt KV) does it degrade to something close to the more pessimistic upper bound $n(M_{\text{prompt}}+M_{\text{generated}})$. Wall time also isn't necessarily $n$ times that of a single candidate — it depends on whether these $n$ candidates can be packed into the same decode iteration in parallel, or must run queued in batches. Schedulers and capacity accounting must book this using the more precise formula rather than crudely "multiplying by n." **No-repeat n-gram** is another kind of hard constraint: once an n-gram has appeared in the already-generated text, its completion token is forbidden from recurring (this too falls under the "mask illegal tokens" framework).

### 3.7　Stop conditions and streaming-output boundaries

Stop conditions must, at minimum, distinguish between: **EOS token**, **stop-token sequence**, **stop string** (string-level, possibly spanning multiple token boundaries), `max_new_tokens`, **total context limit**, **a constrained-decoding terminal state** (e.g., JSON grammar closure), **deadline**, and **client cancellation**.

A stop string spanning token boundaries is a common trap for streaming systems: before pushing to the client, the server must **buffer a span of uncommitted text that "might still be a stop-string prefix,"** and can only safely emit it once it's confirmed this span will no longer match a stop string (otherwise it leaks part of a stop-string prefix to the user). This is the same class of "boundary instability" problem, manifesting at a different layer, as the **prefix instability** covered in the tokenization tutorial's §5.4 (appending characters can trigger a merge across an old boundary, so `encode(s1)` is not a token-prefix of `encode(s1+s2)`), but the two are not the same thing — one occurs at the tokenizer's encode boundary, the other at the string-matching boundary of streaming output against stop strings; this tutorial only covers the latter, while the former and the full mechanism of token healing are covered in [`tokenization_tutorial.md`](tokenization_tutorial.md) §5.4. [D03] in §8 gives an executable minimal streaming-buffer implementation.

## §4 The KV Lifecycle: Allocation, Paging, Sharing, and Routing

### 4.1　Capacity accounting: connecting the sibling tutorial's known quantities to the serving side

Per-token KV byte count **is not re-derived in this tutorial** — it directly reuses the standard MHA/GQA/MQA **ideal payload** result already given in [`kv_cache_speculative_decoding_tutorial.md`](kv_cache_speculative_decoding_tutorial.md) §2.1:

$$m_{\text{token}} = 2\, N_{\text{layer}}\, N_{\text{kv\_head}}\, d_{\text{head}}\, b$$

(the factor of 2 corresponds to K and V; $b$ is bytes per element — this is the **ideal payload**, excluding allocator alignment, block-table metadata, quantization scale/zero-point, or the extra overhead of a special latent cache like MLA). This is the per-token slice of the same formula as the sibling tutorial's §2.1 capacity formula (multiplied by $L_{\text{ctx}}$); **exactly how much MQA/GQA/MLA save, and how the tensor shapes expand, is not re-derived here** — see that tutorial for it.

This tutorial treats $m_{\text{token}}$ as an externally known parameter, and only cares about three things that happen to it once it enters the serving lifecycle: block allocation and internal fragmentation (§4.2), cross-request sharing and copy-on-write (§4.2), and reclamation during preemption/eviction (§4.4) — [D07]–[D08] in §8(c) connect this known quantity to these three things, without re-verifying the formula itself.

### 4.2　PagedAttention: a block table, not automatic swap-out to disk

The core mechanism (full design in [`kv_cache_speculative_decoding_tutorial.md`](kv_cache_speculative_decoding_tutorial.md) §5; this section only covers the half needed for the serving perspective): each sequence's logical KV blocks are mapped through a **block table** to non-contiguous physical blocks in GPU memory, and the attention kernel reads KV via indirect addressing through this table. Fixed-size blocks eliminate the requirement that "every request must reserve a contiguous span of KV space for the full maximum length," reducing external fragmentation and over-reservation; but **the last, not-fully-filled block still has internal fragmentation** (demonstrated by the allocator in [D07] of §8).

> ⚠️ **PagedAttention does not mean "automatically loading KV from disk on a page fault"** — OS paging is only a design-inspiration analogy; PagedAttention's core is the **GPU attention kernel plus block-table indirect-addressing mechanism**, managing the mapping and sharing of non-contiguous physical blocks within GPU memory. This does not mean the whole serving system is thereby cut off from disk/host memory — what genuinely involves "moving out of GPU memory" is a separate, stackable mechanism: host swap / KV offload (moving KV to CPU memory or even slower storage), which is one of the cost options under preemption in §4.4, just **not part of PagedAttention's own default behavior**.

Physical blocks can be shared across a common prefix or beam branches via reference counting; when a shared, writable tail block needs to fork, this typically triggers **copy-on-write** (copying that block and rewriting the mapping, rather than modifying the shared content in place).

### 4.3　Prefix caching and multi-tenant correctness

Prefix caching reuses the KV of an **exact token prefix**; the matching condition must simultaneously satisfy: **identical token ID sequence**, **the same model version/weights**, **the same adapter (LoRA, etc.)**, **the same positional-encoding state**, **the same attention configuration**, **the same KV dtype/memory layout**, and **the same parallel-sharding scheme**.

> ⚠️ **"Semantically similar prompts" cannot share KV** — prefix caching only reduces the cost of **redundant prefill**; it doesn't reduce the prefill work on the miss-range suffix, nor any subsequent decode work. It is not a semantic cache, nor retrieval augmentation.

In multi-tenant settings, the prefix cache is a genuine security-boundary issue: the cache's **hash-key design, access permissions, and eviction policy** must all prevent different tenants from inferring each other's private prefix content via side channels like "whether the cache hit or not" and "differences in response latency" — or worse, from directly reusing cache content they should not have access to.

### 4.4　Eviction and preemption

For standard fully causal attention, **evicting any old KV forfeits the equivalence guarantee with "the exact computation path that retains the complete historical KV"** — the evicted position was originally supposed to participate in the attention computation of every subsequent query, and once it's dropped, the computation is no longer exact — it becomes an approximation that has not been validated by training (this is not "a minor optimization that usually has no material effect"). Safe sliding-window eviction requires either that the model itself was trained/inferred with windowed attention (as in some long-context architectures), or an explicit acceptance of the semantic change introduced by approximation methods like attention sinks or compression — treating it as a costed engineering choice, not a lossless optimization.

Preemption has several common approaches, differing in cost and in whether they actually relieve memory pressure — this is a common taxonomy, not an exhaustive classification:

- **Discard KV and recompute** (recompute): genuinely frees physical KV blocks, at the cost of extra recomputation compute;
- **Migrate to host memory** (swap): also genuinely frees GPU memory, at the cost of extra transfer overhead (growing with KV size);
- **Pause but retain KV (descheduling)**: simply removes this request from this round's scheduling, not participating in this iteration, but it **continues to hold all its physical KV blocks**. This option **should not be lumped together with the previous two as the same class of "memory reclamation method"**: it frees compute/scheduling capacity but does not relieve KV-memory pressure; if the goal is to free KV space for other requests, descheduling alone doesn't help — it must be paired with recompute or swap.

Which one to use (or which combination) depends on the specific tradeoff among recompute cost vs. transfer bandwidth vs. memory pressure — there is no universally optimal solution.

A long-generation request under continuous batching **usually does not monopolize the entire batch** (since the scheduler keeps admitting new requests to fill empty slots), but its interference is not just a capacity-level "occupying KV quota" issue: as its own context grows longer, **every one of its decode steps must read more historical KV and do more attention computation**, which directly lengthens this mixed batch's iteration duration and thus **directly worsens the ITL of other requests in the same batch** (not merely "crowding out quota" — it's that everyone in this round has to wait longer); the KV blocks it continues to occupy also raise memory pressure, increasing the probability of triggering preemption. Summarizing this as the three-fold effect of "lengthening the iteration + occupying capacity + raising preemption risk" is more accurate than "one long request blocks all requests," and more complete than a purely capacity-level framing of the interference.

### 4.5　Cache-aware routing

Under multi-replica deployment, which replica to route to cannot be decided just by "highest prefix-cache hit rate":

> 🎯 **Cache-aware routing's correct objective is to minimize expected completion time or SLO-violation probability, not to maximize hit rate** — the replica with the highest hit rate, if its queue is already deep, can still be slower than a replica with no hit but idle capacity. Routing decisions need to jointly consider: current queue depth, predicted KV usage, the prefill savings from a prefix-cache hit, and failure-domain isolation, rather than greedily optimizing a single metric.

## §5 Scheduling: Scheduler Tick, Batching, and Admission

### 5.1　The scheduler-iteration panorama: what one tick does

What the scheduler does on each tick (one iteration) is the hub connecting admission, KV, prefill, decode, sampling, and streaming, but the tutorial body often splits it across separate modules — here's the complete chain laid out together:

1. **Reclaim last round's terminal requests**: remove FINISHED/CANCELLED/TIMED_OUT/FAILED requests that concluded last round from the active set, releasing their batch slots and KV blocks per the §1.7 cleanup protocol (reference count decremented; a shared block is only truly returned when the last holder releases it).
2. **Decide who gets into the active set this round**: pick candidates from QUEUED (including DEGRADED) per the scheduling policy (FCFS/SJF/decode-priority/prefill-priority, §1.3), provided `max_num_seqs`, current physical KV capacity, and this request's already-reserved share all line up; the chosen request converts from "reserved" to "physically allocated," transitioning from QUEUED → RUNNABLE.
3. **Assign the token budget for this round's mixed batch**: `max_batched_tokens` is the token-workload budget for **this one iteration**, to be split between "prefill chunks (the quota each long prompt gets this round under chunked prefill, §5.4)" and "1 decode token per active request" — this is an iteration-level budget, not a reservation for the whole request lifecycle (the latter is already computed once at admission, see §1.2/§5.5).
4. **Assemble the mixed batch and run forward**: the same forward pass jointly contains several prefill chunks and several decode tokens; the kernel-level differences are covered at the end of §2.3.
5. **Sample, commit, or roll back**: apply the §3 sampling pipeline to the logits produced by forward, yielding the next token for each request; if speculative decoding (§7) or similar verification-requiring candidates are involved, the KV increment corresponding to rejected candidates needs to be rolled back (not counted into this request's committed KV, §1.7).
6. **Decode/emit, or backpressure**: hand the new token to streaming/stop-string detection (§3.7), sending whatever can be safely emitted to the client; if the client can't keep up, enter BACKPRESSURED (§0.2) instead of buffering without bound.
7. **Check termination conditions**: EOS/stop/length/deadline/cancellation — whichever fires, mark the request with the corresponding terminal state and hand it to the reclaim step at the start of the next tick.
8. **Perform preemption when needed**: if this round's KV pressure is too high or resources need to be freed for a higher-priority request, pick victims from the active set and handle them via one of the §4.4 approaches (recompute/swap/descheduling — note only the first two truly free memory).

This flow itself is precisely what the new **engine tick** demo in §8(d) is meant to string together — it maps these 8 steps onto the sampling/batch-scheduling/KV-computation functions already implemented in §3/§4/§8(a)(b)(c), rather than leaving each of them standing alone.

### 5.2　Static batching's real limitation

> ⚠️ **Static batching's key limitation is not "must wait for all requests to finish before outputting anything"** — a sequence that has already finished immediately produces its own token (no need to wait for the others); the real limitation is that **the batch slot it leaves behind usually cannot be replaced by a new request before the whole batch ends** — even empty, it must wait until the longest request in this batch finishes before the scheduler assembles the next batch.

### 5.3　Continuous (in-flight) batching

Continuous batching removes finished requests and admits new ones into the freed slots at **iteration boundaries**; its main benefit comes from **reducing decode padding, empty slots, and head-of-line blocking**.

> ⚠️ **Continuous batching does not guarantee lower per-request latency under every workload** — when all requests have similar lengths, continuous and static have almost no empty slots to save, and their makespans can be exactly the same (the equal-length counterexample in [D04], §8).

### 5.4　Chunked prefill

Chunked prefill splits a long prompt's prefill into multiple fixed-size chunks, letting the scheduler interleave decode work between chunks, thereby capping the per-iteration token workload and preventing one long prefill from monopolizing an entire iteration (head-of-line blocking).

> ⚠️ **Chunked prefill typically improves decode ITL and fairness under mixed workloads, but can increase this long prompt's own TTFT** — because its own prefill work is now spread across more iterations, and each iteration gets a smaller prefill budget. The exact outcome depends on the decode-priority policy, chunk size, and kernel-batching efficiency — [D05] in §8 computes both of these opposing effects using a simplified token-budget model. The "decode gap" here measures the **decode service gap** — how many iterations elapse between a decode request being served — a **proxy** for ITL, not ITL itself; only under the simplifying assumption that "each iteration takes roughly the same time" does a service gap dropping from 3 iterations to 1 correspond to lower ITL — in a real system, the actual duration of different iterations varies with batch composition and the prefill/decode mixing ratio, so the proxy and the real ITL are not a strict linear conversion.

### 5.5　Admission control and the joint KV/token-budget constraint

Bringing §1.2's two-phase admission "scheduler admission + KV reservation" step down to the scheduler-implementation level typically requires jointly maintaining: `max_num_seqs` (concurrent-sequence cap), `max_batched_tokens` (the per-iteration token budget, counting both prefill chunks and decode tokens — the iteration-level budget mentioned in §5.1), remaining KV block capacity, and per-tenant/per-request-type quotas. If any of these four constraints hits its cap, new admissions must be rejected or queued — not just checking one of them; the judgment here only affects whether the "reservation" succeeds — the request doesn't actually get physical KV blocks until it's picked by a tick in step 2 of §5.1.

## §6 From a Single Machine to a Cluster: Replicas, Parallelism, and Prefill/Decode Disaggregation

### 6.1　Replica routing

Under multi-replica deployment, routing decisions must trade off among **current queue depth, predicted KV usage, prefix-cache locality, and failure domains** (§4.5 already expanded on the cache-aware part) — it's not as simple as "always send to the replica with the highest cache-hit rate."

### 6.2　TP/PP/EP: an inference-only, forward-pass perspective

There's no backward pass and no optimizer state at inference time, but that doesn't mean parallel communication disappears — the full derivation of communication-volume analysis and backward/ZeRO/FSDP/training pipeline schedules is in [`distributed_training_tutorial.md`](distributed_training_tutorial.md); here we only cover three points specific to inference:

- **TP**: each layer's forward pass still needs collective communication (e.g., an all-reduce after row-parallel); **under a small decode batch, per-token communication latency can easily outweigh the compute savings** — but TP isn't only a tool for "the model doesn't fit" either: if a latency or throughput target requires spreading a single request's compute across multiple GPUs (even when the model fits on one GPU), TP can be a reasonable way to hit that target — the benefit needs to be measured against the actual latency target, batch size, and interconnect bandwidth, and cannot be generalized.
- **EP (MoE)**: expert parallelism at inference still needs token dispatch/combine communication, facing uneven expert load; **the absence of backward does not eliminate all-to-all latency, hot experts, or small-message communication overhead**; EP's use is also mainly driven by the MoE's expert layout, not purely the rule "the model doesn't fit."
- **PP**: under single-request, low-concurrency scenarios, PP **cannot automatically reduce latency**, since the end-to-end path still goes through every stage; enough concurrent sequences or micro-batches are needed to fill the pipeline, otherwise most of the time is spent waiting on bubbles.

> ✅ **When the model fits on a single GPU and the goal is aggregate throughput, data-parallel replicas are usually the preferred baseline** — avoiding cross-GPU communication for a single request, while still allowing free routing based on load and prefix locality (§4.5, §6.1). But this doesn't mean TP/PP/EP are only meaningful when "the model doesn't fit": whether they help must be evaluated empirically for the target latency, batch size, and interconnect bandwidth — **the model not fitting is the most common hard constraint, but not the only reason to shard with parallelism**.

### 6.3　Prefill/Decode Disaggregation

The motivation for splitting prefill and decode onto different nodes/resource pools: the two phases have very different resource characteristics (prefill is compute-intensive, decode is bandwidth-intensive), they can be scaled independently, and it reduces the interference of a long prefill on same-node decode ITL.

> ⚠️ **Disaggregation is not an unconditional speedup** — the benefit must exceed: the cost of cross-node KV transfer, extra queuing and metadata coordination, load imbalance, and the complexity of failure recovery. Under short prompts (where KV-transfer cost is relatively large compared to the total benefit) or weak interconnects, disaggregation can actually be slower — this is an engineering decision that requires measuring the break-even point, not an architecture that's default-superior. A prefix-cache hit shortens the effective prefill, changing the resource ratio between the prefill and decode pools, which further affects how this break-even point should be computed.

## §7 Combining Optimizations and Bottleneck Migration

Each optimization technique has a clean benefit story on its own, but **when combined, the gains cannot simply be multiplied or added** — together they change the batch size, memory headroom, kernel shapes, and the location of the bottleneck itself:

- **Quantization**: weight quantization mainly reduces bandwidth and capacity pressure from weights; **KV-cache quantization is an independent design point** — you cannot assume KV should use the same bit-width just because the model weights use low bits (numerical sensitivity and the choice of per-channel/per-token granularity are completely different). Numerical details (scale/zero-point, a concrete error analysis of GPTQ/AWQ/FP8) are in [`quantization_tutorial.md`](quantization_tutorial.md); this section only gives the decision dimensions:

  | Quantization target | Main effect | Coupling point with serving |
  | --- | --- | --- |
  | Weights | Bandwidth, memory capacity | Shifts the turning point of decode arithmetic intensity (§2.3) |
  | KV cache | Memory capacity, allowed batch/context ceiling | Affects §4's capacity accounting and block allocation |
  | Activations | Kernel support, numerical stability | Affects which fused kernels can be used, prefill's actual throughput |

- **Speculative decoding**: the full derivation of acceptance rate and the expected speedup $E[\tau]$ is in [`kv_cache_speculative_decoding_tutorial.md`](kv_cache_speculative_decoding_tutorial.md) §7; this tutorial only emphasizes serving bookkeeping: each target-verification pass processes multiple candidate tokens, producing a **variable number** of accepted tokens; the scheduler must book "number of tokens verified," "number of tokens output," and "KV increment" separately — it cannot treat this as a fixed-step decode. Quantizing the target or draft model can shift the distributional gap between draft and target via numerical error, and thus the acceptance rate — **speculative decoding and quantization are not strictly orthogonal**.
- **Prefix caching × continuous batching × disaggregation**: a prefix hit shortens the effective prefill length, which changes the actual workload of the prefill pool in a disaggregated deployment, which in turn changes the ratio at which the two resource pools should be provisioned — the three are mutually coupled, and evaluating any one in isolation yields misleading conclusions.

> 🎯 **Production symptom → diagnostic path (troubleshooting table)**

| Symptom | Check first |
| --- | --- |
| High TTFT | Queuing duration, admission rejection rate, prefix-cache hit rate, whether prefill is being crowded out by chunking |
| TPOT/ITL jitter | Whether a long prefill cut in line, preemption frequency, whether batch composition changes frequently |
| KV OOM | Block-allocation fragmentation rate, occupancy from long-generation requests, admission's KV-reservation policy |
| Low cache hit rate | Hash-key design, whether tenant-isolation policy over-partitions, whether routing breaks locality |
| Low GPU utilization but latency also poor | Check first whether batches are too small/fragmented, rather than rushing to add GPUs — this is usually a scheduling or admission problem |
| p99 suddenly worsens | First split out queueing time vs. service time; then instrument stage by stage along queue → cache hit → prefill → decode → KV pressure → preemption → kernel/collective → network → streaming (cache hit determines whether/how much prefill is needed, so it comes before prefill) — don't just stare at GPU utilization |

## §8 From Scratch: Sampling / Batch Scheduling / KV Allocation, and a Strung-Together Engine Tick

The full runnable script is [`code/inference_serving.py`](code/inference_serving.py) (pure standard library, no third-party dependencies, runs [D01]–[D09] in seconds on CPU). The first three demo groups correspond to §3/§5/§4, each verifying one isolated mechanism; (d) strings them together into one conceptual engine tick, corresponding to the scheduler panorama in §5.1.

**(a) Sampling**: split into "computing filtered probabilities" (`top_k_filter`/`top_p_filter`/`min_p_filter`/`typical_filter`, pure functions that never touch randomness) and "drawing" (`sample`, the only function that calls the RNG). Key assertion snippet:

```python
# top-p: after sorting, the smallest prefix whose cumulative probability first reaches or exceeds the threshold; the boundary token must be kept
p_ex = [0.4, 0.3, 0.2, 0.1]
filt = top_p_filter(p_ex, 0.65)
support = {i for i, p in enumerate(filt) if p > 0}
assert support == {0, 1}                                   # cumulative 0.7 >= 0.65
assert sum([p_ex[0]]) < 0.65                                # "strictly less than p0" would drop the boundary token

# min-p's support set changes with temperature (not a fixed logit-gap threshold)
sizes = [len(min_p_support_from_logits(logits_t, T, alpha=0.3))
         for T in (0.2, 0.5, 1.0, 2.0, 5.0)]
assert sizes == sorted(sizes) and sizes[0] < sizes[-1]      # larger T -> larger support set

# filter order is not commutative
support_a = support_of(top_p_filter(top_k_filter(probs, 3), 0.8))   # top_k -> top_p
support_b = support_of(top_k_filter(top_p_filter(probs, 0.8), 3))   # top_p -> top_k
assert support_a != support_b                               # {0,1} vs {0,1,2}
```

**(b) Continuous-batching simulation**: 3 requests A/B/C arrive simultaneously, capacity=2, requiring 1/4/1 decode tokens respectively, with each active request producing 1 token per tick:

```python
reqs = [Request("A", 0, 1), Request("B", 0, 4), Request("C", 0, 1)]
comp_static, _  = simulate_batching(reqs, capacity=2, policy="static")
comp_cont,   _  = simulate_batching(reqs, capacity=2, policy="continuous")
assert comp_static == {"A": 1, "B": 4, "C": 5}    # C has to wait until t=4 to start
assert comp_cont   == {"A": 1, "C": 2, "B": 4}    # once A finishes at t=1, C is inserted immediately

# note: "utilization" here computes "scheduler slot utilization" (total serviced tokens /
# (capacity * makespan)), a capacity metric within this unit-cost toy model, not GPU
# SM/HBM utilization; in a real system the actual per-iteration duration would vary with
# batch composition and the prefill/decode mix ratio, not the "equal per-tick duration" assumed here.
u_static, _ = utilization(comp_static, reqs, 2)   # slot utilization: 6 / (2*5) = 0.60
u_cont,   _ = utilization(comp_cont,   reqs, 2)   # slot utilization: 6 / (2*4) = 0.75
```

The equal-length counterexample (two requests both needing 3 ticks) verifies `comp_static_eq == comp_cont_eq` — continuous batching **does not always** strictly dominate. The scheduling simulator applies the same checks to every policy: total generated-token count is conserved, nothing runs before its arrival, active count never exceeds capacity, and completion immediately releases (no longer occupies KV).

**(c) KV block allocation and sharing**: the per-token byte count is taken directly from the quantity established in the companion tutorial in §4.1 ($m_{\text{token}}=64$ bytes is simply that known quantity plugged into one example configuration, not a new result this tutorial sets out to verify); what this section actually cares about is the three things that happen to this number once it enters the serving side — the block allocator reports both logical and allocated numbers (internal fragmentation); the prefix-sharing scenario verifies the physical block count drops after sharing; and an explicit check that "KV heads not evenly divisible by the TP degree cannot simply be divided":

```python
m_token = kv_bytes_per_token(n_layer=2, n_kv_head=2, d_head=4, bytes_per_elem=2)
assert m_token == 64                                        # 2*2*2*4*2 bytes/token

logical_slots, allocated_slots = block_alloc_slots([1, 5], block_size=4)
assert (logical_slots, allocated_slots) == (6, 12)          # internal fragmentation: 12 > 6

no_share = physical_blocks_no_sharing([6, 7], block_size=4)  # 4
with_share = no_share - 1                                    # sharing one full block -> 3
```

**(d) Integrated engine-tick walkthrough (conceptual, stringing together the three groups above)**: (a)/(b)/(c) each verified one isolated mechanism, but the "one tick" described in §5.1 is really about assembling them in sequence. The driver code below directly calls the functions whose assertions already ran above, demonstrating one full chain in the order admission → assemble batch → forward (placeholder) → sample → emit/backpressure → finished/preempted → reclaim (to accurately reflect the script's current state: this driver is currently a documentation-level integration illustration and has not yet been folded into `inference_serving.py`'s `main()` automated-assertion list; turning it into an independently verifiable 10th sanity check would require adding a `run_d10`-style function in the .py file that calls it):

```python
# Conceptual driver: strings together the three already-verified subsystems D01 (sampling)/D04 (batch scheduling)/D07 (KV block)
# into one engine tick in state-machine order, introducing no new randomness or unverified logic.
def engine_tick(active_requests, waiting_queue, capacity, block_size, m_token):
    # 1) reclaim last round's terminal requests + 2) pull requests from waiting into active (reuses D04's batch-scheduling logic)
    admitted = list(active_requests)
    while len(admitted) < capacity and waiting_queue:
        admitted.append(waiting_queue.pop(0))

    # 3) assign token/KV budget: use D07's block allocator to compute how many physical blocks this batch needs
    lengths = [r.ticks_needed for r in admitted]          # approximate occupied length via remaining decode steps
    _, allocated_slots = block_alloc_slots(lengths, block_size)
    kv_bytes_needed = allocated_slots * m_token             # §4.1's known quantity lands here

    # 4) forward: this tutorial does not implement a real model forward pass, this is just a placeholder --
    #    engine_tick's focus is how scheduling/KV/sampling fit together, not the numerical forward pass itself
    logits_per_request = {r.name: [4.0, 3.0, 2.0, 1.0] for r in admitted}

    # 5) sample/commit: reuse D01's sampling primitives to turn logits into the next token
    rng = random.Random(0)
    next_tokens = {r.name: sample(top_k_filter(softmax(logits_per_request[r.name]), k=2), rng)
                   for r in admitted}

    # 6) emit or backpressure: a real implementation would also go through D03's stop-string buffering;
    #    here we only do the "admitted/next_tokens" bookkeeping, without reimplementing D03

    # 7) finished/preempted: reuse D04's termination check of ticks_needed decrementing to 0
    finished = [r.name for r in admitted if r.ticks_needed <= 1]

    # 8) reclaim: finished requests immediately release the blocks they held (reuses D07's allocator semantics)
    remaining = [r for r in admitted if r.name not in finished]
    remaining_lengths = [r.ticks_needed - 1 for r in remaining]
    if remaining_lengths:
        _, allocated_after = block_alloc_slots(remaining_lengths, block_size)
    else:
        allocated_after = 0
    kv_bytes_after_reclaim = allocated_after * m_token

    return {"admitted": [r.name for r in admitted], "next_tokens": next_tokens,
            "finished": finished, "kv_bytes_before_reclaim": kv_bytes_needed,
            "kv_bytes_after_reclaim": kv_bytes_after_reclaim}

# example: A/B arrive, capacity=2; A only needs 1 decode token, so it becomes FINISHED and gets reclaimed this round
waiting = [Request("A", 0, 1), Request("B", 0, 4), Request("C", 0, 1)]
result = engine_tick([], waiting, capacity=2, block_size=4, m_token=64)
assert result["admitted"] == ["A", "B"] and result["finished"] == ["A"]
assert result["kv_bytes_after_reclaim"] < result["kv_bytes_before_reclaim"]  # 512B -> 256B, reclaim genuinely takes effect
```

This is not a new automated assertion added to the script — it's just arranging the functions already verified individually in (a)/(b)/(c) in the 8-step order of §5.1, for the reader to walk through against the state machine — the three return values `admitted`/`next_tokens`/`finished` correspond respectively to "the admission result," "this step's sampling result," and "which requests reached a terminal state"; `kv_bytes_before/after_reclaim` dropping from 512 bytes to 256 bytes directly demonstrates the §0.2 cleanup invariant "completion means release."

## §9 25 High-Frequency Interview Questions

Organized into three difficulty tiers; answers should be structured as "framework → key formula/definition → common mistakes," without repeating the main-text derivations.

### L1 Must-know questions

<details>
<summary>Q1. What stages does a request go through from arrival to receiving its first token?</summary>

- ingress gate → tokenize/validate → prefix lookup → scheduler admission + KV reservation → queued/runnable → prefill → first-token sampling (directly from the logits of the last prefill forward pass) → **detokenize + the first actual emit to the client** (§1, §0.2)
- "Receiving the first token" refers to the first emit observed on the client side, not the moment the server internally samples the id — there's still detokenize and network transit in between, and this time should normally be counted into TTFT
- Any stage can be interrupted by client cancellation/disconnect, requiring immediate KV reclamation (§1.7)
- Cold start (model loading/kernel autotuning/graph capture) is not counted into steady-state TTFT; if this request happens to be among the first batch right after a replica starts, the observed TTFT will include the cold-start tail, and a benchmark should report this separately rather than mixing it into steady-state statistics (§1.5, §2.4)

Treating "prefill finishes, then a separate decode run produces the first token" as the default implementation; missing admission's two-phase structure, the detokenize/first-emit step, and the need to separately report cold-start requests.

</details>

<details>
<summary>Q2. How are TTFT/ITL/TPOT/throughput/goodput each defined?</summary>

- $TTFT=t_{\text{first emit}}-t_{\text{arrival}}$, $ITL_j=t_{\text{emit},j}-t_{\text{emit},j-1}$, $TPOT=\frac{t_{\text{last emit}}-t_{\text{first emit}}}{N_{\text{out}}-1}$ (§2.1)
- Throughput is a **rate** — state up front whether it's request throughput (requests/s), output-token throughput (output tokens/s), or total-token throughput (including prompt tokens)
- **SLO attainment** (a fraction, $A=N_{\text{SLO}}/N_{\text{total}}$) is a proportion; **goodput** ($G=N_{\text{SLO}}/\Delta t$) is a rate — the two are quantities on different dimensions and must not be conflated (§2.2)
- The system boundary must be stated: whether it includes queuing/tokenization/network transit; mean or p95/p99

Only reciting formulas without stating the system boundary; conflating TPOT with "total latency / token count"; describing goodput as "the fraction of requests meeting the SLO" (that's SLO attainment — goodput is a rate).

</details>

<details>
<summary>Q3. Why are prefill's and decode's bottlenecks usually different? Is this always the case?</summary>

- Prefill: $L$ tokens computed in one shot, high weight-reuse rate, usually compute-bound; decode: repeatedly reading weights + KV under small batches, usually bandwidth-bound (§2.3)
- **Not a theorem**: a larger decode batch raising arithmetic intensity can swing it back to compute-bound; quantization shifts the turning point; at very long context, prefill too can become IO/kernel-limited

Treating this empirical conclusion as a universal theorem, unable to give any counterexample.

</details>

<details>
<summary>Q4. What are the exact support sets of top-k/top-p/min-p/temperature?</summary>

- top-k: the $k$ tokens with the largest logits ($1\le k\le|E|$, $E$ being the legal finite-logit set at this step; tie-break by $(-z_i,\text{token\_id})$ if there's a tie at position $k$); top-p: after sorting, the **smallest prefix whose cumulative probability first reaches or exceeds the threshold** (keeps the boundary token); min-p: $p_i\ge\alpha\max_j p_j$, equivalent logit threshold $z_i\ge z_{\max}+T\log\alpha$ (§3.3)
- Both common incorrect definitions of top-p ("$p_i\ge p_0$", "cumulative strictly $<p_0$") can mis-drop the boundary token
- min-p's support set changes with $T$ — it is not a fixed logit gap
- **Temperature's own support set**: as long as logits are finite and there's no extra mask, the softmax support set for any $T>0$ is the **entire vocabulary** (every token's probability is strictly positive, only their relative sizes differ); $T=0$ does not belong to this curve — it's a separate greedy branch (§3.2), whose "support set" is one specific member of the tied-max-logit set $A$ (uniquely determined after tie-break), while the $T\to0^+$ limit is uniform over $A$

Only reciting the formulas, unable to say specifically what's wrong with the two incorrect definitions; assuming the min-p threshold is independent of temperature; misremembering "first crosses" as "first strictly exceeds," dropping the boundary case where it equals the threshold; forgetting to answer what temperature's own support set is (default: whole vocabulary, with T=0 as a separate branch).

</details>

<details>
<summary>Q5. What problem does PagedAttention actually solve?</summary>

- Logical KV blocks are mapped to non-contiguous physical blocks via a block table, eliminating the requirement of "reserving a contiguous span for the full maximum length," reducing fragmentation (§4.2)
- Supports sharing (prefix/beam branches) + copy-on-write
- **Not** a reduction in attention's computational complexity, and **not** "automatic loading from disk on a page fault" — that's a separate host swap/offload mechanism

Answering "it reduces attention's computational complexity"; conflating it with disk paging.

</details>

<details>
<summary>Q6. What is static batching's key limitation?</summary>

- **Not** "must wait for all requests to finish before outputting anything" — completed requests can produce results immediately
- The real limitation: the slot left behind by a completed sequence **cannot be replaced by a new request before the whole batch ends** (§5.2)

Answering "must wait for everyone," unable to give the more accurate statement "the slot can't be replaced."

</details>

<details>
<summary>Q7. Why is continuous batching more efficient? Is it always faster?</summary>

- Removes finished requests and admits new ones at the iteration level; the main benefit is reducing empty slots/padding/head-of-line blocking (§5.3)
- **Doesn't guarantee lower latency under every workload**: when request lengths are similar, static and continuous can have exactly the same makespan (the [D04] counterexample in §8)

Only able to say "it's faster," unable to give the counterexample where the two are equivalent under equal-length requests.

</details>

<details>
<summary>Q8. What does prefix caching reuse? What are its limitations?</summary>

- Reuses the KV of an **exact token prefix**, requiring simultaneous matches on token ID, model version, adapter, positional-encoding state, attention configuration, KV dtype/layout, and parallel sharding scheme (§4.3)
- Only reduces redundant prefill, not the miss-range suffix or subsequent decode work
- In multi-tenant settings, hash-key design/access permissions/eviction policy need to guard against cross-tenant side channels

Assuming "semantically similar prompts" can also hit; not knowing that a hit only saves prefill while decode workload is unchanged.

</details>

### L2 Advanced questions

<details>
<summary>Q9. How is constrained decoding (JSON/schema/regex) implemented?</summary>

- At its core: at each step, set the logits of tokens illegal in the current state to $-\infty$, then run softmax + sampling as usual (§3.6)
- "Illegal in the current state" is determined by an automaton/parser that tracks the already-generated text: an FSM (regex/simple schema), a PDA (nested structures like JSON's bracket matching), or an incremental grammar parser — this state-transition logic can itself be quite complex
- The legal-token set is **tokenizer-aware**: candidate next tokens may span multiple characters, requiring character-by-character matching against the current automaton state to determine legality — it cannot be judged at the string level alone
- **Dead ends** need to be handled: some token may be legal right now, but choosing it would walk the automaton into a state from which an accepting state can never be reached again; a complete implementation must either do lookahead pruning in advance, or at least document that this edge case is unhandled
- As far as the sampling pipeline itself is concerned, no matter how complex the automaton, it always boils down to "one more masking step" at this point — no additional sampling theory is needed

Describing constrained decoding as requiring a brand-new sampling theory; stopping at the single line "mask illegal tokens," unable to identify automaton state, tokenizer-aware matching, and dead ends as the actually tricky parts.

</details>

<details>
<summary>Q10. How do you run a trustworthy serving benchmark?</summary>

- First distinguish **open-loop** (keeps sending at a fixed arrival rate) from **closed-loop** (fixed concurrency) — closed-loop automatically lowers the effective arrival rate as the system slows down, easily producing "falsely stable" numbers for an overloaded system
- Watch out for **coordinated omission**: compute latency against the planned arrival time rather than the actual send time, otherwise you'll systematically drop the slowest samples, making the measured p99 look more optimistic than reality
- Use real (or at least distribution-matched) input/output length distributions rather than fixed lengths
- Do a **load sweep**: sweep from no-load to saturation, reporting the throughput-latency curve and the saturation knee rather than a single-point throughput number
- The cold-start tail should be reported separately from the steady-state distribution; a tail quantile like p99 needs a sufficient sample size (§2.4)

Only able to say "run a load-testing script to measure throughput," unable to explain the open/closed-loop distinction, and unaware of how coordinated omission would distort measured latency.

</details>

<details>
<summary>Q11. Why does the execution order of sampling filters matter? Give a non-commutative example</summary>

- top-p depends on "whether the previous step has already renormalized" — the cumulative-probability base differs under different orders (§3.3); each filter passed through must renormalize within its support set, so the distribution seen by the next step is no longer the original one
- A concrete counterexample (§8 [D02], input distribution $p=[0.5,0.2,0.15,0.1,0.05]$): `top_k(3)` first truncates the support set to the 3 highest-probability tokens and renormalizes, then `top_p(0.8)` gives a support set of size 2; conversely, doing `top_p(0.8)` first (accumulating to 0.8 on the original distribution) then `top_k(3)` gives a support set of size 3 — same distribution, different order, different final support set

Only able to say "order might matter," unable to give a concrete input distribution and a verified counterexample; forgetting to mention that renormalization at each step is the root cause of the order-sensitivity.

</details>

<details>
<summary>Q12. What problem does chunked prefill solve? What is the cost?</summary>

- Splits a long prompt's prefill into chunks, avoiding monopolizing an iteration and blocking decode in the same batch (head-of-line blocking) (§5.4)
- Typically improves decode ITL and fairness under mixed workloads, but **may increase this long prompt's own TTFT** ([D05] in §8 quantifies this tradeoff with a token-budget toy model)

Only answering "it improves latency," unable to answer the negative impact on the long prompt's own TTFT.

</details>

<details>
<summary>Q13. What signals should admission control base its decisions on?</summary>

- Two phases: ingress gate (body/rate/quota, before tokenize) → scheduler admission + KV reservation (after tokenize, §1.2)
- **Prompt length is a determined value once tokenize completes**, so this portion of KV usage can be computed exactly; **output length can only be predicted before generation finishes** (heuristics/historical statistics), and the prediction error is an important source of overload — the two cannot be handled with the same deterministic treatment
- KV capacity headroom, `max_batched_tokens`, concurrent-sequence cap, tenant quota (§1.2, §5.5) — note that `max_batched_tokens` is the token-workload budget for **this one scheduling iteration** (§5.1), not the same as the capacity reserved for the whole request lifecycle — request-level capacity reservation is a separate accounting done at admission time
- Three responses to overload: reject (REJECTED), queue (QUEUED, with a deadline), degrade (DEGRADED)

Only saying "look at GPU utilization," missing the key but hard-to-estimate factor of predicted output length; conflating `max_batched_tokens` (an iteration-level budget) with the request-level KV reservation.

</details>

<details>
<summary>Q14. What are the different ways to implement preemption? What are their respective costs?</summary>

- Common approaches (not an exhaustive taxonomy, §4.4): **discard KV and recompute** (extra compute, genuinely frees memory), **migrate to host (swap)** (extra transfer, cost growing with KV size, genuinely frees memory), **pause but retain KV (descheduling)** (continues to occupy memory, only frees compute/scheduling share)
- **Descheduling should not be lumped together with the other two as a "memory-reclamation method"**: it doesn't relieve KV memory pressure — it just temporarily frees compute — if the goal is to free KV space, it must be paired with recompute or swap
- Basis for choosing: the specific tradeoff among recompute cost vs. transfer bandwidth vs. memory pressure — there's no universal optimum

Only knowing the word "preemption," unable to name the three common paths and their respective costs; treating "pause and retain KV" as also being able to save memory.

</details>

<details>
<summary>Q15. What is cache-aware routing's correct objective?</summary>

- **Not** solely maximizing prefix-cache hit rate, nor always solely minimizing expected completion time — the actual objective can be a combination of these: minimizing expected completion time/SLO-violation probability, maximizing goodput, controlling cost, or ensuring cross-tenant fairness (§4.5)
- A replica with the highest hit rate but a deep queue can be slower than an idle replica with no hit at all; there is no single universally correct objective — you first need to be clear which one (or which weighted combination) this routing pass is optimizing for

Treating "highest hit rate" or "shortest completion time" as the single, exclusive correct routing objective, ignoring that cost and fairness may also be dimensions routing needs to trade off.

</details>

<details>
<summary>Q16. How is speculative decoding accounted for within a serving iteration?</summary>

- The draft→verify→accept/reject step sits inside a decode iteration; the derivation of acceptance rate/$E[\tau]$ is in the sibling tutorial — this tutorial only covers the serving perspective (§7)
- Each verification produces a **variable number** of accepted tokens; the scheduler must book "tokens verified," "tokens output," and "KV increment" separately — it cannot treat this as a fixed step size
- **Accepted, emitted, and committed KV are not always equal**: the KV increment corresponding to rejected candidate tokens needs to be rolled back and must not be counted into this request's committed KV; the output side may also include a bonus/residual token (e.g., one extra free token the target model produces after a successful verification) — whether this token counts toward "number of output tokens" is determined by the specific implementation's billing convention and must be stated explicitly (§1.7)

Treating speculative decoding as a fixed-multiple throughput boost, ignoring the scheduling complexity introduced by the variable acceptance count; assuming generated tokens, truly committed KV, and the final billed output-token count are always the same thing.

</details>

<details>
<summary>Q17. Why doesn't the same random seed guarantee token-identical output across deployments?</summary>

- Under continuous batching, batch composition changes, so the kernel's batch shape changes with it; floating-point addition is not associative, so a changed reduce order changes the numeric result (§1.6)
- Parallel topology (all-reduce order) can also shift with load; **when multiple requests share the same RNG stream, which one consumes which segment of the stream first depends on scheduling order**, which also renders "fixed seed" meaningless
- Standard MoE **does not automatically change routing just because load changes** — it selects experts by the top-k of router logits; only under specific conditions involving **a capacity limit/drop, or numerical perturbation changing the top-k ordering** can load changes indirectly change which expert a token lands on — it's not accurate to broadly say "MoE routing changes with load"
- Most of the time this is normal behavior for a "batching + parallelism + floating point" composite system, but it's **not an excuse to be applied reflexively** — if results still don't match even with batch composition, parallel topology, and kernel path all fixed, you should suspect a genuine implementation bug

Assuming "fixed seed" guarantees token-identical reproduction across environments, unable to explain the root cause of floating point/batching; treating "that's just how the system is" as a blanket excuse for any reproducibility failure; unaware that MoE routing changes require specific conditions, and missing the shared-RNG-stream factor.

</details>

### L3 Top-lab questions

<details>
<summary>Q18. Under what circumstances is prefill/decode disaggregation worthwhile?</summary>

- Motivation: the two phases have very different resource characteristics, can be scaled independently, and it reduces the interference of a long prefill on same-node decode ITL (§6.3)
- **Not an unconditional speedup**: the benefit must exceed the extra cost of cross-node KV transfer, queuing coordination, and failure recovery; it can be slower under short prompts/weak interconnects
- A prefix-cache hit changes the effective prefill length, which in turn changes the two resource pools' optimal ratio

Treating disaggregation as a default-superior architecture, not discussing break-even conditions.

</details>

<details>
<summary>Q19. How do you choose TP/PP/EP and the number of replicas?</summary>

- **Parallelism choice**: when the model fits on a single GPU and the goal is aggregate throughput, data-parallel replicas are usually the preferred baseline, avoiding cross-GPU communication (§6.2); whether TP/PP/EP help must be measured against the latency target, batch size, and interconnect bandwidth — "the model doesn't fit" is the most common but not the only motivation — TP's per-token communication latency under a small decode batch can easily outweigh compute savings; PP cannot automatically reduce latency at low concurrency, needing enough micro-batches to fill the pipeline; EP's no-backward doesn't eliminate all-to-all/hot-expert latency
- **How to size the number of replicas** (the real focus of this question — answering only the parallelism choice is just half the answer):
  1. First determine the **goodput** a single replica can reach under the given SLO constraint (not raw throughput, §2.2) — this needs to be measured via the §2.4 load sweep under the target latency constraint, not estimated purely theoretically
  2. Divide the target arrival rate by the single-replica goodput to get the theoretical lower bound on the number of replicas needed
  3. Add **peak headroom** on top — don't size to the average arrival rate; leave headroom for expected peak/burst traffic
  4. Add **N+1 redundancy**, ensuring a single-replica failure/rolling upgrade doesn't cause overall overload
  5. Factor in **topology**: whether replicas are spread across different failure domains, whether they share the same switches — this affects routing (§4.5, §6.1) and the actually available aggregate goodput

Only able to say "if the model is big, use TP/PP," unable to answer that these parallelism methods can actually be slower under small-batch/low-concurrency scenarios; not discussing how to size the number of replicas at all, or just saying "add a few more replicas" without giving a concrete goodput-based sizing method.

</details>

<details>
<summary>Q20. Serving p99 suddenly worsens — how do you localize it?</summary>

- First split latency into **queueing time** (waiting for admission/scheduling) and **service time** (actually running prefill/decode) — quantile degradation is often a queuing-side problem, not the individual request itself getting slower
- Then instrument stage by stage in state-machine order: queue → cache hit → prefill → decode → KV pressure → preemption → kernel/collective → network → streaming (the §7 diagnostic table; cache hit comes before prefill because it determines whether/how much prefill this request needs to do)
- Check whether the **workload mix** has changed (a sudden shift in the input/output length distribution, some tenant suddenly sending a burst of long prompts); with a staged trace, directly compare which stage's time-share has risen
- Check for **cold-start/autoscaling** issues (a new replica just came up and is still cold-starting, or the replica count transiently dropped during scale-up)
- Check whether the load test/monitoring itself has **coordinated omission** (measuring against actual processing time rather than planned arrival time systematically drops the slowest requests)
- Don't just look at GPU utilization — low utilization can also be a scheduling/admission problem rather than needing more GPUs

Only able to say "look at GPU utilization," with no staged-localization methodology; placing cache hit after prefill, which doesn't match the real execution order; not distinguishing queuing time from service time.

</details>

<details>
<summary>Q21. How does a long-generation request affect other requests' SLOs?</summary>

- Under continuous batching it **usually does not monopolize the entire batch**, but the impact isn't just the capacity-level issue of "occupying KV quota":
  1. The longer its context grows, the more historical KV each decode step must read and the more attention computation it must do, **directly lengthening this mixed batch's iteration duration** — other requests' ITL in the same batch worsens along with it. This is a direct effect, not an indirect effect of "crowding out quota"
  2. It **occupies KV blocks and active-sequence quota over the long term**, raising memory pressure
  3. Rising memory pressure **increases the probability of triggering preemption**, in turn affecting more requests (§4.4)
- The more accurate statement is the superposition of these three effects — "directly lengthening the iteration + occupying capacity + raising preemption risk" — not a blanket "capacity-level interference," and certainly not "blocking all requests"

Using the overly strong statement "one long request will block all requests"; or conversely narrowing the impact down to just "occupying capacity," missing the more direct mechanism of it directly lengthening other requests' ITL.

</details>

<details>
<summary>Q22. What security risks does prefix caching pose in multi-tenant settings?</summary>

- Poor hash-key design can let different tenants' prefixes collide or have their existence inferred; the difference in response latency caused by hit/miss is itself a side channel (§4.3)
- Hit determination should do **tenant-namespace isolation** + **exact token verification** (a hash collision doesn't mean it's genuinely the same prefix — before a hit is honored, the full token sequence must be verified to actually be equal), otherwise different tenants might hit each other's cached content
- If the eviction policy lacks tenant isolation, tenant A's requests might evict tenant B's sensitive prefix, or vice versa be used to probe it
- "Encrypting the KV cache at rest" defends against the category of threat that is **static storage leakage** (e.g., disk/host memory being read directly), but doesn't solve the core problem here — **cross-tenant authorization isolation** and the **hit/latency side channel** are problems at the access-control and information-leakage layer, not a matter of whether the data itself is encrypted; you cannot say "it's encrypted so this topic is irrelevant"

Only answering "encrypt storage" and treating it as a solution to everything; unable to explain the concrete hit-rate/latency side-channel mechanism, and not knowing that hit determination also needs tenant-namespace isolation and exact token verification.

</details>

<details>
<summary>Q23. Why can't you casually do sliding-window eviction on standard fully causal attention?</summary>

- The more rigorous statement is: evicting any old KV **forfeits the equivalence guarantee with "the exact computation path that retains the complete historical KV"** — every query in standard attention could theoretically depend on information at any historical position, so once a position's KV is dropped, subsequent attention is no longer an exact computation — it introduces an approximation not validated by training (not the stronger, unnecessary claim that "every specific output value must change")
- Safe approach: either the model itself was trained/inferred with windowed attention, or you explicitly accept the approximate semantic change introduced by methods like attention sinks/compression, treating eviction as a costed engineering choice

Assuming eviction is a "free optimization" that can be losslessly stacked onto any model; or conversely asserting "it will definitely change every output value," over-committing an argument about an exactness guarantee into a claim about specific numerical values.

</details>

<details>
<summary>Q24. What extra cost does best-of-n/parallel sampling impose on KV and scheduling?</summary>

- $n$ candidates are generated in parallel, but they **share the same prompt** — the prompt's KV can be shared via copy-on-write without needing to be scaled up by $n$ overall; the more accurate approximation is $M_{KV}\approx M_{\text{prompt}}+\sum_j M_{\text{branch},j}$, where only the aggregate KV for the **generated suffixes** scales approximately with the number of branches (§3.6)
- Only when the implementation **does not do CoW sharing** does it degrade to something close to the more pessimistic upper bound $n(M_{\text{prompt}}+M_{\text{generated}})$; wall time also isn't necessarily $n$ times a single candidate's — it depends on whether these candidates can be packed into the same decode iteration
- Admission/KV capacity accounting must book this increment, otherwise the capacity-estimation error will not surface until runtime

Treating best-of-n as a lightweight parameter that "samples a few more times at almost no extra cost"; or conversely mistakenly booking it as "KV scaled by n overall," ignoring that the prompt prefix can be shared (a common mistake that self-contradicts the earlier CoW discussion).

</details>

<details>
<summary>Q25. When a long prompt arrives, how do you protect the SLOs of currently running online decode requests?</summary>

- Chunked prefill caps the prefill token budget per iteration, letting the scheduler interleave decode work (§5.4)
- decode-priority scheduling, an independent prefill resource pool (evolving further into disaggregation, §6.3), and diverting by predicted length at the admission stage
- Tradeoff: a technique like chunked prefill, which "squeezes in a time-slice within the same pool of resources," typically lengthens that long prompt's own TTFT — an active tradeoff for latency fairness; but with an **independent prefill resource pool/disaggregation**, as long as it's sized correctly, it can actually improve the long prompt's own TTFT at the same time (it no longer needs to compete with decode for the same resources) — "every technique that protects decode sacrifices prefill's TTFT" is not a universally true conclusion; it depends on which specific technique is used

Only answering with the single trick "use chunked prefill," unable to explain that this is a combination of multiple tradeoffs; or over-asserting that "these techniques all lengthen the long prompt's TTFT," ignoring that an independent resource pool/disaggregation can be a win-win.

</details>

## §A Appendix: Sanity Checks

This tutorial's from-scratch implementation should satisfy the following key invariants (pure Python standard library, no third-party dependencies, seconds on CPU; script at [`code/inference_serving.py`](code/inference_serving.py)):

1. **[D01] Sampling support sets**: greedy takes the lowest index on a tie for the maximum; `temperature_probs` raises for $T\le0$; with a unique maximum logit, $T\to0^+$ concentrates on the argmax, and with tied maximum logits the mass splits evenly; the top-k support set has size exactly $k$; top-p at $p=[0.4,0.3,0.2,0.1]$, threshold $0.65$, gives a support set of the first two tokens (cumulative $0.7$); min-p always keeps the highest-probability token, with the support set monotonically non-decreasing as $\alpha\downarrow$ and monotonically non-decreasing as $T\uparrow$ (for $\alpha<1$); typical sampling excludes the highest-probability token in the constructed example; adding a constant to all logits does not change the min-p/top-k/top-p support sets.
2. **[D02] Filter order is not commutative**: on the same distribution, `top_k->top_p` and `top_p->top_k` yield different support sets ($\{0,1\}$ vs $\{0,1,2\}$).
3. **[D03] Streaming stop detection**: a stop string spanning 3 token chunks is still detected, and no unsafe partial prefix is ever leaked early along the way; when there's no match, the full text can be completely recovered via flush.
4. **[D04] Batch scheduling**: static gives $A=1,B=4,C=5$ (utilization $0.60$), continuous gives $A=1,C=2,B=4$ (utilization $0.75$); under the equal-length-request counterexample, both policies have the same makespan; the scheduler satisfies token-count conservation, no running before arrival, active count never exceeding capacity, and immediate release on completion.
5. **[D05] Chunked prefill tradeoff**: max decode gap drops from 3 (unchunked) to 1 (chunked), but that long prompt's own TTFT rises from 3 to 4.
6. **[D07]–[D08] KV block allocation and sharing** (per-token byte count reuses the quantity established in the companion tutorial in §4.1, $m_{\text{token}}=64$ bytes, with $N_{\text{layer}}=2,N_{\text{kv\_head}}=2,d_{\text{head}}=4,b=2$, not separately verified in this tutorial): lengths $[1,5]$, block size $4$ give logical/allocated slots $=6/12$ (corresponding to $384/768$ bytes — internal fragmentation genuinely exists); lengths $[6,7]$ sharing 1 full block drops the physical block count from 4 to 3; rejects direct division when KV heads are not evenly divisible by the TP degree.
7. **[D09] Roofline ridge point** is used only as an upper-bound reference quantity, not as a latency predictor.
8. **(d) Engine-tick walkthrough** (a documentation-level integration illustration, not in the automated D-check list): strings [D01]'s `top_k_filter`/`sample`, [D04]'s batch-scheduling termination check, and [D07]'s block allocator into one tick in the 8-step order of §5.1; running it through shows KV usage dropping from 512 bytes to 256 bytes once a request is FINISHED and reclaimed — a direct demonstration of the "completion means release" invariant — but this code is currently not folded into the script's `main()`, and is not counted in the real output below.

The **actual output** of running `python3 code/inference_serving.py`:

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

(The above is the actual terminal output captured after re-running `code/inference_serving.py`, identical, verbatim, to the script's current version.)

## 📚 References

- **PagedAttention / vLLM** — Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention*, arXiv 2309.06180 (2023), SOSP 2023.
- **Orca (iteration-level / continuous batching)** — Yu et al., *Orca: A Distributed Serving System for Transformer-Based Generative Models*, arXiv 2204.05120 (2022), OSDI 2022.
- **Sarathi-Serve (chunked prefill)** — Agrawal et al., *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve*, arXiv 2403.02310 (2024), OSDI 2024.
- **DistServe (disaggregation, goodput)** — Zhong et al., *DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving*, arXiv 2401.09670 (2024), OSDI 2024.
- **Splitwise (phase splitting)** — Patel et al., *Splitwise: Efficient Generative LLM Inference Using Phase Splitting*, arXiv 2311.18677 (2023), ISCA 2024.
- **SGLang / RadixAttention (prefix-tree prefix caching)** — Zheng et al., *SGLang: Efficient Execution of Structured Language Model Programs*, arXiv 2312.07104 (2023).
- **Nucleus sampling (top-p)** — Holtzman et al., *The Curious Case of Neural Text Degeneration*, arXiv 1904.09751 (2019), ICLR 2020.
- **Top-k sampling** — Fan, Lewis & Dauphin, *Hierarchical Neural Story Generation*, arXiv 1805.04833 (2018), ACL 2018.
- **Min-p sampling** — Nguyen et al., *Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs*, arXiv 2407.01082 (2024).
- **Locally typical sampling** — Meister et al., *Locally Typical Sampling*, arXiv 2202.00666 (2022), TACL 2023.
- **Speculative decoding** — Leviathan, Kalman & Matias, *Fast Inference from Transformers via Speculative Decoding*, arXiv 2211.17192 (2022), ICML 2023.
- **Speculative sampling** — Chen et al. (DeepMind), *Accelerating Large Language Model Decoding with Speculative Sampling*, arXiv 2302.01318 (2023).
- **FlashAttention (attention-kernel IO optimization, not the same technique as PagedAttention)** — Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*, arXiv 2205.14135 (2022), NeurIPS 2022.
- **Megatron-LM (TP)** — Shoeybi et al., *Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism*, arXiv 1909.08053 (2019).
- **Roofline model** — Williams, Waterman & Patterson, *Roofline: An Insightful Visual Performance Model for Multicore Architectures*, Communications of the ACM (2009) (no arXiv id).
- **Constrained decoding / grammar-constrained generation** — Willard & Louf, *Efficient Guided Generation for Large Language Models*, arXiv 2307.09702 (2023). This paper supports a concrete implementation of guided/grammar-constrained generation, but should not be used to argue that "every constrained-decoding implementation is essentially just the single mask-and-renormalize mechanism" — this tutorial's §3.6 statement "it's just masking at its core" is an abstract summary at the sampling-primitive level, not a restatement of that paper's scope.
