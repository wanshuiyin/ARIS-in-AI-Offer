## §0 TL;DR Cheat Sheet

> 💡 **Distributed training in 7 dimensions, one page** — DP (data split) × TP (tensor split) × PP (layer split) × SP (sequence split) × CP (context split) × EP (expert split) × activation recompute. See §1–§11 for derivations.

1. **DDP** (PyTorch): every GPU holds a full model copy; gradients are synced via NCCL **all-reduce** during backward. Bucket fusion + computation/communication overlap is the key engineering optimization.

2. **ZeRO 1/2/3** (Rajbhandari et al., SC 2020): shard **optimizer state / gradient / parameter** respectively across $N$ GPUs, reducing per-GPU memory from $\Phi (2+2+12) = 16\Phi$ bytes to $16\Phi/N$ (fp16 training, Adam, parameter count $\Phi$).

3. **FSDP / FSDP2** (Zhao et al., VLDB 2023; PyTorch 2.4+, 2024): PyTorch-native ZeRO-3. FSDP2 replaces FSDP1's flat-parameter mode with **per-parameter DTensor sharding**, composing more naturally with TP / PP.

4. **TP** (Megatron-LM, Shoeybi 2019; Narayanan SC 2021): **column-parallel → row-parallel** pairing, with each layer requiring only 2× all-reduce per forward; attention is split by head, FFN uses col-parallel first layer + row-parallel second layer.

5. **PP**: GPipe (Huang NeurIPS 2019) → 1F1B / PipeDream (Narayanan SOSP 2019) → Megatron-LM **interleaved 1F1B** (Narayanan SC 2021). Bubble ratio $\approx (P-1)/M$ ($P$ stages, $M$ micro-batches), with interleaved compressing it by a further factor of $V$.

6. **SP / CP / EP**: SP (Korthikanti et al., MLSys 2023) shards LayerNorm/Dropout activations outside TP along the sequence; CP (Megatron-LM 2024 / Ring Attention Liu 2024) shards long context; EP partitions MoE experts across GPUs with **all-to-all** routing in forward.

7. **2024 frontier**: **DualPipe** (DeepSeek-V3, Dec 2024) bidirectional pipeline fully overlaps forward/backward computation with communication; **Llama 3 405B** (Meta 2024) uses 16K H100s, 3.8e25 FLOPs, 54 days with 466 interruptions (419 unexpected + 47 planned maintenance); **TorchTitan** (Liang et al., ICLR 2025) integrates FSDP2 + TP + PP + SP + Float8.

## §1 Intuition — why one GPU cannot hold a large model

**A softball question that few people fully answer**: when training a model, what exactly is stored in a single GPU's memory?

Consider fp16 mixed-precision training + Adam optimizer with parameter count $\Phi$:

| Item | Per-GPU usage | Notes |
|---|---|---|
| Parameter (fp16) | $2\Phi$ | used in forward / backward |
| Gradient (fp16) | $2\Phi$ | accumulated in backward |
| Optimizer state | $12\Phi$ | fp32 master copy ($4\Phi$) + Adam $m$, $v$ ($4\Phi + 4\Phi$) |
| **Subtotal (model states)** | **$16\Phi$** | independent of batch size |
| Activation | $O(B \cdot L \cdot D \cdot \text{depth})$ | linear in batch size, seq len, and depth |
| Workspace / temp buffer | varies | NCCL / cuDNN workspace |

A 7B model has $\approx 112$ GB of model states alone — too large even for an 80GB H100. So distributed training is **first and foremost about sharding model states + activations**.

> 💡 **Three orthogonal sharding dimensions** — training tasks can be sharded along these three axes, theoretically independently and in practice combined:
- **Data dimension** (DP / ZeRO / FSDP): shard the batch; replicas exchange gradients
- **Model dimension (depth)** (PP): shard layers; stages pass activations
- **Model dimension (width)** (TP / SP / CP / EP): shard tensors within a single layer; communicate intermediate results each step

3D / 4D / 5D parallelism is the Cartesian product of these axes. Llama 3 / DeepSeek-V3 both use 4D (DP × TP × PP × a subset of CP/SP/EP).

## §2 NCCL communication primitives and topology

99% of distributed-training communication goes through NCCL; you must be familiar with five primitives' semantics and traffic.

### 2.1 The five collective primitives

Suppose there are $N$ GPUs, each holding a buffer of size $S$.

| Primitive | Input → output | Equivalent semantics | Ring-algorithm traffic / GPU |
|---|---|---|---|
| **all-reduce** | $N$ × $S$ → $N$ identical $S$ | sum + broadcast | $2(N-1)/N \cdot S \approx 2S$ |
| **reduce-scatter** | $N$ × $S$ → $N$ different $S/N$ | sum then slice | $(N-1)/N \cdot S \approx S$ |
| **all-gather** | $N$ × $S/N$ → $N$ × $S$ | concatenate shards across ranks | $(N-1)/N \cdot S \approx S$ |
| **broadcast** | rank0's $S$ → $N$ × $S$ | single source | $S$ (tree) |
| **all-to-all** | $N \times N$ block transpose | shuffle | $(N-1)/N \cdot S \approx S$ |

**Key identity**:

$$\boxed{\;\text{all-reduce} = \text{reduce-scatter} + \text{all-gather}\;}$$

Each ring step transmits $S/N$, total $2(N-1)$ steps → per-GPU total $2(N-1)S/N \approx 2S$ bytes (almost independent of $N$ — this is the elegance of the ring algorithm).

> ⚠️ **Interview bonus: NCCL is not a single algorithm** — NCCL uses **tree all-reduce** for small messages (latency-bound, $O(\log N)$ hops); **ring all-reduce** for large messages (bandwidth-bound, throughput-optimal). On NVLink topologies it also offers **NVLS (NVLink SHARP)** — hardware-side reduction (H100/H200 NVSwitch supports this).

### 2.2 NVLink / IB / topology

| Link | Unidirectional bandwidth (H100-generation) | Use |
|---|---|---|
| NVLink 4.0 | 450 GB/s (18 lanes aggregated, per GPU; the official 900 GB/s figure is bidirectional aggregate, unidirectional is about half) | intra-node GPU↔GPU |
| PCIe 5.0 x16 | 64 GB/s | GPU↔CPU, slow path |
| InfiniBand NDR 400G | 50 GB/s (per port) | inter-node |

**Rule of thumb**: intra-node communication is **10-20×** faster than inter-node. So TP must fit inside a node; DP / PP can cross nodes. Llama 3 training topology: TP=8 (intra-node NVLink) × PP=16 (inter-node IB) × DP=128.

### 2.3 NCCL calls in PyTorch

```python
import torch
import torch.distributed as dist

dist.init_process_group(backend="nccl")  # backend fixed to nccl
rank = dist.get_rank()
world_size = dist.get_world_size()

# all-reduce (default SUM; can also be AVG / MIN / MAX / PRODUCT)
buf = torch.ones(1024, device=f"cuda:{rank % 8}") * rank
dist.all_reduce(buf, op=dist.ReduceOp.SUM)
# now buf == sum(0..world_size-1) * ones(1024)

device = torch.device(f"cuda:{rank % 8}")

# reduce-scatter
input_list = [torch.full((1024,), float(rank + i), device=device) for i in range(world_size)]
output = torch.empty(1024, device=device)
dist.reduce_scatter(output, input_list, op=dist.ReduceOp.SUM)

# all-gather
local = torch.full((1024,), float(rank), device=device)
gathered = [torch.empty(1024, device=device) for _ in range(world_size)]
dist.all_gather(gathered, local)

# all-to-all (essential for MoE)
in_split = list(torch.randn(world_size, 1024, device=device).unbind(0))
out_split = [torch.empty(1024, device=device) for _ in range(world_size)]
dist.all_to_all(out_split, in_split)
```

## §3 DDP — DistributedDataParallel

### 3.1 Algorithm skeleton

DDP is the simplest data-parallel implementation:

1. **Replicate**: every rank holds a complete model copy (parameters / gradients / optimizer states are all full)
2. **Shard batch**: the global batch $B$ is split into $N$ micro-batches; each rank computes its forward + backward on its share
3. **Sync gradients**: after backward, perform **all-reduce** on all gradients (SUM then divide by $N$ for averaging, equivalent to AVG)
4. **Local optimizer step**: each rank runs the same optimizer on the same gradients; parameters stay consistent

Mathematically (loss $\mathcal{L}$ as a mean over the mini-batch):

$$g_\text{global} = \frac{1}{N}\sum_{i=1}^N \nabla_\theta \mathcal{L}(\theta; \mathcal{B}_i) = \text{all-reduce-mean}(g_1, \dots, g_N)$$

Each rank gets the same $g_\text{global}$, so the $\theta$'s on $N$ ranks always stay synchronized (same init + same gradient + same optimizer).

### 3.2 Bucket fusion + overlap (DDP engineering essence)

Naive implementation: after backward completes → concatenate all gradient tensors → one all-reduce. This **idles the GPU waiting for communication**, an enormous waste.

What PyTorch DDP actually does:

- **Bucket**: pack multiple gradient tensors into a **bucket** (default 25 MB) in reverse-computation order
- **Hook**: trigger a hook when each parameter's gradient is computed
- **Overlap**: when a bucket fills (all its grads are computed), **immediately launch an all-reduce on a background stream**, while the main stream continues backward on earlier layers
- **Result**: communication is fully or partially hidden by the backward of earlier layers

```
Backward time axis →

Layer N:   [grad N]──┐
Layer N-1: [grad N-1]┼─bucket_N─[all-reduce N]
Layer N-2: [grad N-2]┘                       ↓ (background stream)
Layer N-3: [grad N-3]──┐
Layer N-4: [grad N-4]──┼─bucket_N-1─[all-reduce N-1]
...                                          ↓
Layer 1:   [grad 1]──────────────[all-reduce 1]

Main stream (compute):  ████████████████████████████████
Background (NCCL):           ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
                              ↑ overlap with compute
```

> ⚠️ **Bucket too small / too large are both bad** — too small: communication launch overhead dominates, bandwidth utilization is low; too large: filling takes too long, the first all-reduce starts late. PyTorch's default 25 MB is the common sweet spot for large models. Tune via `DDP(bucket_cap_mb=...)`.

### 3.3 PyTorch code (with overlap)

```python
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_ddp(rank, world_size, local_rank):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)   # multi-node: local_rank != global rank

def main(rank, world_size, local_rank):
    setup_ddp(rank, world_size, local_rank)
    device = torch.device(f"cuda:{local_rank}")

    model = MyModel().to(device)
    model = DDP(
        model,
        device_ids=[local_rank],
        bucket_cap_mb=25,                # bucket size
        gradient_as_bucket_view=True,     # memory opt: grad is a bucket view
        static_graph=False,               # if you're sure the compute graph is the same every iteration (no conditional branches / re-entrant checkpoint changes), set True; the main benefit is skipping the per-iteration unused-parameter traversal and enabling gradient accumulation across iterations / multiple backward passes, not "more kernel fusion"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loader = make_distributed_loader(rank, world_size)

    for batch in loader:
        x, y = batch[0].to(device), batch[1].to(device)
        loss = nn.functional.cross_entropy(model(x), y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()       # ← bucket hooks trigger all-reduce here
        optimizer.step()
```

### 3.4 Complexity

> 💡 **Notation convention (used throughout)** — $\Phi$ denotes **parameter count** (number). In fp16 / bf16 training, one parameter is 2 bytes, so the fp16 weight buffer is $2\Phi$ bytes. In the table below, "$2\Phi$ bytes" means byte count, "$\Phi$ params" means count.

| Item | Quantity | Note |
|---|---|---|
| Per-GPU memory | $16\Phi$ bytes + activation | model states not sharded (fp16 params/grads + fp32 master/Adam) |
| Per-step communication | $\approx 2 \cdot 2\Phi = 4\Phi$ bytes (ring all-reduce on fp16 gradient) | equivalent to reduce-scatter + all-gather |
| Scalability | compute scales linearly; communication bandwidth $O(1)$ (ring independent of $N$) | but latency-bound for small models, still affected by $N$ |

**DDP's fatal flaw**: model states are not sharded. With Adam training a 70B model, the per-GPU model states alone are 1.12 TB — DDP cannot fit. So 7B+ models must use ZeRO / FSDP.

## §4 ZeRO 1/2/3 — Zero Redundancy Optimizer

Rajbhandari et al. **"ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"** (SC 2020) is the second epochal work in distributed training (the first being Megatron-TP). Core idea: DDP wastes memory by replicating model states $N$ times — shard them into $N$ pieces so each GPU only holds $1/N$, with only slightly more communication.

### 4.1 Three-stage memory math

Let $\Phi$ be the parameter count and $N$ the DP world size. With fp16 mixed precision + Adam, per-GPU model states:

$$\text{DDP}: \quad 2\Phi + 2\Phi + 12\Phi = 16\Phi$$

The three ZeRO stages differ in **"what is sharded"**:

| Stage | Sharded content | Per-GPU model states | Communication (per step) |
|---|---|---|---|
| **ZeRO-1** | optimizer state | $2\Phi + 2\Phi + 12\Phi/N$ | $2\Phi$ (all-reduce = reduce-scatter + all-gather; ZeRO-1 only reduce-scatters grad) |
| **ZeRO-2** | optimizer state + gradient | $2\Phi + 2\Phi/N + 12\Phi/N$ | $2\Phi$ (same as ZeRO-1) |
| **ZeRO-3** | opt state + grad + **parameter** | $2\Phi/N + 2\Phi/N + 12\Phi/N = 16\Phi/N$ | $3\Phi$ (forward + backward each do one all-gather, backward does one reduce-scatter) |

> ✅ **In the limit** — when $N$ is large enough (e.g. 1024 H100s), ZeRO-3 per-GPU model states $16\Phi / 1024 = 0.0156\Phi$. For a 65B model that's $\approx 1$ GB / GPU; with activation checkpointing a single H100 can fully fit it.

### 4.2 ZeRO-3 workflow (most commonly used, forward / backward / optimize)

ZeRO-3 shards parameters themselves, so parameters must be **all-gathered** before use.

**Forward** (per layer / module):

```
1. all-gather: collect full W^(ℓ) from N GPUs  ──[comm: φ_ℓ bytes]
2. compute:    y = f(x; W^(ℓ))                   ──[compute]
3. release:    drop the shards that don't belong locally  ──[free memory]
```

**Backward**:

```
1. all-gather: re-fetch W^(ℓ) (forward released it)  ──[comm: φ_ℓ]
2. compute:    grad_W^(ℓ), grad_x
3. reduce-scatter: reduce grad_W^(ℓ) and slice back to per-shard  ──[comm: φ_ℓ]
4. release: drop non-local shards
```

Pseudo-code:

```python
# ZeRO-3 forward (single-layer abstraction, simplified)
def zero3_forward(layer_idx, x, sharded_W):
    # 1. all-gather full weight
    full_W = all_gather(sharded_W)    # [Φ_ℓ / N, ...] × N -> [Φ_ℓ, ...]
    # 2. compute
    y = layer_forward(x, full_W)
    # 3. release full_W (keep only sharded_W)
    del full_W
    return y

def zero3_backward(layer_idx, dy, sharded_W, cached_input):
    # 1. all-gather (already released after forward)
    full_W = all_gather(sharded_W)
    # 2. compute local gradients
    dW_local, dx = layer_backward(dy, full_W, cached_input)
    del full_W
    # 3. reduce-scatter gradient to the corresponding shard
    dW_sharded = reduce_scatter(dW_local)  # [Φ_ℓ, ...] / N -> [Φ_ℓ/N, ...]
    return dW_sharded, dx
```

### 4.3 ZeRO-1/2/3 vs DDP communication comparison (important interview question)

Total parameter count $\Phi$ (count). Below, the unit is **"fp16 weight buffer equivalents"** (i.e. $\Phi$ in the traffic column represents $2\Phi$ bytes of fp16-buffer traffic). Per-step **per-GPU** communication (ring assumption) for one forward+backward+update:

| Mode | Forward | Backward | Optim | Total (fp16 buffer equiv.) |
|---|---|---|---|---|
| DDP | 0 | $2\Phi$ (all-reduce grad) | 0 | $2\Phi$ (i.e. $4\Phi$ bytes) |
| ZeRO-1 | 0 | $\Phi$ (reduce-scatter grad) | $\Phi$ (all-gather updated params) | $2\Phi$ |
| ZeRO-2 | 0 | $\Phi$ (reduce-scatter grad) | $\Phi$ (all-gather) | $2\Phi$ |
| ZeRO-3 | $\Phi$ (all-gather params, on-the-fly) | $\Phi$ (all-gather) + $\Phi$ (reduce-scatter grad) | 0 (already in backward) | $3\Phi$ |

> 💡 **Key conclusion** — ZeRO-1/2 communication is the same as DDP ($2\Phi$ buffer) but memory drops sharply; ZeRO-3 has $1.5\times$ communication for $N\times$ memory reduction. In practice ZeRO-3's communication can also be partially hidden by **prefetch + overlap**, making it the mainstream choice for 70B+ models. In bytes: DDP $\approx 4\Phi$ bytes, ZeRO-3 $\approx 6\Phi$ bytes.

### 4.4 ZeRO-Offload / ZeRO-Infinity

**ZeRO-Offload** (Ren et al., USENIX ATC 2021): offload optimizer state + part of gradient to **CPU**; CPU runs the Adam update. Cost: CPU↔GPU PCIe traffic + slow CPU compute. Best for **small clusters + large models** (e.g. 13B on a single 8-GPU node).

**ZeRO-Infinity** (Rajbhandari et al., SC 2021): further offload parameters / optimizer to **NVMe**. Theoretically lets a single machine train 1T parameters (throughput is bounded by NVMe bandwidth and training is very slow in practice, but the design goal and the paper's whole pitch is "train larger models with a limited GPU count," not inference / fine-tuning).

> ⚠️ **Offload is a trade-off** — CPU offload typically increases per-step time 1.5-3×; NVMe offload is 5-10× slower. Use only when "doesn't fit + can't afford more GPUs". Production training prefers scaling out GPU count.

## §5 FSDP / FSDP2 — PyTorch native ZeRO-3

Zhao et al. **"PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel"** (VLDB 2023) brings the ZeRO-3 idea into the PyTorch mainline. FSDP2 (PyTorch 2.4+, 2024) is a major rewrite: **per-parameter DTensor sharding** replaces FSDP1's flat-parameter mode.

### 5.1 FSDP1 vs FSDP2 core differences

| Dimension | FSDP1 (2022-2023) | FSDP2 (2024+) |
|---|---|---|
| Data structure | **FlatParameter**: concatenate all params in the wrap unit into one 1D buffer, then chunk | **DTensor per-parameter**: each param sharded independently along dim-0 |
| State dict | needs all-gather to produce unflattened state dict | sharded state dict, communication-free |
| Frozen parameters | Default (`use_orig_params=False`): within a unit, all must be frozen or all trainable; with `use_orig_params=True`, the original `nn.Parameter` objects are preserved, and mixed frozen/trainable within a unit is possible (the standard setup for LoRA-style partial fine-tuning) | each param independent, mixed frozen/trainable is natural |
| TP composition | difficult (flat-buffer conflicts with TP's different shard axis) | **natively compatible**: DTensor describes multi-axis placement (`Shard(0)`, `Replicate`, `Shard(1)` combos) |
| API | `FullyShardedDataParallel` | `fully_shard()` functional wrap |

### 5.2 FSDP wrap policy (the most important design decision)

FSDP does not treat the whole model as one unit. It shards by **custom unit boundaries**; params in a unit all-gather / reduce-scatter together.

```python
import torch
import torch.nn as nn
from torch.distributed.fsdp import fully_shard           # FSDP2 API
from torch.distributed.fsdp import MixedPrecisionPolicy

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.mlp  = FeedForward(d_model)
        self.ln1  = nn.LayerNorm(d_model)
        self.ln2  = nn.LayerNorm(d_model)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

# Treat each TransformerBlock as one FSDP unit
model = MyLLM(n_layers=32, d_model=4096, n_heads=32).cuda()

mp_policy = MixedPrecisionPolicy(
    param_dtype=torch.bfloat16,        # forward uses bf16
    reduce_dtype=torch.float32,        # gradient reduce uses fp32 to avoid error accumulation
)

for block in model.blocks:
    fully_shard(block, mp_policy=mp_policy)
fully_shard(model, mp_policy=mp_policy)   # root also needs wrapping
```

> ⚠️ **Engineering trade-off of wrap granularity** — smaller unit (e.g. wrap each linear): each all-gather only fetches one layer, peak memory low, but many communication calls and prefetch is hard; larger unit (e.g. a whole block or several blocks): fewer comms, easy to overlap, but higher peak memory. **TransformerBlock granularity is the standard for LLaMA / GPT-class models**.

### 5.3 FSDP2 + mixed precision + activation checkpoint

```python
from torch.distributed.fsdp import CPUOffloadPolicy, OffloadPolicy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper, CheckpointImpl
)

# 1. Activation checkpoint (gradient checkpointing) — recompute trades memory
def apply_ac(model):
    for i, block in enumerate(model.blocks):
        model.blocks[i] = checkpoint_wrapper(
            block,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )
apply_ac(model)

# 2. FSDP2 wrap
mp_policy = MixedPrecisionPolicy(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
)
offload_policy = OffloadPolicy()                   # or CPUOffloadPolicy() to offload to CPU

for block in model.blocks:
    fully_shard(block, mp_policy=mp_policy, offload_policy=offload_policy)
fully_shard(model, mp_policy=mp_policy)

# 3. Training (note: optimizer must be torch._foreach_ friendly)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=True)

for batch in loader:
    loss = compute_loss(model, batch)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
```

### 5.4 FSDP vs ZeRO-3 communication difference (L3 question)

**Conclusion**: **completely the same** (at the wrap-unit = layer granularity). FSDP is the PyTorch-native implementation of ZeRO-3; the algorithms are equivalent. The differences are engineering:

| Dimension | DeepSpeed ZeRO-3 | FSDP2 |
|---|---|---|
| Integration | external library + monkey patch | PyTorch mainline |
| state_dict | needs a separate API to stitch | native `set_model_state_dict` / DCP |
| TP composition | DeepSpeed-Megatron bridge | DTensor native |
| Freezing / LoRA | complex (flat-param conflicts) | natural |
| Compilation | restricted | `torch.compile(fullgraph=False)` friendly |

Practical choice: **new projects use FSDP2; DeepSpeed is still used heavily in 70B+ MoE / multi-framework scenarios**.

### 5.5 HSDP — Hybrid Sharded DP

**Problem**: 1024 GPUs all on ZeRO-3 means a single all-gather across 1024 GPUs → per-GPU traffic $(N-1)/N \cdot \phi \to \phi$ stays the same, but **latency rises sharply** (IB cross-node is expensive).

**HSDP**: ZeRO-3 within a group (e.g. 8 GPUs in a node), DDP across groups. Equivalent to dividing 1024 GPUs into 128 ZeRO-3 groups (8 each).

| Mode | Shard scope | Per-GPU model states | Inter-node communication |
|---|---|---|---|
| Pure ZeRO-3 | full 1024 | $16\Phi / 1024$ | many |
| HSDP (8) | 8 GPUs per group | $16\Phi / 8 = 2\Phi$ | few (inter-group is grad all-reduce) |

Trade-off: **memory traded for communication efficiency**. Llama 3 uses HSDP at some stages of training.

## §6 ZeRO++ — communication optimizations (2023)

Wang et al. **"ZeRO++: Extremely Efficient Collective Communication for Giant Model Training"** (NeurIPS MLSys workshop 2023, arXiv 2306.10209). Three additions on top of ZeRO-3:

### 6.1 qwZ: Quantized Weight all-gather (quantize forward comms)

ZeRO-3 forward all-gathers fp16 weight per layer. `qwZ` converts fp16 → int8 before all-gather, **halving traffic**.

```
forward (vanilla): weight_shard (fp16) ──all-gather──> full_weight (fp16) ──compute
forward (qwZ):    weight_shard (fp16) ──quant──> shard (int8)
                                        ──all-gather──> full (int8)
                                        ──dequant──> full (fp16) ──compute
```

Cost: do block-wise quantization / dequantization before and after each all-gather (typical block size 2048-4096 elements with per-block scale). Block-quant is much more precise than per-tensor quant, with < 1% effect on training loss.

### 6.2 hpZ: Hierarchical Partition

Observation: **all-gather during backward is more expensive than during forward** (deeper layers backprop first). `hpZ` maintains a **secondary ZeRO-3-style partition** of the parameters within a node:

- Intra-node: H GPUs (e.g. H=8) collectively hold one full copy of the weight, but it is still sharded ZeRO-3-style — each GPU only holds an extra Φ/H (not the full Φ)
- Inter-node: weight is still sharded globally ZeRO-3-style (unaffected by hpZ)

During backward, an intra-node (NVLink) all-gather reassembles the H shards into the full weight, and the cross-node (IB) all-gather is no longer needed there — much cheaper. Cost: extra memory per GPU ≈ Φ/H (with H=8, roughly an extra Φ/8 on top of the original ZeRO-3 per-GPU cost), not a full copy of Φ per GPU as "intra-node DDP" would imply.

### 6.3 qgZ: Quantized Gradient reduce

The tail reduce-scatter on gradients during backward uses **INT4 (4-bit)** block quantization (not INT8), paired with a hierarchical **all-to-all + dequant + local sum** reduction path to control precision loss. The vanilla reduce-scatter SUM cannot be naïvely quantized (quant + sum accumulates error), which is why the direct quantized all-reduce path is avoided in the first place.

### 6.4 Combined effect

ZeRO++ paper reports **2.16× throughput improvement at 384 GPU scale**, with 4× lower traffic — this is the combined effect of qwZ (INT8 on fwd/bwd all-gather, 2×) + hpZ (the cross-node backward all-gather is replaced by an intra-node one, eliminating that cross-node traffic entirely) + qgZ (INT4 on reduce-scatter, 4×), not simply "2×(int8) times 2 collective primitives". Cost: complex implementation, precision requires careful ablation.

> ⚠️ **Quantized communication ≠ quantized training** — Here we only quantize **transient buffers in the collective communication path**; weight storage and compute remain fp16 / bf16. So loss impact is usually negligible, and this is different from fp8 training (e.g. fp8 GEMM on Hopper).

## §7 Tensor Parallel — Megatron-LM

Shoeybi et al. **"Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism"** (arXiv 2019) + Narayanan et al. **"Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM"** (SC 2021).

### 7.1 Core idea: column-parallel → row-parallel pairing

Consider an MLP layer: $Y = \text{GELU}(X W_1) W_2$, with $X \in \mathbb{R}^{B \times D}$, $W_1 \in \mathbb{R}^{D \times 4D}$, $W_2 \in \mathbb{R}^{4D \times D}$.

**Column Parallel** (shard $W_1$'s output dim $4D$):

$$W_1 = [W_1^{(1)} \mid W_1^{(2)} \mid \cdots \mid W_1^{(T)}], \quad W_1^{(i)} \in \mathbb{R}^{D \times 4D/T}$$

Each rank $i$ holds $W_1^{(i)}$ and independently computes $Y_1^{(i)} = X W_1^{(i)} \in \mathbb{R}^{B \times 4D/T}$. **Input $X$ is fully replicated; output is sharded along columns**.

**Row Parallel** (shard $W_2$'s input dim $4D$):

$$W_2 = \begin{bmatrix} W_2^{(1)} \\ W_2^{(2)} \\ \vdots \\ W_2^{(T)} \end{bmatrix}, \quad W_2^{(i)} \in \mathbb{R}^{4D/T \times D}$$

Input is sharded along rows: $Z^{(i)} = \text{GELU}(Y_1^{(i)}) W_2^{(i)} \in \mathbb{R}^{B \times D}$. Each rank computes its partial sum, then **all-reduce** to sum them:

$$Y = \sum_{i=1}^T Z^{(i)} = \text{all-reduce}(Z^{(1)}, \dots, Z^{(T)})$$

### 7.2 Communication analysis

Column parallel: input $X$ is fully replicated, output sharded along columns → **forward has no communication** (output is distributed across ranks).

Row parallel: input is row-sharded, output requires all-reduce → **forward: 1× all-reduce $(BD)$**.

**Col + row pairing**: the column-parallel tail has no communication (its result feeds directly into row-parallel), and the row-parallel tail does 1× all-reduce. The full MLP block forward involves **only 1× all-reduce**.

Backward mirrors: each MLP block backward also performs 1× all-reduce (gradient flow through col-row mirrors the communication).

**Total**: a single transformer block (MLP + Attention, where attention is also col-row) forward + backward = **4× all-reduce, each of size $BLD$** ($L$ = seq len).

### 7.3 TP sharding of attention

Multi-head attention has $W_Q, W_K, W_V \in \mathbb{R}^{D \times D}$. Shard **directly by head dim** (each head is independent, naturally column-parallel):

```
H heads, T-way TP:
  - Each rank holds W_Q^(i), W_K^(i), W_V^(i) for H/T heads
  - Each rank computes its head_h(Q W_Q^h, ...) independently
  - Output W_O uses row-parallel → tail all-reduce
```

> 💡 **Why attention must be sharded on head boundaries** — hidden_dim itself can be sharded, but the shard boundary must align with heads (each rank gets a whole number of complete heads), so the QK^T/softmax computation within each head still happens entirely on one GPU, requiring no cross-rank communication of intermediates. What's actually wrong is sharding **inside** a head (splitting one head's $d_{head}$ across ranks), which breaks the complete dot product that softmax normalization needs and forces extra communication. Code-wise it's also simpler — whole heads map directly to ranks.

### 7.4 TP code skeleton

Below is a pedagogical col-parallel / row-parallel implementation, with explicit communication pairs in forward and backward (key: col-parallel forward has no comm, backward all-reduces the input grad; row-parallel forward all-reduces the output, backward has no comm):

```python
import math
import torch
import torch.nn as nn
import torch.distributed as dist

class _CopyToTPRegion(torch.autograd.Function):
    """ Identity in forward, all-reduce in backward (col-parallel entry) """
    @staticmethod
    def forward(ctx, x, tp_group):
        ctx.tp_group = tp_group
        return x
    @staticmethod
    def backward(ctx, grad_out):
        dist.all_reduce(grad_out, op=dist.ReduceOp.SUM, group=ctx.tp_group)
        return grad_out, None

class _ReduceFromTPRegion(torch.autograd.Function):
    """ All-reduce in forward, identity in backward (row-parallel exit) """
    @staticmethod
    def forward(ctx, x, tp_group):
        dist.all_reduce(x, op=dist.ReduceOp.SUM, group=tp_group)
        return x
    @staticmethod
    def backward(ctx, grad_out):
        return grad_out, None

class ColumnParallelLinear(nn.Module):
    """ Y = X W,  W ∈ R^{in×out},  shard out dim into T pieces;
        forward: input replicated -> output [B, out/T] sharded along last dim
        backward: input grad must be all-reduced (each TP rank computed partial dX) """
    def __init__(self, in_features, out_features, tp_group, tp_size):
        super().__init__()
        assert out_features % tp_size == 0
        self.tp_group = tp_group
        self.out_per_rank = out_features // tp_size
        self.weight = nn.Parameter(torch.empty(self.out_per_rank, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    def forward(self, x):
        # x: [B, in] (replicated). Entry _CopyToTPRegion lets backward auto all-reduce dX
        x = _CopyToTPRegion.apply(x, self.tp_group)
        return torch.nn.functional.linear(x, self.weight)   # [B, out/T]

class RowParallelLinear(nn.Module):
    """ Y = X W,  W ∈ R^{in×out},  shard in dim into T pieces;
        forward: input [B, in/T] (sharded), output partial sum -> all-reduce
        backward: dW/dX are computed locally, no comm needed """
    def __init__(self, in_features, out_features, tp_group, tp_size):
        super().__init__()
        assert in_features % tp_size == 0
        self.tp_group = tp_group
        self.in_per_rank = in_features // tp_size
        self.weight = nn.Parameter(torch.empty(out_features, self.in_per_rank))
        # Can't call kaiming_uniform_(self.weight) directly: it would auto-derive fan_in
        # from self.weight.size(1) = in_per_rank, inflating the variance by T×.
        # Must compute bound from the global in_features by hand.
        gain = nn.init.calculate_gain('leaky_relu', math.sqrt(5))
        bound = gain * math.sqrt(3.0 / in_features)   # global in_features, not self.in_per_rank
        nn.init.uniform_(self.weight, -bound, bound)
    def forward(self, x):
        # x: [B, in/T] (sharded along last dim)
        local_out = torch.nn.functional.linear(x, self.weight)  # [B, out] partial sum
        return _ReduceFromTPRegion.apply(local_out, self.tp_group)

class TPTransformerMLP(nn.Module):
    """ Megatron-LM standard col+row pairing """
    def __init__(self, d_model, tp_group, tp_size):
        super().__init__()
        d_ff = 4 * d_model
        self.fc1 = ColumnParallelLinear(d_model, d_ff, tp_group, tp_size)   # out [B, L, 4D/T]
        self.fc2 = RowParallelLinear(d_ff, d_model, tp_group, tp_size)      # in [B, L, 4D/T] -> [B, L, D]
    def forward(self, x):
        # x: [B, L, D] (replicated)
        h = torch.nn.functional.gelu(self.fc1(x))   # [B, L, 4D/T]
        return self.fc2(h)                          # [B, L, D] after all-reduce
```

> 💡 **TP communication pairs (must memorize)** — col-parallel: `forward = identity, backward = all-reduce(dX)`; row-parallel: `forward = all-reduce(Y), backward = identity`. A transformer block (col + row × 2 sets: MLP + attention) has 4 all-reduces total — 2 in forward, 2 in backward. This is the source of the 4× count in §7.2.

### 7.5 TP must fit inside a node (NVLink)

Each transformer block forward + backward = 4× all-reduce on activation tensors of size $\approx BLD$. For a 7B model with $D = 4096$ and $B \times L = 2048$ → each is $\approx 32$ MB; per-block × 4 × 32 layers = **128 calls per step, $\approx 4$ GB per step**.

NVLink's 900 GB/s (bidirectional) gives $\approx 450$ GB/s unidirectional, so 4GB transfers in $\approx 4\text{GB}/450\text{GB/s} \approx 8.9$ ms; on IB at 50 GB/s it's $\approx 80$ ms. So **TP must fit inside a node** (NVLink domain). This is why LLaMA / Llama 3 / GPT-3 all use TP=8 (one 8-GPU node) rather than TP=16.

## §8 Sequence Parallel — saving activation memory

Korthikanti et al. **"Reducing Activation Recomputation in Large Transformer Models"** (MLSys 2023).

### 8.1 Motivation: activation memory is mostly element-wise ops like LayerNorm / Dropout

TP shards the intermediate activations of attention / MLP (along head / hidden dim), but the **portion outside TP** (LayerNorm input/output, Dropout, residual) is still **fully replicated** — each TP rank stores a full $[B, L, D]$ activation.

For a 7B model with $B \times L \times D = 2048 \times 4096$, fp16 = 16 MB per activation. A layer has about 4-6 such full-replica activations. 32 layers × 5 → 2.5 GB / GPU wasted.

### 8.2 SP solution: shard along the sequence dim

Shard the non-TP activations along $L$ (sequence) across TP ranks too.

```
TP-only:          [B, L, D]                       [B, L, D]
                  full replica                    full replica
                  ↓                                ↑
                  LayerNorm  → split → Attention → all-reduce → LayerNorm
                                       (TP)       

TP + SP:          [B, L/T, D]                      [B, L/T, D]
                  sharded along L                  sharded along L
                  ↓                                ↑
                  LayerNorm  → all-gather → Attention → reduce-scatter → LayerNorm
                                          (TP)
```

**Key operation changes**:

- TP-only: forward inside attention/MLP requires broadcast / no-op at entry, all-reduce at exit
- TP + SP: at entry **all-gather** (assemble the SP-sharded $L$ back to full length), at exit **reduce-scatter** (reduce the full length back to $L$ shard)

Communication volume: a standard Megatron decoder block has **two** col/row-parallel pairs (attention + MLP), so each transformer block under TP+SP needs 2× all-gather + 2× reduce-scatter in forward (replacing pure-TP forward's 2× all-reduce), and forward + backward together match the 4× all-reduce equivalent of §7. Within each pair, an all-gather + reduce-scatter has the same volume as one all-reduce, so the total is identical to pure TP. **$\boxed{\text{Same total communication as pure TP}}$** — but LayerNorm / Dropout activations drop from $BLD$ to $BLD/T$.

> 💡 **Why SP has the same traffic as pure TP** — because $\text{all-reduce} = \text{reduce-scatter} + \text{all-gather}$, pure TP's tail 1× all-reduce $= $ SP-mode 1× reduce-scatter + the following 1× all-gather (at next block's entry). **Communication is redistributed; total is unchanged, but activation memory is saved**.

### 8.3 How much activation memory does SP save (L3 question)

Let the **non-sharded** single transformer block total activation memory be $A = A_\text{in} + A_\text{out}$, where:

- $A_\text{in}$: the TP-**internal** part (attention's Q/K/V/scores, MLP's intermediate [B, L, 4D]) — TP shards it along the hidden dim
- $A_\text{out}$: the TP-**external** part (LayerNorm/Dropout/residual [B, L, D]) — fully replicated under pure TP

**Pure-TP per-GPU activation**: $A^\text{TP}_\text{per-card} = A_\text{in}/T + A_\text{out}$.

**TP + SP per-GPU activation**: SP also shards the TP-outer part along seq dim (each GPU holds $[B, L/T, D]$) → $A^\text{TP+SP}_\text{per-card} = A_\text{in}/T + A_\text{out}/T = A/T$.

**SP's savings over pure TP**:

$$\boxed{\;A^\text{TP}_\text{per-card} - A^\text{TP+SP}_\text{per-card} = A_\text{out} \cdot \left(1 - \frac{1}{T}\right)\;}$$

For LLaMA-class models, $A_\text{out}$ is about 30-50% of the unsharded total activation. With TP=8, this saves $7/8 \cdot A_\text{out}$ ≈ **a 25-40% reduction in total activation** (consistent with Korthikanti 2023).

### 8.4 Selective activation recompute

Korthikanti et al. in the same paper proposes **selective recompute**: only recompute certain ops (e.g. attention's $QK^\top$ and softmax — the quadratic-memory ones), leaving the rest stored. Compared to full activation checkpointing it reduces compute overhead by 30-40% while saving the same memory.

## §9 Pipeline Parallel

### 9.1 Naive PP and the bubble

Shard $L$ layers along depth into $P$ stages ($L/P$ layers per stage). Naive PP runs each mini-batch sequentially through all stages:

```
Stage 0 (GPU0): [F1]                                  [B1]
Stage 1 (GPU1):       [F1]                       [B1]
Stage 2 (GPU2):              [F1]           [B1]
Stage 3 (GPU3):                    [F1] [B1]
                  ↑ many GPUs idle ↑               ↑ many idle ↑
```

GPU utilization = $1/P$, completely unusable.

### 9.2 GPipe (Huang et al., NeurIPS 2019)

Split the mini-batch into $M$ **micro-batches**; pipeline them:

```
Stage 0: [F1][F2][F3][F4]                                    [B4][B3][B2][B1]
Stage 1:     [F1][F2][F3][F4]                            [B4][B3][B2][B1]
Stage 2:         [F1][F2][F3][F4]                    [B4][B3][B2][B1]
Stage 3:             [F1][F2][F3][F4]            [B4][B3][B2][B1]
         |←─ "warm-up" ─→|       |← all micro-batch backward ─→| "cool-down"
```

**Bubble** (GPU-idle fraction):

$$\boxed{\;\text{bubble ratio} = \frac{P - 1}{M + P - 1}\;}$$

Derivation: each micro-batch traverses $P$ stages in $P$ steps; the warm-up phase (stage $i$ waits for $i$ previous micro-batches) has $P-1$ idle steps; the cool-down phase mirrors it, $P-1$ idle steps; total steps = $M + P - 1$ (forward) + $M + P - 1$ (backward) = $2(M+P-1)$, of which idle = $2(P-1)$. Idle fraction = $(P-1)/(M+P-1)$.

> ⚠️ **GPipe's flaw** — backward can only start after all $M$ micro-batches have completed forward; all micro-batches' activations must be held, so **activation memory grows linearly with $M$**. 1F1B solves this.
>
> Note that this $O(M)$ / $O(P)$ counts microbatch units under a **fixed micro-batch size** (also the standard convention in the 1F1B paper); if instead you fix the total mini-batch $B$ and split it into $M$ pieces (each $B/M$), GPipe's total byte count is actually $O(B)$ (independent of $M$), and 1F1B's peak is $O(PB/M)$ (which *decreases* as $M$ grows) — both conventions are valid, they just differ in which quantity is held fixed as a training hyperparameter.

### 9.3 1F1B / PipeDream (Narayanan SOSP 2019, Megatron-LM-2 SC 2021)

**1F1B = 1 Forward 1 Backward**: each stage, after forwarding a micro-batch, **immediately** backwards the previously completed one:

```
Stage 0: [F1][F2][F3][F4][B1][F5][B2][F6][B3][F7][B4]...
                          ↑ once micro-batch 1's backward is ready (its forward reached stage P)
                            do B1 immediately, freeing micro-batch 1's activation memory
```

**Key properties**:

- Bubble ratio is **still $(P-1)/(M+P-1)$** (warm-up + cool-down unchanged)
- But the number of simultaneously alive activations per stage is **$P$** (not GPipe's $M$) — saving significant activation memory

```
Stage i in steady state has the following activations in memory:
  - micro-batches that finished forward but are awaiting backward
  - because there are P-i stages after i, each running forward / backward one step
  - so stage i holds P-i forwarded-but-not-backwarded activations
  - first stage (i=0) holds the most: P; last stage holds the least: 1
```

### 9.4 Interleaved 1F1B (Megatron-LM-2, SC 2021)

Further **compress the bubble**. Each GPU does not hold $L/P$ consecutive layers but rather $V$ segments of **non-consecutive layers** (**virtual stages**):

```
Original 1F1B (P=4, L=8 layers):
  GPU0: layers 0,1     GPU1: layers 2,3     GPU2: layers 4,5     GPU3: layers 6,7

Interleaved (P=4, V=2, L=8):
  GPU0: layers 0,4     GPU1: layers 1,5     GPU2: layers 2,6     GPU3: layers 3,7
  (each GPU holds V=2 segments, total L/(PV) = 1 layer/segment)
```

Each micro-batch flows through stage 0 (layer 0) → stage 1 (layer 1) → ... → stage 3 (layer 3) → back to stage 0 (layer 4) → ... → stage 3 (layer 7). **A single micro-batch passes through the stage column $V$ times**.

**Bubble ratio** (Narayanan et al., SC 2021, Eq. 4):

$$\boxed{\;\text{interleaved bubble} = \frac{P-1}{V \cdot M + P - 1} \approx \frac{P-1}{V \cdot M}\;}$$

Replace $M$ in the denominator with $V \cdot M$ (the pipeline now has $V$× more virtual-stage passes), and warm-up / cool-down's $P-1$ remains. Cost: **send/recv across GPUs per micro-batch becomes $V$× more** (each micro-batch traverses the stage column $V$ times), so $V$ should not be too large (typically $V = 2$ to $4$).

> 💡 **Intuitive interleaved bubble derivation** — Vanilla 1F1B: bubble = $(P-1)/(M+P-1)$, denominator is micro-batch count $M$ plus warm/cool $P-1$; interleaved: each micro-batch traverses the stage column $V$ times, so the **effective micro-batch count is $V \cdot M$**, giving bubble = $(P-1)/(V M + P - 1) \approx (P-1)/(VM)$. The core intuition is "$V$× more micro-batches".

### 9.5 1F1B schedule pseudo-code

```python
def one_f_one_b_schedule(P, M, stage_rank, num_warmup_microbatches):
    """
    stage_rank: pipeline stage index held by current GPU (0..P-1)
    num_warmup_microbatches = P - 1 - stage_rank
    """
    # ===== Warm-up: stage_rank runs (P-1-stage_rank) forwards
    for i in range(num_warmup_microbatches):
        activation = recv_activation_from_prev_stage()
        out = forward(model, activation)
        send_activation_to_next_stage(out)

    # ===== Steady state: F1B1 alternating
    num_microbatches_remaining = M - num_warmup_microbatches
    for i in range(num_microbatches_remaining):
        # forward
        activation = recv_activation_from_prev_stage()
        out = forward(model, activation)
        send_activation_to_next_stage(out)

        # backward (the one previously forwarded, now arriving)
        grad_out = recv_grad_from_next_stage()
        grad_in = backward(model, grad_out)
        send_grad_to_prev_stage(grad_in)

    # ===== Cool-down: remaining backwards
    for i in range(num_warmup_microbatches):
        grad_out = recv_grad_from_next_stage()
        grad_in = backward(model, grad_out)
        send_grad_to_prev_stage(grad_in)
```

### 9.6 PP communication characteristics

PP only transmits **activations / gradients** at stage boundaries, with each transmission about $(B/M) \cdot L \cdot D \cdot q$ bytes per micro-batch ($q$ = bytes per element, $q=2$ for fp16/bf16). **Traffic is very small** (much less than TP / DP), so PP can cross nodes (IB is enough).

### 9.7 PP's flaws: load imbalance

- Layers / compute per stage need manual balancing (embedding + LM head are heavy and often need special handling)
- One stage's OOM / slowdown blocks the entire pipeline (weakest-link effect)
- Too few micro-batches $M$ → large bubble; too many $M$ → activation memory grows

## §10 Context Parallel — sharding long sequences

As context windows grow from 4K → 128K → 1M, a single GPU cannot hold the full attention computation. **Context Parallel (CP)** shards along the sequence dim.

### 10.1 Ring Attention (Liu et al., arXiv 2310.01889, 2024)

Shard $Q, K, V$ along seq dim across $C$ ranks: each rank holds $L/C$ tokens of Q/K/V.

```
Rank 0: Q[0:L/C],   K[0:L/C],   V[0:L/C]
Rank 1: Q[L/C:2L/C], K[L/C:2L/C], V[L/C:2L/C]
...
```

But attention requires every query to see all keys (causal: all past keys). The **ring attention solution**:

1. Each rank computes a partial attention with its local Q and local K, V
2. Forward its local K, V along the ring to the next rank
3. The next rank computes partial attention with its local Q × the previous rank's K, V, accumulating into output (online-softmax style)
4. After $C$ rotations, every Q has seen all K, V — attention is complete

**Communication**: each rank holds $L/C$ tokens of K, V, size $2 \cdot L/C \cdot D$ (× 2 bytes in fp16). The ring rotates $C-1$ times so every K, V shard visits every rank. **Per-rank total transmission** $\approx (C-1) \cdot 2 L D / C \approx 2 L D$ (almost independent of $C$ — a typical ring property).

**Key**: use **online softmax** (same as FlashAttention) so partial attention can be accumulated without materializing the full $L \times L$ score matrix.

> 💡 **Relationship between Ring attention and FlashAttention** — FlashAttention does block tiling within a single GPU (blocks in SRAM). Ring attention pushes this tiling to **multi-GPU / multi-node level**: each rank holds a K, V block and computes partial attention in ring order, accumulating. The two share the same math — online softmax + block-wise accumulation.

### 10.2 Llama 3 CP implementation

The Llama 3 paper (arXiv 2407.21783) reports using **CP=16** during the **128K long-context** phase (short-context 8K phase doesn't need CP, redirected to DP). Combined with FlashAttention v3, 128K context per-step time goes from untrainable → seconds.

### 10.3 CP and TP orthogonality

- TP shards hidden / head dim → intra-node
- CP shards sequence dim → can cross nodes (traffic $\propto L$, not $L^2$)
- Composed: each GPU holds $L/C$ tokens of a $D/T$-dim sub-tensor

## §11 Expert Parallel — MoE routing

### 11.1 Basic MoE structure

Mixture-of-Experts replaces a single FFN with $E$ expert FFNs + a gate / router:

$$y = \sum_{e=1}^E G_e(x) \cdot \text{Expert}_e(x), \quad G \in \mathbb{R}^E, \quad \sum_e G_e = 1$$

In practice, use **top-K routing**: only pick the top $K$ experts by largest gate output $G$ (typically $K = 1, 2$), with other experts not computed. **Compute is independent of expert count $E$; only depends on $K$** — this is what lets MoE scale parameters.

### 11.2 Expert Parallel: experts across GPUs

| Mode | Experts per GPU | Communication |
|---|---|---|
| Not parallel (replicate) | $E$ | 0 |
| TP-style expert split | shard each expert, all GPUs compute each expert | many all-reduces |
| **EP**: each GPU holds $E/N$ experts | $E/N$ | **all-to-all** dispatch + combine |

EP forward:

```
1. Each GPU computes gates for its token batch → routing decision per token (assigned to e_1, e_2, ..., e_K)
2. all-to-all dispatch: send tokens to the GPU of their assigned expert
   - Input: each GPU holds B/N tokens, each carrying K (expert_id, token_data)
   - Output: each GPU receives the tokens destined for its local experts
3. Each GPU runs FFN on its local experts
4. all-to-all combine: send expert outputs back to the token's original GPU
5. Each GPU merges K expert outputs by gate weight
```

> ⚠️ **EP's two flaws** — (1) **load imbalance**: gates favor a few experts, overloading some GPUs; fix: load balancing loss (Switch Transformer / GShard). (2) **all-to-all traffic is heavy**: proportional to token count × hidden, with inter-node IB as the bottleneck. DeepSeek-V3 uses **node-limited routing** to cap each token's dispatch to at most $M$ nodes, reducing IB traffic.

### 11.3 EP all-to-all code skeleton (pseudo-code)

Below is the **pseudo-code skeleton** of EP forward — the focus is the two all-to-all calls for dispatch / combine, with routing / bucketing engineering details omitted (production code: Megatron-Core MoE or DeepSpeed-MoE):

```python
def expert_parallel_forward(
    x,                # [B, L, D]
    gate,             # nn.Module: tokens [BL, D] -> (top_k_ids, top_k_w) each [BL, K]
    experts_local,    # E_local local experts (nn.ModuleList)
    ep_group, ep_size, ep_rank,
    E_total, K,
):
    """ Pedagogical: each GPU holds E_local = E_total / ep_size experts """
    B, L, D = x.shape
    tokens = x.reshape(B * L, D)

    # 1. routing: each token picks K experts
    top_k_ids, top_k_w = gate(tokens)             # both [BL, K]

    # 2. expand: top-K duplicates each token K times, one expert_id each
    expanded = tokens.unsqueeze(1).expand(-1, K, -1).reshape(B * L * K, D)
    expand_ids = top_k_ids.reshape(B * L * K)     # [BL*K]
    expand_w   = top_k_w.reshape(B * L * K)

    # 3. compute which EP rank each duplicated token should go to
    target_rank = expand_ids // (E_total // ep_size)        # [BL*K]

    # 4. sort by target_rank + count send_count per rank
    perm = torch.argsort(target_rank)
    sorted_tokens = expanded[perm]
    send_counts = torch.bincount(target_rank, minlength=ep_size).tolist()

    # 5a. exchange send_counts -> recv_counts (each rank tells others how many to expect)
    send_t = torch.tensor(send_counts, dtype=torch.int64, device=x.device)
    recv_t = torch.empty_like(send_t)
    dist.all_to_all_single(recv_t, send_t, group=ep_group)         # one small a2a to sync counts
    recv_counts = recv_t.tolist()

    # 5b. sync token expert_ids (for assigning to local experts)
    sorted_ids = expand_ids[perm]
    received_ids = torch.empty(sum(recv_counts), dtype=sorted_ids.dtype, device=x.device)
    dist.all_to_all_single(received_ids, sorted_ids,
                           output_split_sizes=recv_counts,
                           input_split_sizes=send_counts,
                           group=ep_group)

    # 5c. sync token data
    received_tokens = torch.empty(sum(recv_counts), D, device=x.device, dtype=x.dtype)
    dist.all_to_all_single(received_tokens, sorted_tokens,
                           output_split_sizes=recv_counts,
                           input_split_sizes=send_counts,
                           group=ep_group)

    # 6. local expert compute
    received_out = torch.zeros_like(received_tokens)
    for local_eid in range(len(experts_local)):
        global_eid = ep_rank * (E_total // ep_size) + local_eid
        mask = (received_ids == global_eid)
        if mask.any():
            received_out[mask] = experts_local[local_eid](received_tokens[mask])

    # 7. all-to-all combine: reverse, swap split sizes
    combined = torch.empty_like(sorted_tokens)
    dist.all_to_all_single(combined, received_out,
                           output_split_sizes=send_counts,    # reversed
                           input_split_sizes=recv_counts,
                           group=ep_group)

    # 8. inverse permute + gate weight merge
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(perm.numel(), device=perm.device)
    out_expanded = combined[inv_perm]                          # [BL*K, D]
    out_expanded = out_expanded * expand_w.unsqueeze(-1)
    out = out_expanded.view(B * L, K, D).sum(dim=1)            # [BL, D]
    return out.view(B, L, D)
```

> ⚠️ **Production implementations are far more complex** — real code handles (a) capacity factor (preventing expert overload); (b) **dropping tokens** when an expert is full; (c) NVL/IB hierarchical all-to-all (DeepSeek's node-limited routing); (d) backward's mirror all-to-all + gate gradient. This skeleton only illustrates the dispatch / combine bidirectional flow.

## §12 Activation memory optimization

### 12.1 Gradient Checkpointing (Chen et al., arXiv:1604.06174, 2016)

Don't store intermediate activations; recompute on backward. Space $O(\sqrt{L})$ (with optimal segmentation), time +33% (one extra forward).

```python
from torch.utils.checkpoint import checkpoint

def block_forward(x):
    return transformer_block(x)

# Backward recomputes block_forward; no intermediate activation stored
y = checkpoint(block_forward, x, use_reentrant=False)
```

### 12.2 Selective recompute (Korthikanti 2023)

Only recompute **quadratic-memory ops** (attention's $QK^\top$ and softmax); store everything else. Reduces 30-40% compute overhead compared to full recompute while saving the same memory. Megatron-LM enables this by default.

### 12.3 Offload (ZeRO-Infinity)

Offload activations / optimizer state to CPU RAM or NVMe. CPU offload is suitable for 13B-30B single-machine training; NVMe offload has very low throughput and is mainly used to explore the upper bound of trainable parameter scale on a single machine, not for production training or inference.

### 12.4 Activation memory formula (must memorize)

A single transformer block's activations are roughly (fp16, full save):

$$A_\text{block} \approx \underbrace{34 \cdot B \cdot L \cdot D}_{\text{LayerNorm/QKV/output/MLP residual}} + \underbrace{5 \cdot B \cdot L^2 \cdot H}_{\text{attention intermediates}}$$

See Korthikanti et al. 2023 Table 2 for details. When $L \gg D$, the second term dominates (FlashAttention kills this part); when $L \approx D$, the first term dominates (SP reduces it by a factor of $T$).

## §13 Synthesis: 3D / 4D / 5D Parallelism

Combine the dimensions. With world size $W$:

$$W = D_{DP} \times T_{TP} \times P_{PP} \times C_{CP} \times E_{EP}$$

Roles of each axis:

| Axis | Shards what | Where | Communication primitive |
|---|---|---|---|
| DP / FSDP | batch + model states | OK across nodes via IB | all-reduce / reduce-scatter + all-gather |
| TP | hidden / head within a layer | **intra-node NVLink** | all-reduce (4× per block) |
| SP | LayerNorm / Dropout activations outside TP | intra-node (shares group with TP) | reduce-scatter + all-gather |
| PP | layer depth | OK across nodes via IB | point-to-point send/recv |
| CP | sequence dim | either intra-node or across nodes (traffic $\propto L$) | ring K/V |
| EP | MoE experts | OK across nodes via IB (heavy all-to-all) | all-to-all |

### 13.1 Llama 3 405B training topology (public)

Meta 2024 ([2407.21783](https://arxiv.org/abs/2407.21783)):

- 16K H100 GPUs (16384 total)
- Parallelism (total GPU count constant; CP up means DP down):
  - **Short-context phase (8K)**: TP=8 × CP=1 × PP=16 × DP=128 = 16384
  - **Long-context phase (128K)**: TP=8 × CP=16 × PP=16 × DP=8 = 16384
- Training precision: **BF16** (paper Table reports BF16 MFU); FP8 is used for inference quantization, **not for 405B training**
- **54 days, 466 interruptions** (419 unexpected + 47 planned/maintenance), 78% of unexpected ones from hardware causes
- Effective training time > 90%

### 13.2 DeepSeek-V3 training topology (public)

DeepSeek 2024 ([2412.19437](https://arxiv.org/abs/2412.19437)):

- 2048 H800 GPUs
- Parallelism: TP=1 (**no TP**! offset by ZeRO + EP + PP) × PP=16 × EP=64 (across 8 nodes) × ZeRO-1 DP
- **DualPipe** bidirectional pipeline (see next section) + all-to-all overlap
- fp8 GEMM + FP32 accumulation (promoted on CUDA cores)

> 💡 **Why DeepSeek-V3 doesn't use TP** — V3 uses MLA (multi-head latent attention) + many MoE experts; TP-on-head returns are small (the latent attention head dim is already small); and EP's all-to-all overlap with DualPipe hides communication completely. The combination **PP × EP × ZeRO** is enough to fit 671B parameters.

## §14 DualPipe — the 2024 pipeline frontier

DeepSeek 2024's **DualPipe** algorithm, published in the V3 paper and an independent repo (arXiv 2412.19437 + github.com/deepseek-ai/DualPipe).

### 14.1 Core idea

1F1B's bubble comes from the **warm-up + cool-down** phases. DualPipe runs **two directions of the pipeline simultaneously** — one group of micro-batches goes stage 0 → P, the other group goes P → 0; when they meet in the middle, they exactly fill the warm-up / cool-down gaps.

```
Traditional 1F1B (P=4):
Stage 0: [F1][F2][F3][F4][B1][F5][B2]...
Stage 1:     [F1][F2][F3][F4][B1][F5][B2]...
                                         (bubbles at both ends)

DualPipe (P=4):
Forward-direction micro-batch:  [F1][F2][F3][F4]...
Reverse-direction micro-batch: ...[F4'][F3'][F2'][F1']
Stage 0: simultaneously processes forward's F + reverse's F' + corresponding B  ← compute / comms fully overlap
```

More precisely: DualPipe designs a **bidirectional schedule** in which every GPU at every time has two micro-batches in overlapped forward / backward execution; expert-parallel all-to-all communication is also hidden in the gaps of the two micro-batch streams.

### 14.2 DualPipe vs vanilla 1F1B properties

| Dimension | 1F1B | Interleaved 1F1B | DualPipe |
|---|---|---|---|
| Bubble | $(P-1)/(M+P-1)$ | $(P-1)/(VM+P-1)$ | **approaches 0 / greatly reduced** (warm-up / cool-down complementary; not a strict zero-bubble schedule) |
| Activation memory | $P$ × per stage | $V \cdot P$ × | $\approx 2 \times P$ |
| Communication overlap | partial | same as 1F1B | **near-100% all-to-all overlap** |
| Implementation complexity | medium | high | very high (needs forward & reverse scheduling) |

### 14.3 DualPipe costs

- **Massive code complexity**: a stage simultaneously runs forward (direction 1) + forward (direction 2) + backward; CUDA stream management is very hard
- **2× activation memory**: both directions must store activations, $\approx 2\times$ over 1F1B
- **Stages must be balanced**: any stuck stage blocks both directions

DeepSeek uses DualPipe because EP all-to-all traffic is so heavy it must be hidden. **For ordinary training, vanilla 1F1B is sufficient**.

## §15 TorchTitan — PyTorch-native 4D platform

Liang et al. (ICLR 2025, arXiv 2410.06511) **"TorchTitan: One-stop PyTorch Native Solution for Production-Ready LLM Pre-training"**.

### 15.1 Design goals

- **No more monkey-patching**: integrate FSDP2 / TP / PP / SP / CP / Float8 / `torch.compile` into PyTorch mainline
- **DTensor as the unified language**: all sharding described by DTensor placement (`Shard(d)`, `Replicate`, `Partial`)
- **Composable**: FSDP2 composes naturally with TP on native DTensor; composing FSDP1 with TP is engineering-hard (flat-buffer layout conflicts with TP's different shard axis, requiring special adaptation, e.g. splitting TP submodules into separately-wrapped, fine-grained FSDP units and hand-managing the interaction), at a cost far higher than FSDP2 — but not mathematically impossible

### 15.2 Code style (vs DeepSpeed monkey patch)

```python
# TorchTitan style: declarative DTensor placement
import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    parallelize_module, RowwiseParallel, ColwiseParallel,
    SequenceParallel, PrepareModuleInput,
)
from torch.distributed.fsdp import fully_shard

# 1. Build the 4D device mesh
mesh = init_device_mesh(
    "cuda",
    mesh_shape=(2, 8, 4, 8),                # PP=2, FSDP=8, CP=4, TP=8
    mesh_dim_names=("pp", "fsdp", "cp", "tp"),
)

# 2. Apply TP + SP (declare sharding strategy per sub-module)
parallelize_module(
    model,
    mesh["tp"],
    {
        "attn.wq": ColwiseParallel(),
        "attn.wk": ColwiseParallel(),
        "attn.wv": ColwiseParallel(),
        "attn.wo": RowwiseParallel(),
        "mlp.fc1": ColwiseParallel(),
        "mlp.fc2": RowwiseParallel(),
        "norm1":   SequenceParallel(),
        "norm2":   SequenceParallel(),
    },
)

# 3. FSDP2 wrap (auto-detects mesh["fsdp"])
for block in model.blocks:
    fully_shard(block, mesh=mesh["fsdp"])
fully_shard(model, mesh=mesh["fsdp"])

# 4. PP (pipeline schedule, 1F1B interleaved, pseudo-illustration)
# Real ScheduleInterleaved1F1B needs List[PipelineStage] (each rank holds V virtual stages)
from torch.distributed.pipelining import PipelineStage, ScheduleInterleaved1F1B
stages = [PipelineStage(submod_v0, ...), PipelineStage(submod_v1, ...)]  # V=2 virtual stages
schedule = ScheduleInterleaved1F1B(stages, n_microbatches=32, loss_fn=loss_fn)
```

### 15.3 Float8 training (Hopper / Blackwell)

TorchTitan integrates **Float8 training** (H100/B100 fp8 GEMM) — weights / activations in fp8, accumulation in fp32. Typical torchao usage (refer to current torchao docs for exact API):

```python
import torch.nn as nn
from torchao.float8 import convert_to_float8_training, Float8LinearConfig

convert_to_float8_training(
    model,
    config=Float8LinearConfig(),    # default dynamic scaling
    module_filter_fn=lambda m, n: isinstance(m, nn.Linear) and "lm_head" not in n,
)
```

Effect: H100 throughput +20-40%; loss is nearly identical (block-wise scaling is the key).

### 15.4 Async checkpointing

```python
import torch.distributed.checkpoint as DCP

# Async save: returns a Future, does not block training
future = DCP.async_save(model.state_dict(), checkpoint_id="step_10000")
# ... continue training
future.result()                 # wait if needed
```

DCP (Distributed Checkpoint) combined with FSDP2's sharded state dict makes **single checkpoint writes take only a few seconds** (each rank writes its portion to distributed storage), without blocking training.

## §16 Communication primitives summary

To recall quickly in interviews, a closing table.

| Operation | When the all-rank total is $S$ (each rank has $S$) | Per-rank traffic (ring) | Used in |
|---|---|---|---|
| `all-reduce(buf, SUM)` | $N \cdot S$ → all ranks $S$ | $2(N-1)/N \cdot S$ | DDP gradient sync, TP block tail |
| `reduce-scatter(buf)` | $N \cdot S$ → each rank $S/N$ | $(N-1)/N \cdot S$ | ZeRO grad reduce, SP exit |
| `all-gather(shard)` | $N \cdot S/N$ → all ranks $S$ | $(N-1)/N \cdot S$ | ZeRO-3 forward param, SP entry |
| `broadcast(buf, src)` | rank src's $S$ → all ranks $S$ | $S$ (tree) / $(N-1)/N \cdot S$ (ring) | model init broadcast |
| `all-to-all(buf)` | $N \cdot N$ blocks → transpose | $(N-1)/N \cdot S$ | MoE EP routing, sequence sharding transform |
| `point-to-point send/recv` | 1 → 1 | $S$ | PP stage boundary |

## §17 25 frequently-asked interview questions

Ordered by L1 / L2 / L3, with collapsible answer key points.

### L1 must-know (any ML engineer role will ask)

<details>

<summary>Q1. Difference between DDP and DP (DataParallel)?</summary>

- DP (`nn.DataParallel`): single-process multi-GPU; main rank scatters input + gathers output; **GIL + main-rank bottleneck**, deprecated
- DDP (`DistributedDataParallel`): multi-process multi-GPU, one GPU per process, NCCL all-reduce for grad sync
- DDP is 1.5-3× faster than DP and scales far better

Footgun: saying DP is "fast" — wrong, DP is a legacy API; production uses DDP.

</details>

<details>

<summary>Q2. NCCL all-reduce traffic?</summary>

- Ring algorithm: per-GPU total traffic $2(N-1)/N \cdot S \approx 2S$ bytes
- Nearly independent of $N$ (the ring's essence)
- Equivalent to reduce-scatter ($S$) + all-gather ($S$)

Footgun: saying $N \cdot S$ or $S/N$; forgetting the ring step count and per-step traffic.

</details>

<details>

<summary>Q3. Per-GPU model states under Adam mixed-precision training?</summary>

- Parameters fp16: $2\Phi$
- Gradients fp16: $2\Phi$
- Optimizer (fp32 master + Adam m + v): $4\Phi + 4\Phi + 4\Phi = 12\Phi$
- **Total $16\Phi$ bytes**

Footgun: forgetting the fp32 master copy; or treating Adam $m, v$ as fp16 (they're fp32).

</details>

<details>

<summary>Q4. What do ZeRO 1/2/3 shard?</summary>

- ZeRO-1: shard optimizer state
- ZeRO-2: + gradient
- ZeRO-3: + parameter (most aggressive)

Per-GPU model states drop from $16\Phi$ to $16\Phi/N$ (ZeRO-3).

Footgun: getting the order wrong; not knowing ZeRO-1/2 traffic is the same as DDP ($2\Phi$).

</details>

<details>

<summary>Q5. Difference between FSDP and ZeRO-3?</summary>

- Algorithmically: **completely equivalent** (FSDP is PyTorch's ZeRO-3 implementation)
- Engineering: FSDP is integrated in the PyTorch mainline; ZeRO-3 is in the DeepSpeed library
- FSDP2 uses per-parameter DTensor sharding, composing naturally with TP, with simpler state_dict

Footgun: saying "FSDP has less traffic than ZeRO" or vice versa — wrong, the traffic is the same.

</details>

<details>

<summary>Q6. Which dim does Tensor Parallel shard attention along?</summary>

- Shard the **head dim** (each GPU holds $H/T$ heads)
- $W_Q, W_K, W_V$ are column-parallel (shard output dim)
- $W_O$ is row-parallel (shard input dim)
- Each attention block forward 1× all-reduce, backward 1× all-reduce

Footgun: saying it shards inside hidden_dim (splitting one head's $d_{head}$ across ranks); or forgetting that col + row pairing makes the middle communication-free.

</details>

<details>

<summary>Q7. Where must TP be placed?</summary>

- **Intra-node NVLink domain** (900 GB/s bandwidth)
- Going across nodes via IB (50 GB/s) crushes TP performance
- This is why TP=8 is the golden size (one node, 8 GPUs)

Footgun: saying TP can cross nodes arbitrarily; or confusing intra-node / inter-node bandwidths.

</details>

<details>

<summary>Q8. How is Pipeline Parallel's bubble computed?</summary>

- Naive PP: $M = 1$, bubble = $(P-1)/P$ (huge)
- GPipe / 1F1B with $M$ micro-batches: $(P-1)/(M+P-1)$
- Typically need $M \geq 4P$ for bubble < 20%

Footgun: forgetting $M$ in the denominator; saying only $1/P$ without mentioning $M$.

</details>

<details>

<summary>Q9. What advantage does 1F1B have over GPipe?</summary>

- **Same bubble ratio** $(P-1)/(M+P-1)$
- But 1F1B's simultaneously-alive activation count per stage = $P$ (not GPipe's $M$)
- Saves significant **activation memory**

Footgun: saying 1F1B reduces the bubble — wrong, 1F1B doesn't reduce bubble; it saves activations.

</details>

<details>

<summary>Q10. Cost of activation checkpointing (gradient checkpointing)?</summary>

- Memory: $O(L) \to O(\sqrt{L})$
- Time: +33% (one extra forward)
- Production almost always enables this (70B+ models OOM without it)

Footgun: saying "memory is halved" — imprecise, theory is $\sqrt{L}$; or saying "time is halved".

</details>

### L2 advanced (research-oriented roles)

<details>

<summary>Q11. Why does ZeRO-3 forward need all-gather?</summary>

- Parameters are sharded across $N$ GPUs → each GPU holds $1/N$
- A layer forward needs the full $W^{(\ell)}$ → temporarily **all-gather** to all ranks
- Forward completes → **release** immediately, keep only the shard
- Backward needs to all-gather again (released after forward)

Footgun: thinking parameters are resident on all ranks; or forgetting backward also all-gathers.

</details>

<details>

<summary>Q12. Derive: DDP vs ZeRO-3 traffic difference.</summary>

- DDP: backward 1× all-reduce = $2\Phi$, total $2\Phi$
- ZeRO-3: forward 1× all-gather ($\Phi$) + backward 1× all-gather ($\Phi$) + 1× reduce-scatter ($\Phi$) = $3\Phi$
- **ZeRO-3 has 50% more traffic than DDP, in exchange for $N\times$ memory reduction**

Footgun: thinking ZeRO reduces traffic — wrong, ZeRO reduces memory, may increase traffic (depending on the stage).

</details>

<details>

<summary>Q13. NCCL ring all-reduce single-step traffic?</summary>

- Total steps: $2(N-1)$ (reduce-scatter $N-1$ + all-gather $N-1$)
- Per step: each rank sends $S/N$ bytes
- Per-GPU total: $2(N-1) \cdot S/N \approx 2S$
- Bandwidth utilization $\to 1$ as $N \to \infty$

Footgun: saying $S$ instead of $2S$; or saying $N \cdot S$ (forgets ring's essence).

</details>

<details>

<summary>Q14. What does SP (Sequence Parallel) save?</summary>

- Does not save traffic (same total as pure TP)
- Saves **TP-external activation memory** (LayerNorm / Dropout)
- Shards the fully-replicated $[B, L, D]$ to $[B, L/T, D]$
- Total activation memory drops 25-40%

Footgun: saying "SP reduces traffic" — wrong, SP only redistributes the all-reduce into reduce-scatter + all-gather (same total), but activation is sharded.

</details>

<details>

<summary>Q15. How does interleaved 1F1B reduce the bubble?</summary>

- Each stage holds $V$ virtual stages (V discontinuous layer segments)
- Bubble: $(P-1)/(VM+P-1) \approx (P-1)/(VM)$
- For the same $M$, bubble drops $V$×
- Cost: communication count × $V$

Footgun: saying "interleaved reduces compute" — wrong, it only redistributes the time axis; or forgetting the communication × V cost.

</details>

<details>

<summary>Q16. EP all-to-all traffic in MoE?</summary>

- Each token picks $K$ experts
- Dispatch: each GPU sends its $B/N$ tokens to the appropriate expert GPU
- Combine: reverse
- Per-rank total traffic $\approx 2 \cdot K \cdot B/N \cdot D$ (bidirectional all-to-all)

Footgun: forgetting top-K (not $E$); forgetting that dispatch + combine = 2 calls.

</details>

<details>

<summary>Q17. HSDP vs FSDP difference?</summary>

- FSDP: all GPUs in the same sharding group
- HSDP: FSDP / ZeRO-3 within groups, DDP across groups
- HSDP has less intra-group communication (intra-node NVLink); inter-group uses grad all-reduce (no weight all-gather)
- Trade-off: more model states intra-group (not sharded to full world), exchanged for less inter-node communication

Footgun: thinking HSDP reduces total traffic — strictly, it reduces inter-node traffic, with intra-group / inter-group being a trade-off.

</details>

<details>

<summary>Q18. What parallelism did Llama 3 405B use?</summary>

- 16K H100s (16384 total, constant; CP up means DP down)
- Short context (8K): TP=8 × CP=1 × PP=16 × DP=128
- Long context (128K): TP=8 × CP=16 × PP=16 × DP=8
- 54 days, 466 interruptions (419 unexpected + 47 planned), 90%+ effective training time
- Training in BF16 (not FP8 — FP8 is for inference quantization)
- Meta 2024, arXiv 2407.21783

Footgun: saying it used EP — wrong, Llama 3 is dense, no EP; or forgetting the CP phase exists.

</details>

<details>

<summary>Q19. What are ZeRO++'s three tricks?</summary>

- **qwZ**: forward all-gather uses int8 quantization (block-wise quant)
- **hpZ**: maintains a secondary ZeRO-3 partition within a node (each GPU stores an extra Φ/H, not a full replica), backward all-gather runs over intra-node NVLink, not across nodes
- **qgZ**: backward gradient reduce-scatter uses INT4 (4-bit) block quantization
- Total traffic drops 4×, throughput on 384 GPUs +116% (Wang 2023)

Footgun: thinking ZeRO++ changes training precision — it only quantizes buffers in the communication path; weights and compute remain fp16/bf16.

</details>

<details>

<summary>Q20. Can a 7B model be trained in fp16 on 8 A100-40Gs?</summary>

- Model states (Adam): $16 \times 7$B $= 112$ GB
- DDP: per-GPU 112 GB, **does not fit** (A100 40G)
- ZeRO-3 / FSDP: per-GPU $112/8 = 14$ GB ✓
- + activation (with checkpoint): several GB ✓
- + workspace: several GB
- **Conclusion: FSDP/ZeRO-3 + activation checkpointing can run it**

Footgun: saying "DDP also works"; or only counting parameter $\Phi$ without model states; or forgetting fp32 master copy.

</details>

### L3 advanced variants (top labs / in-house infra roles)

<details>

<summary>Q21. Derive 1F1B bubble ratio + how interleaved further reduces it.</summary>

- 1F1B: warm-up $P-1$ steps (forward fill) + cool-down $P-1$ steps (backward drain) + steady $M-P+1$ steps
- Total steps $= 2M$ (forward + backward) takes $2(M + P - 1)$ time slots (including $P-1$ idle slots each at warm/cool)
- Bubble = $2(P-1) / [2(M+P-1)] = (P-1)/(M+P-1)$
- **Interleaved 1F1B** (Narayanan SC 2021, Eq. 4): each GPU holds $V$ non-contiguous layer segments (virtual stages); each micro-batch traverses the stage column $V$ times → pipeline has effectively $V \cdot M$ micro-batches in queue
- Bubble = $(P-1) / (V \cdot M + P - 1) \approx (P-1)/(V \cdot M)$
- **For the same $M$, bubble drops $V$×; cost: cross-GPU send/recv ×$V$**

Footgun: forgetting interleaved's communication × V cost; saying "interleaved reduces forward compute" — it only redistributes time.

</details>

<details>

<summary>Q22. FSDP vs ZeRO-3 traffic + applicable scenarios.</summary>

- **Traffic is completely the same**: forward all-gather $\Phi$ + backward all-gather $\Phi$ + reduce-scatter $\Phi$ = $3\Phi$
- Engineering differences:
  - FSDP2: uses DTensor to describe per-param sharding, composes naturally with TP / PP, `torch.compile` friendly
  - DeepSpeed ZeRO-3: uses flat-parameter, needs monkey-patching, but has a complete ZeRO++ / Offload / Infinity ecosystem
- Choice:
  - New project (dense + 4D parallelism) → **FSDP2 / TorchTitan**
  - MoE + cross-framework + offload → **DeepSpeed**

Footgun: saying "FSDP has less traffic than ZeRO" — algorithms are the same; or saying "FSDP doesn't support offload" — FSDP2 OffloadPolicy supports it.

</details>

<details>

<summary>Q23. How much activation memory does TP + SP save vs pure TP?</summary>

- Let transformer-block activation be $A_\text{block}$
- TP-internal (attention intermediates / MLP intermediates): about $A_\text{block} \times 0.5-0.7$, TP shards it
- TP-external (LayerNorm / Dropout / residual): about $A_\text{block} \times 0.3-0.5$, **TP does not shard (full replica)**
- Pure-TP per-GPU activation = $A_\text{TP-in}/T + A_\text{TP-out}$
- TP+SP per-GPU activation = $A_\text{TP-in}/T + A_\text{TP-out}/T = A_\text{block}/T$
- **Saves $A_\text{TP-out} \cdot (1 - 1/T)$**, about **25-40% of total activation** (depends on model)

Footgun: saying "SP reduces activation by $T$×" — imprecise, only for the TP-outer part; or forgetting that SP traffic is the same as pure TP.

</details>

<details>

<summary>Q24. Core improvement of DualPipe over 1F1B?</summary>

- **Bidirectional pipeline**: one group of micro-batches goes stage 0 → P, another P → 0
- Meeting in the middle, they exactly fill 1F1B's warm-up / cool-down bubble
- **Bubble greatly reduced, approaches 0** (ideally the two sides are complementary, but DeepSeek-V3 Table 2 gives a non-zero bubble expression — it is not a strict zero-bubble schedule)
- Key gain: **all-to-all comms fully overlap** (important for EP)
- Cost: activation memory × 2 (both directions store activations); extremely complex implementation
- DeepSeek-V3 December 2024 (arXiv 2412.19437) + github.com/deepseek-ai/DualPipe

Footgun: saying "DualPipe is a new way to shard PP" — it's a new schedule; or forgetting the 2× activation cost.

</details>

<details>

<summary>Q25. How is 4D / 5D parallelism composed in frontier training (e.g. DeepSeek-V3 / Llama 3)?</summary>

- **5 orthogonal dimensions**: DP / FSDP × TP × PP × CP × EP
- World size $W = D \times T \times P \times C \times E$
- Rules of thumb:
  - TP=8 (must fit in one NVLink domain)
  - PP=8-32 (OK across nodes via IB; bubble controlled via $M \geq 4P$)
  - CP depends on context length (Llama 3 uses CP=16 at 128K; 1M may need CP=64+)
  - EP depends on MoE expert count (DeepSeek-V3 uses EP=64)
  - FSDP / DP uses the remaining world size
- **Llama 3 405B** (16K GPUs constant, CP up means DP down):
  - 8K context: TP=8 × CP=1 × PP=16 × DP=128
  - 128K context: TP=8 × CP=16 × PP=16 × DP=8
- **DeepSeek-V3 671B**: TP=1 + PP=16 × EP=64 × ZeRO-1 DP=2, 2048 H800s total
- **Key engineering points**:
  - Place communication primitives on the right topology (TP on NVLink, PP / DP on IB)
  - DualPipe / Interleaved 1F1B to compress PP bubble
  - FlashAttention v3 + SP / CP to compress activations
  - fp8 GEMM (H100 / B100) + FP32 accumulation (promoted on CUDA cores) to lift throughput

Footgun: reversing the dimension order; forgetting TP must be intra-node; thinking DeepSeek uses TP (V3 actually has TP=1).

</details>

## §A Appendix: complete 4D wrap code skeleton (with a placeholder CP dim)

Below is a minimal end-to-end **3D-parallelism (PP × FSDP × TP+SP)** wrap example, in TorchTitan style. The mesh reserves a CP dimension (set to 1 in the code below, i.e. not actually enabled — just a placeholder for extending to true 4D); to demonstrate full 4D, you'd need to set `cp` > 1 and apply context-parallel sharding (e.g. Ring Attention) to attention.

```python
import torch
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    parallelize_module, RowwiseParallel, ColwiseParallel,
    SequenceParallel, PrepareModuleInput, PrepareModuleOutput,
)
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.pipelining import pipeline, SplitPoint, ScheduleInterleaved1F1B
from torch.utils.checkpoint import checkpoint

def build_4d_model(model, world_size):
    """ 3D parallelism wrap (PP × FSDP × TP+SP); mesh reserves a CP dim (currently 1, not actually enabled) """
    # Step 1. Build 4D device mesh
    # e.g. 64 GPUs = PP=2 × FSDP=4 × CP=1 × TP=8
    mesh = init_device_mesh(
        "cuda",
        mesh_shape=(2, 4, 1, 8),
        mesh_dim_names=("pp", "fsdp", "cp", "tp"),
    )

    # Step 2. TP + SP wrap (mesh["tp"])
    tp_plan = {
        # Attention
        "attn.wq":      ColwiseParallel(),
        "attn.wk":      ColwiseParallel(),
        "attn.wv":      ColwiseParallel(),
        "attn.wo":      RowwiseParallel(),
        # MLP
        "mlp.fc1":      ColwiseParallel(),
        "mlp.fc2":      RowwiseParallel(),
        # SP: norm / residual sharded along seq dim
        "ln1":          SequenceParallel(),
        "ln2":          SequenceParallel(),
    }
    for block in model.blocks:
        parallelize_module(block, mesh["tp"], tp_plan)

    # Step 3. Activation checkpoint (one per block)
    for i, block in enumerate(model.blocks):
        model.blocks[i] = _ckpt_wrap(block)

    # Step 4. PP split (mesh["pp"])
    # Split model into 2 segments (PP=2)
    split_spec = {"blocks.16": SplitPoint.BEGINNING}
    pipe = pipeline(
        model,
        mb_args=(torch.randn(1, 4096, device="cuda"),),
        split_spec=split_spec,
    )
    pp_stage = pipe.build_stage(
        stage_index=mesh["pp"].get_local_rank(),
        device=torch.device(f"cuda:{torch.cuda.current_device()}"),
        group=mesh["pp"].get_group(),
    )

    # Step 5. FSDP2 wrap (mesh["fsdp"])
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    )
    for module in pp_stage.submod.modules():
        if isinstance(module, TransformerBlock):
            fully_shard(module, mesh=mesh["fsdp"], mp_policy=mp_policy)
    fully_shard(pp_stage.submod, mesh=mesh["fsdp"], mp_policy=mp_policy)

    return pp_stage, mesh

def _ckpt_wrap(block):
    """ Wrap a TransformerBlock with activation checkpointing """
    class Wrapped(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
        def forward(self, *args, **kw):
            return checkpoint(self.inner, *args, use_reentrant=False, **kw)
    return Wrapped(block)

# ============================================================
# Training loop with PP schedule
# ============================================================
def train_step_4d(pp_stage, schedule, optimizer, batch):
    """ 1 step of 4D parallel training """
    optimizer.zero_grad(set_to_none=True)
    # PP schedule runs forward + backward across all stages
    losses = []
    schedule.step(batch, losses=losses)  # triggers FSDP all-gather / TP all-reduce / etc.; losses is only populated on the last pipeline stage, empty on the rest
    optimizer.step()
    if len(losses) > 0:
        return torch.stack(losses).mean()
    return None  # no local loss to return on non-last stages
```

**Note**: the code above is a pedagogical skeleton; for production, use the TorchTitan repo directly (pytorch/torchtitan) — it includes a complete `Trainer`, checkpointing, profiling, loss / lr schedules. This section only sketches concepts.

## §B References

- **DDP**: PyTorch documentation, `nn.parallel.DistributedDataParallel`
- **ZeRO**: Rajbhandari et al., "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models", SC 2020 ([arXiv:1910.02054](https://arxiv.org/abs/1910.02054))
- **ZeRO-Offload**: Ren et al., USENIX ATC 2021
- **ZeRO-Infinity**: Rajbhandari et al., SC 2021
- **ZeRO++**: Wang et al., "ZeRO++: Extremely Efficient Collective Communication for Giant Model Training", arXiv:2306.10209, 2023
- **FSDP**: Zhao et al., "PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel", VLDB 2023
- **FSDP2 / TorchTitan**: Liang et al., "TorchTitan: One-stop PyTorch Native Solution", ICLR 2025 ([arXiv:2410.06511](https://arxiv.org/abs/2410.06511))
- **Megatron-LM TP**: Shoeybi et al., "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism", arXiv:1909.08053, 2019
- **Megatron-LM-2 (Interleaved 1F1B)**: Narayanan et al., "Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM", SC 2021
- **GPipe**: Huang et al., "GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism", NeurIPS 2019
- **PipeDream / 1F1B**: Narayanan et al., "PipeDream: Generalized Pipeline Parallelism for DNN Training", SOSP 2019
- **Sequence Parallel + Selective Recompute**: Korthikanti et al., "Reducing Activation Recomputation in Large Transformer Models", MLSys 2023
- **Ring Attention**: Liu et al., "Ring Attention with Blockwise Transformers for Near-Infinite Context", ICLR 2024 ([arXiv:2310.01889](https://arxiv.org/abs/2310.01889))
- **Gradient Checkpointing**: Chen et al., "Training Deep Nets with Sublinear Memory Cost", arXiv:1604.06174, 2016
- **DualPipe**: DeepSeek-V3 Technical Report, [arXiv:2412.19437](https://arxiv.org/abs/2412.19437), December 2024
- **Llama 3**: Meta AI, "The Llama 3 Herd of Models", [arXiv:2407.21783](https://arxiv.org/abs/2407.21783), 2024
- **GShard / Switch Transformer (MoE EP)**: Lepikhin et al. 2020 / Fedus et al. JMLR 2022
