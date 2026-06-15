## §0 TL;DR Cheat Sheet

> 💡 **8 sentences to nail LoRA / PEFT** — one page covering the interview essentials (see §2–§10 for derivations).

1. **Core formula**: freeze the pretrained weight $W_0$, learn only a low-rank increment $\Delta W = BA$, forward $h = W_0 x + \frac{\alpha}{r} BA\,x$, where $B \in \mathbb{R}^{d\times r}$, $A \in \mathbb{R}^{r\times k}$, $r \ll \min(d,k)$.

2. **Why low rank works**: pretrained models have a very low "intrinsic dimension" (Aghajanyan 2020) — optimizing in a low-dimensional subspace already approximates full fine-tuning; Hu 2021 hypothesizes accordingly that the weight update $\Delta W$ is approximately low-rank, so a low-rank matrix with $r=4\sim64$ can approximate it (empirical hypothesis + evidence, not a strict theorem).

3. **Initialization**: $A$ random (Kaiming), $B = 0$, so the **training starting point has $\Delta W = 0$** (it does not perturb the pretrained model), yet gradients are nonzero so it can still learn. Zeroing both means it never learns.

4. **Scaling $\alpha/r$**: decouples $r$ from the learning rate, so changing $r$ needs no lr re-tuning. rsLoRA shows that at high rank you should switch to $\alpha/\sqrt{r}$ to prevent gradient collapse.

5. **Zero inference latency**: after training you can **merge** $\frac{\alpha}{r}BA$ into $W_0$ to get $W' = W_0 + \frac{\alpha}{r}BA$; inference is then structurally identical to the original model with no extra latency (the key advantage of LoRA over Adapter / Prefix).

6. **What memory it saves**: mainly the **optimizer states + gradients** (full fine-tuning costs 16 bytes/param under Adam; LoRA pays that only for the ~0.1% trainable params); **activation memory is NOT saved automatically** (you still backprop through the frozen base) — that needs gradient checkpointing.

7. **QLoRA**: quantize the base with **NF4** (4-bit NormalFloat, information-theoretically optimal for normal weights) + **double quantization** + **paged optimizer**, keep LoRA adapters in bf16, and a single 48GB GPU can fine-tune a 65B model.

8. **The family**: DoRA (magnitude–direction decomposition), rsLoRA ($\sqrt{r}$ scaling), PiSSA (principal-component init), AdaLoRA (adaptive rank budget), (IA)³ (scaling vectors), LoRA+ (different lr for A and B) — each patches one weakness of LoRA.

## §1 Intuition: why we need PEFT

**The pain point of full fine-tuning is memory, not compute.** Fine-tuning a $\Psi$-parameter model with Adam in mixed precision costs, per parameter: bf16 weight 2 B + bf16 gradient 2 B + fp32 master weight 4 B + Adam first moment $m$ 4 B + Adam second moment $v$ 4 B = **16 B/param**. For a 7B model that part alone is 112 GB — too big for a single 80GB GPU, and that excludes activations.

The core idea of PEFT (Parameter-Efficient Fine-Tuning): **freeze the vast majority of pretrained parameters and train only a tiny set of new or selected parameters**, compressing the trainable fraction from 100% down to 0.01%–1%. This means:

- optimizer states / gradients are paid only for that tiny trainable set → memory drops sharply;
- one base + many lightweight adapters (tens of MB each) can serve many tasks, hot-swappable;
- on small data it overfits less and suffers milder catastrophic forgetting.

**The low-rank hypothesis is LoRA's theoretical anchor.** Aghajanyan et al. (2020, arXiv 2012.13255) showed empirically that pretrained language models have a very low "intrinsic dimension" — optimizing in a random subspace far smaller than the full parameter count reaches 90% of full fine-tuning's quality. Hu et al. (2021) followed this intuition: since fine-tuning "doesn't travel far," the weight **update** $\Delta W$ should itself be low-rank, so they parameterize it as a product of two thin matrices $BA$.

> 💡 **One-sentence mental model** — Full fine-tuning is "rewriting the whole book"; LoRA is "sticking low-rank sticky notes $\Delta W = BA$ in the margins." At inference you copy the sticky-note content back into the main text (merge), and the reader (the inference engine) never knows the notes existed.

## §2 LoRA core formula and derivation

### 2.1　Main formula and shapes

For an adapted linear layer with frozen original weight $W_0 \in \mathbb{R}^{d \times k}$, LoRA learns a low-rank increment:

$$\boxed{\;h = W_0 x + \Delta W x = W_0 x + \frac{\alpha}{r} B A\, x\;}$$

Shapes:

- $A \in \mathbb{R}^{r \times k}$ (down-projection, compress input to $r$ dims)
- $B \in \mathbb{R}^{d \times r}$ (up-projection, lift from $r$ dims back to $d$)
- $\Delta W = BA \in \mathbb{R}^{d \times k}$, rank $\le r$
- $r \ll \min(d, k)$, typically $r \in \{4, 8, 16, 32, 64\}$
- $\alpha$ is a scaling hyperparameter, $\frac{\alpha}{r}$ is the scaling factor (see §2.3)

The trainable parameter count drops from $d \times k$ to $r(d + k)$. E.g. $d=k=4096$, $r=8$: from 16.8M down to 65.5K, a **256× compression**.

### 2.2　Why $B=0$ and $A$ random init (must-know)

The official LoRA initialization: $A$ is initialized with Kaiming uniform, $B$ is initialized to **all zeros**. Thus:

$$\Delta W_{t=0} = B_0 A_0 = 0 \cdot A_0 = 0 \;\Rightarrow\; h_{t=0} = W_0 x$$

i.e. **the starting point equals the pretrained model exactly**, introducing no perturbation. This matters: if the start already deviates from pretraining, you throw away pretrained knowledge and may blow up the loss.

So why not **zero both**? Because it would never learn. Look at the first-step gradients (loss $L$, output-direction gradient $g = \partial L / \partial h$):

$$\frac{\partial L}{\partial B} = \frac{\alpha}{r}\, g\, (A x)^\top, \qquad \frac{\partial L}{\partial A} = \frac{\alpha}{r}\, B^\top g\, x^\top$$

- $A$ nonzero → generically $Ax \neq 0$ → $\partial L/\partial B \neq 0$: **$B$ can move on step one** (unless $g=0$ or $x$ happens to lie in $A$'s null space, etc.);
- $B = 0$ → $\partial L/\partial A = 0$: $A$ doesn't move on step one, but once $B$ becomes nonzero, $A$ gets gradient on the next step.

If $A=B=0$, both gradients are identically zero and $\Delta W$ stays 0 forever. **So it must be "one zero, one nonzero": $A$ nonzero guarantees learnability, $B=0$ guarantees a clean start.** (Symmetrically, $A=0$ with $B$ random also works; both conventions appear in PEFT libraries — the key is exactly one being zero.)

> ⚠️ **Common misconception** — "$B=0$ so LoRA learns nothing on step one" is wrong. On step one $B$ is indeed updating (nonzero gradient); it is just that $\Delta W$ is numerically 0 at $t=0$. It is "output increment is zero," not "gradient is zero."

### 2.3　Scaling factor $\alpha/r$ and rsLoRA

LoRA scales the increment by $\frac{\alpha}{r} BA$. Hu et al. put it: "treat $\alpha$ like a learning rate, fix it to the first $r$ you try, and don't re-tune lr when you change $r$." Intuition: $\frac{\alpha}{r}$ keeps $\Delta W$'s magnitude roughly comparable across different $r$, thereby **decoupling rank from learning rate**.

But Kalajdzievski (2023, rsLoRA, arXiv 2312.03732) pointed out that $\frac{\alpha}{r}$ **over-shrinks at large $r$**, causing high-rank LoRA to collapse — quality plateaus instead of improving. The analysis: to keep forward/backward activation magnitudes stable as $r$ grows (rank-stabilized), the scaling factor should be $\frac{\alpha}{\sqrt{r}}$:

$$\gamma_r^{\text{LoRA}} = \frac{\alpha}{r} \quad\text{vs}\quad \gamma_r^{\text{rsLoRA}} = \frac{\alpha}{\sqrt{r}}$$

The intuition is variance: each output element of $BA$ is a sum of $r$ terms; if the terms are i.i.d. with equal variance, the standard deviation $\propto \sqrt{r}$. Dividing by $\sqrt{r}$ pulls the magnitude back to a constant; dividing by $r$ over-suppresses, and the larger $r$ the harder the suppression. So **low rank ($r\le 32$) with $\alpha/r$ is usually fine; when you want the high-rank payoff, switch to rsLoRA's $\alpha/\sqrt{r}$**.

### 2.4　Merging and zero inference latency (LoRA's killer feature)

After training, $\Delta W$ can be **merged once** into the base weight:

$$W' = W_0 + \frac{\alpha}{r} BA$$

Inference then uses only $W'$, and the forward $h = W' x$ is **structurally identical** to the original linear layer — no extra matmul, no extra latency, no extra memory. This is LoRA's fundamental advantage over Adapter (inserts serial submodules) and Prefix-Tuning (occupies sequence length / KV): **both add inference overhead, LoRA after merging adds zero.**

When you need to revert to base or switch adapters, just **unmerge**: $W_0 = W' - \frac{\alpha}{r}BA$. For multi-task serving, a common approach is to **not merge** and add $\frac{\alpha}{r}B(Ax)$ online, so one base can host multiple dynamically-routed adapters (at the cost of restoring a little inference overhead).

> ⚠️ **QLoRA cannot merge directly** — the base is NF4-quantized; adding the bf16 $\Delta W$ into 4-bit weights requires dequantize → merge → (optionally) re-quantize, which loses precision. In practice one usually **dequantizes the base to fp16 and then merges** rather than merging back into 4-bit. See §5.5.

### 2.5　Which matrices to adapt / how to choose $r$

The original paper applied LoRA to the attention projections and found by ablation: **at equal parameter budget, adapting $W_q, W_v$ beats only $W_q$; spreading the budget across more matrices ($q,k,v,o$) usually beats piling high rank onto a few matrices.** Later practice (e.g. QLoRA) further recommends **adding LoRA to all linear layers** (including the FFN gate/up/down), which is often more stable than attention-only.

- $r$ choice: simple tasks / small data $r=4\sim8$; hard tasks / large data $r=16\sim64$. Too large $r$ not only saves little but may overfit or hit the $\alpha/r$ collapse (see rsLoRA).
- $\alpha$ heuristic: commonly $\alpha = 2r$ (scaling $\approx 2$) or $\alpha = r$ (scaling $=1$), tuned on validation.
- target_modules: attention `q_proj,k_proj,v_proj,o_proj` + FFN `gate_proj,up_proj,down_proj` is the common all-in recipe for LLaMA-style models.

## §3 Implementing LoRALinear from scratch

Below is a runnable `LoRALinear`: it wraps a frozen `nn.Linear`, adds $A,B$ with scaling, and supports merge / unmerge.

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Frozen base linear + low-rank bypass BA. forward: W0 x + (alpha/r) B A x."""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16,
                 dropout: float = 0.0, rslora: bool = False):
        super().__init__()
        assert r > 0
        self.base = base
        for p in self.base.parameters():       # freeze the base
            p.requires_grad_(False)

        in_f, out_f = base.in_features, base.out_features
        self.r, self.alpha = r, alpha
        # scaling: standard alpha/r; rsLoRA uses alpha/sqrt(r)
        self.scaling = alpha / (math.sqrt(r) if rslora else r)

        # A: [r, in] Kaiming random; B: [out, r] zero -> start with ΔW = 0
        self.lora_A = nn.Parameter(torch.empty(r, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.merged = False

    def forward(self, x):
        out = self.base(x)                      # frozen main path W0 x (+bias)
        if not self.merged:
            lora = self.lora_dropout(x) @ self.lora_A.t() @ self.lora_B.t()
            out = out + self.scaling * lora      # add the bypass
        return out

    @torch.no_grad()
    def merge(self):
        """Add scaling*BA (standard alpha/r or rsLoRA alpha/sqrt(r)) into base.weight; zero inference latency."""
        if self.merged:
            return
        dW = self.scaling * (self.lora_B @ self.lora_A)   # [out, in]
        self.base.weight.add_(dW.to(self.base.weight.dtype))
        self.merged = True

    @torch.no_grad()
    def unmerge(self):
        if not self.merged:
            return
        dW = self.scaling * (self.lora_B @ self.lora_A)
        self.base.weight.sub_(dW.to(self.base.weight.dtype))
        self.merged = False


def inject_lora(model: nn.Module, target_names=("q_proj", "v_proj"),
                r=8, alpha=16, dropout=0.0):
    """Replace nn.Linear modules in `model` whose name matches target_names with LoRALinear."""
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and child_name in target_names:
                setattr(module, child_name,
                        LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
    return model
```

Key points:

- Note `x @ A.t() @ B.t()`: $x$ has shape $[\dots, k]$, $A^\top$ is $[k, r]$, $B^\top$ is $[r, d]$, output $[\dots, d]$, aligned with the base output.
- `merge()` uses `@torch.no_grad()` + `add_` to modify `base.weight` in place; after merging, `forward` skips the bypass.
- dropout acts only on the **input entering the bypass**, not the main path (the standard LoRA practice).

> 💡 **dropout on the bypass input** — LoRA dropout acts on $x$ before it enters $A$, acting as regularization on the low-rank adaptation; the main path $W_0 x$ has no dropout (the base is frozen and needs no regularization).

## §4 The memory account: where exactly does LoRA save

Fine-tuning a $\Psi$-parameter model with Adam, broken down by bytes (mixed-precision training, per param):

| Component | Full fine-tuning | LoRA (bf16 base) |
| --- | --- | --- |
| Base weight (bf16) | $2\Psi$ | $2\Psi$ (frozen, still resident) |
| Base gradient | $2\Psi$ | $0$ |
| Base fp32 master | $4\Psi$ | $0$ |
| Base Adam $m,v$ (fp32) | $8\Psi$ | $0$ |
| LoRA params + grad + optimizer | — | $16 \cdot \Psi_{\text{lora}}$ |
| **Trainable-related total** | $\approx 16\Psi$ | $\approx 2\Psi + 16\Psi_{\text{lora}}$ |

where $\Psi_{\text{lora}} \ll \Psi$ (typically 0.1%–1%). So LoRA cuts almost all of full fine-tuning's "16 bytes/param" big chunk, leaving only the $2\Psi$ resident base + a tiny bit of LoRA overhead.

**But one thing must be stated explicitly: LoRA does NOT save activation memory automatically.** Because the bypass $BA$ hangs on a frozen layer, backprop still flows through the whole network to compute $\partial L / \partial x$, so intermediate activations are still stored. LoRA saves **optimizer states + gradients + master weights**, not activations. To save activations you additionally need **gradient checkpointing** (trading compute for activation memory).

Trainable parameter count (adapting $M$ linear layers, layer $i$ is $d_i \times k_i$):

$$\Psi_{\text{lora}} = \sum_{i=1}^{M} r_i (d_i + k_i)$$

**Concrete example**: LLaMA-7B, adapting $q,k,v,o$ (each $4096\times4096$), $r=8$, 32 layers:

$$\Psi_{\text{lora}} = 32 \times 4 \times 8 \times (4096+4096) = 8.39\text{M} \approx 0.12\% \text{ of } 7\text{B}$$

Memory comparison (7B, single GPU, ignoring activations):

- Full FT Adam: $7\text{e}9 \times 16 \approx 112$ GB (won't fit a single GPU)
- LoRA (bf16 base): $7\text{e}9 \times 2 + 8.4\text{e}6 \times 16 \approx 14$ GB + 0.13 GB ≈ **14 GB**
- QLoRA (NF4 base): $7\text{e}9 \times 0.5\,\text{B/param} + \text{overhead} \approx 3.5$ GB + LoRA ≈ **<5 GB** (see §5; NF4 = 4 bit/param = 0.5 **byte**/param)

## §5 QLoRA: 4-bit base + LoRA

QLoRA (Dettmers et al., 2023, arXiv 2305.14314, NeurIPS 2023) lets a single 48GB GPU fine-tune a 65B model. The core is three pieces: **NF4 quantization** + **double quantization** + **paged optimizer**, plus the "frozen 4-bit base, bf16 LoRA" recipe.

### 5.1　NF4: 4-bit NormalFloat (information-theoretically optimal)

Observation: neural network weights approximately follow a zero-mean normal distribution $\mathcal{N}(0,\sigma^2)$. Plain 4-bit integer quantization (INT4, equally-spaced bins) wastes bins on a normal distribution — tail bins receive almost no values. **NF4's idea is to make each quantization bin receive an equal number of weights (equal probability mass); under the assumption "weights are zero-mean normal + quantile quantization," this is information-theoretically optimal for that fixed distribution.**

Construction (simplified): take $2^4=16$ quantile points of the standard normal $\mathcal{N}(0,1)$ as quantization levels so that adjacent levels carry equal probability mass; make it **asymmetric** so 0 is represented exactly (zero-preserving, friendly to pruning / padding). When quantizing, normalize weights to $[-1,1]$ per block (QLoRA uses block size 64) via the absmax, then look up the nearest NF4 level:

$$w \;\xrightarrow{\text{normalize by absmax}}\; \hat{w} \in [-1,1] \;\xrightarrow{\text{nearest NF4 level}}\; q \in \{n_0,\dots,n_{15}\}$$

Dequantization: $w \approx c \cdot n_q$, where $c$ is the block's absmax scaling constant.

> 💡 **NF4 vs INT4 vs FP4** — INT4 is equally spaced and wastes tail bins on normal data; FP4 assigns non-uniform levels via a float exponent but doesn't necessarily match the weight distribution; NF4 places levels directly at normal quantiles, an optimal match to the fact that "weights are nearly normal." The paper measures NF4 clearly beating INT4/FP4.

### 5.2　Double Quantization

Each NF4 block (64 weights) stores one fp32 absmax scaling constant $c$, which amortizes to $32/64 = 0.5$ bit/param of extra overhead. **Double quantization quantizes these scaling constants themselves once more**: quantize $c$ (fp32) with block size 256 into 8-bit, with the second-level scaling constant in fp32:

$$\text{overhead}: \underbrace{0.5\,\text{bit/param}}_{\text{single}} \;\to\; \underbrace{\frac{8}{64} + \frac{32}{64\times256}}_{\text{double}} \approx 0.127\,\text{bit/param}$$

That saves about **0.37 bit per param** on average — about 3 GB for a 65B model, enough to decide whether it fits on one card.

### 5.3　Paged Optimizer

With long sequences / large batches, optimizer states can spike memory and OOM. QLoRA uses NVIDIA **unified memory** to **page** the optimizer states: when memory is tight it pages optimizer states out to CPU memory and back when needed, like OS memory paging, avoiding OOM crashes.

### 5.4　How forward / backward flow

Key: **the base is stored as NF4, but dequantized to bf16 per block for the matmul**; gradients flow only to the bf16 LoRA params, the NF4 base stays frozen with no gradient. So QLoRA training memory ≈ NF4 base (4 bit/param = 0.5 byte/param, plus ~0.127 bit/param quantization constants after double quant) + bf16 LoRA optimizer + activations.

```python
# Conceptual skeleton of the QLoRA forward (not a full kernel, just the data flow)
def qlora_linear_forward(x, nf4_weight, absmax, lora_A, lora_B, scaling):
    # 1) base: dequantize NF4 -> bf16, then main-path matmul (base has no gradient)
    W = dequantize_nf4(nf4_weight, absmax).to(x.dtype)   # [out, in] bf16
    base_out = x @ W.t()
    # 2) bypass: LoRA in bf16, trainable
    lora_out = (x @ lora_A.t() @ lora_B.t()) * scaling
    return base_out + lora_out
```

### 5.5　The QLoRA merge pitfall

As in §2.4, the NF4 base cannot losslessly merge the bf16 $\Delta W$. Standard practice: **dequantize the base → merge LoRA → save as fp16/bf16** (yielding a full-precision merged model), or just keep "NF4 base + unmerged LoRA bypass" at inference. Quantizing $\Delta W$ back to NF4 and adding it introduces noticeable error.

> ⚠️ **QLoRA ≠ inference memory saving** — QLoRA solves fitting a large model on one card **during fine-tuning**. At inference, if you dequantize-and-merge to fp16 the memory goes back to full precision; to also save inference memory you need a dedicated 4-bit inference path (bitsandbytes / GPTQ / AWQ, see the quantization sheet).

## §6 DoRA: magnitude–direction decomposition

DoRA (Weight-Decomposed Low-Rank Adaptation, Liu et al., 2024, arXiv 2402.09353, ICML 2024) observed a systematic difference between full fine-tuning and LoRA in "how the **magnitude** and **direction** of the weights co-evolve," and that LoRA's expressive pattern is limited. DoRA explicitly decomposes the pretrained weight into magnitude and direction, learning them separately.

Decompose the weight **per output neuron (row)** (following weight normalization: each output's weight vector = magnitude × unit direction; $\lVert\cdot\rVert_r$ is the per-row / per-output 2-norm, i.e. over the in dimension `dim=1` for a PyTorch weight $[\text{out},\text{in}]$):

$$W_0 = m \odot \frac{V}{\lVert V \rVert_r}, \qquad m = \lVert W_0 \rVert_r \in \mathbb{R}^{d\times 1}, \quad V = W_0$$

where $m$ is **one scalar magnitude per output neuron** ($d$-dim) and $V/\lVert V\rVert_r$ is the per-row unit direction. DoRA makes **$m$ trainable** and **updates the direction with LoRA**:

$$\boxed{\;W' = m \odot \frac{V + \Delta V}{\lVert V + \Delta V \rVert_r}, \qquad \Delta V = BA\;}$$

(Axis gotcha: DoRA follows weight normalization, so the magnitude is **per-output** — HF PEFT implements `torch.linalg.norm(weight, dim=1)`; the DoRA paper writes $W\in\mathbb{R}^{\text{in}\times\text{out}}$ with a column-wise norm, equivalent to the per-row / per-output norm on a PyTorch $[\text{out},\text{in}]$ weight here.)

Intuition: LoRA couples magnitude and direction inside a single $\Delta W$; DoRA pulls magnitude out into a separate scalar sequence $m$, letting the low-rank bypass focus on learning direction. Empirically DoRA's "magnitude–direction update pattern" is closer to full fine-tuning, often slightly beating LoRA at equal parameter count, especially at low rank (small $r$). The cost: per-row normalization adds a little compute and implementation complexity; when merging you fold $m$ and the normalization back into $W'$ (still a single matrix after merge, zero inference latency).

```python
class DoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        in_f, out_f = base.in_features, base.out_features
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # per-output magnitude: weight is [out, in], norm over the in dim (dim=1) -> [out, 1] (same as HF PEFT)
        self.m = nn.Parameter(self.base.weight.norm(dim=1, keepdim=True))  # [out, 1]

    def forward(self, x):
        dW = self.scaling * (self.lora_B @ self.lora_A)       # [out, in]
        V = self.base.weight + dW
        Vnorm = V.norm(dim=1, keepdim=True) + 1e-8           # [out, 1] per-output norm
        W_eff = self.m * V / Vnorm                            # direction × magnitude (per-output)
        return F.linear(x, W_eff, self.base.bias)
```

> 💡 **Interview point: what DoRA changes** — LoRA only learns $\Delta W=BA$; DoRA = a trainable magnitude $m$ + learning direction with LoRA. It explains "why LoRA underperforms full FT on some tasks" — LoRA's magnitude/direction updates are tied together, and DoRA unties them. DoRA can stack on QLoRA (**QDoRA**) to decompose on a quantized base, balancing memory and quality.

## §7 The LoRA family at a glance

| Variant | One line | What it changes vs LoRA | Paper |
| --- | --- | --- | --- |
| **LoRA** | low-rank bypass $BA$ | baseline | Hu 2021, 2106.09685 |
| **rsLoRA** | scaling → $\alpha/\sqrt{r}$ | high-rank gradient stability | Kalajdzievski 2023, 2312.03732 |
| **DoRA** | magnitude–direction decomp | unties magnitude and direction | Liu 2024, 2402.09353 |
| **PiSSA** | init $A,B$ from $W_0$ principal components | initialization (faster convergence) | Meng 2024, 2404.02948 |
| **LoRA-GA** | make the first step approximate full FT | initialization (gradient alignment) | Wang 2024, 2407.05000 |
| **AdaLoRA** | SVD form + importance-based rank pruning | adaptive rank-budget allocation | Zhang 2023, 2303.10512 |
| **LoRA+** | different lr for $A,B$ | optimization (larger lr for $B$) | Hayou 2024, 2402.12354 |
| **(IA)³** | learn scaling vectors × activations | non-low-rank, even fewer params | Liu 2022, 2205.05638 |

A few worth expanding:

- **PiSSA**: instead of random init, do SVD on $W_0$ and initialize $A,B$ with the **principal singular values/vectors** (adapting the "most important" subspace), freezing the residual $W_0 - BA$. Aligning with principal components from the start gives faster convergence and often better quality.
- **AdaLoRA**: parameterize $\Delta W$ in SVD form $P\Lambda Q$, and during training **dynamically prune** by singular-value importance, allocating the limited rank budget to layers that need it more (different $r$ per layer) instead of uniformly.
- **(IA)³**: instead of a low-rank matrix, learn three **scaling vectors** $l_k, l_v, l_{ff}$, applied element-wise to the key, value, and FFN hidden activations: $k \to l_k \odot k$. Its parameter count is an order of magnitude smaller than LoRA; T-Few uses it for few-shot beating in-context learning.

## §8 Comparison with other PEFT paradigms

PEFT roughly splits into four families: **additive low-rank (the LoRA family)**, **additive module (Adapter)**, **soft prompt (Prompt/Prefix)**, and **selective (BitFit)**.

| Method | What it trains | Where it inserts | Extra inference overhead | Typical trainable fraction | Paper |
| --- | --- | --- | --- | --- | --- |
| **LoRA** | low-rank $BA$ | parallel to a linear layer | **zero** (mergeable) | 0.1%–1% | Hu 2021, 2106.09685 |
| **Adapter** | bottleneck MLP | serial after a sublayer | yes (serial submodule) | 0.5%–5% | Houlsby 2019, 1902.00751 |
| **Prefix-Tuning** | per-layer prefix K/V | prefix to attention K/V | yes (uses KV / sequence) | <0.1%–1% | Li-Liang 2021, 2101.00190 |
| **Prompt Tuning** | input soft prompt | embedding layer only | yes (uses sequence length) | tiny | Lester 2021, 2104.08691 |
| **P-Tuning v2** | per-layer soft prompt | prefix at every layer | yes | 0.1%–3% | Liu 2021, 2110.07602 |
| **BitFit** | biases only | bias terms at each layer | zero (in place) | ~0.08% | Ben-Zaken 2021, 2106.10199 |
| **(IA)³** | scaling vectors | K/V/FFN activations | yes (partly foldable) | <0.05% | Liu 2022, 2205.05638 |

Core comparison axes (interview favorites):

- **Inference latency**: LoRA (after merge) and BitFit are truly zero-overhead; (IA)³'s scaling vectors can often also be folded into adjacent linear-layer weights in the **fixed single-adapter** case (only dynamic multi-adapter incurs online scaling cost). Adapter's serial nonlinear submodule and Prompt/Prefix's sequence/KV occupancy are unavoidable inference costs.
- **Context occupancy**: Prompt/Prefix methods eat into the usable sequence length (soft tokens take slots); LoRA / Adapter / BitFit do not.
- **Expressiveness vs params**: Prompt Tuning is the most param-frugal but weak on small models (Lester: it matches full FT only at large scale); LoRA strikes the best param/quality trade-off and became the de facto standard.

> ⚠️ **Why Adapter has inference latency but LoRA doesn't** — Adapter **serially** inserts a new module (down→nonlinearity→up); inference must run those extra layers and it breaks original operator fusion. LoRA is a **parallel** linear bypass, mathematically mergeable into the original weight (linear + linear = linear), vanishing after merge. A nonlinear Adapter cannot be absorbed this way.

## §9 Engineering practice and common bugs

- **LoRA learning rate should be higher than full FT**: LoRA has few params and only tunes a low-rank subspace; empirically lr is $1\text{e-}4 \sim 3\text{e-}4$ (an order of magnitude higher than full FT's $1\text{e-}5\sim2\text{e-}5$). LoRA+ further argues $B$ should use a larger lr than $A$.
- **target_modules choice**: only `q,v` is frugal but weak; all linear layers (incl. FFN) is usually most stable. Wrong module names (naming differs across models, e.g. `c_attn` vs `q_proj`) make "LoRA not attached, loss not dropping."
- **$\alpha/r$ setting**: commonly $\alpha=2r$. If you change $r$ but forget to move $\alpha$, you've effectively changed the learning rate. For high rank, use rsLoRA to avoid collapse.
- **QLoRA merge pitfall** (§5.5): don't merge LoRA back into a 4-bit base; dequantize to fp16 then merge.
- **Multi-adapter serving**: not merging and adding the bypass online allows hot-swapping multiple adapters; for top inference speed, merge per task into independent weights.
- **dtype consistency**: the bypass $A,B$ and the dequantized base must add in the same precision; mixed dtype errors out or behaves oddly.
- **You save the adapter, not the full model**: `save` stores only $A,B$ (tens of MB), lightweight to distribute; load by attaching onto the same base. The base version must align, or $W_0$ mismatches and $\Delta W$ is meaningless.
- **Pair with gradient checkpointing**: LoRA doesn't save activations, so for long-sequence training always enable gradient checkpointing or activations still OOM.

> ❌ **Classic faceplant** — "LoRA brought memory down, so I can use a huge batch" — wrong. Activation memory is not saved, so a large batch can still OOM. LoRA saves the optimizer/gradient part, not the activation part.

## §10 Complexity and resources

| Dimension | Full FT | LoRA | QLoRA |
| --- | --- | --- | --- |
| Trainable params | $\Psi$ | $\sum_i r_i(d_i+k_i)$ (~0.1%–1%) | same as LoRA |
| Base storage | bf16 $2\Psi$ B | bf16 $2\Psi$ B | NF4 4 bit/param $= 0.5\Psi$ B |
| Optimizer+grad memory | $\approx 14\Psi$ B | $\approx 14\Psi_{\text{lora}}$ B | same as LoRA |
| Activation memory | high | **same as full FT** (needs checkpointing) | same as full FT |
| Training forward compute | $W_0 x$ | $W_0 x + \frac{\alpha}{r}B(Ax)$ (two extra small matmuls) | + dequant overhead |
| Inference latency (after merge) | baseline | **zero extra** | dequant or merge to fp16 |
| Max single-GPU trainable model (80GB · vanilla Adam) | ~3B (7B full FT needs ZeRO / offload / 8-bit optimizer) | ~13B–33B | ~65B |

Bypass forward overhead: each adapted layer adds $2 \cdot 2 L r d$ FLOPs ($Ax$ and $B(\cdot)$, two small matmuls); since $r \ll d$, negligible vs the main path's $2Ld^2$. After merge the bypass vanishes entirely.

> ⚠️ **"Max single-GPU model" is only a rough order of magnitude** — the real ceiling depends heavily on sequence length, batch, gradient checkpointing, optimizer choice (8-bit / offload), target_modules, and whether ZeRO sharding is used. The table is only a vanilla-Adam ballpark, not a hard limit.

## §11 25 high-frequency interview questions

Sorted into three tiers. Click to expand for answer points + pitfalls.

### L1 must-know (any role that has done fine-tuning)

<details>

<summary>Q1. What is LoRA's core formula?</summary>

- Freeze $W_0$, learn a low-rank increment $\Delta W = BA$
- Forward $h = W_0 x + \frac{\alpha}{r} BA\,x$
- $B\in\mathbb{R}^{d\times r}$, $A\in\mathbb{R}^{r\times k}$, $r\ll\min(d,k)$

Saying only "add a small matrix" without making clear $\Delta W$ is a product of two thin matrices of rank $\le r$.

</details>

<details>

<summary>Q2. Why init $B=0$ and $A$ random?</summary>

- Make the start $\Delta W = B A = 0$, not perturbing pretraining
- But $A$ nonzero ensures gradients aren't all zero, so it can learn
- Both zero → gradients identically zero → never learns

Thinking "$B=0$ means no gradient on step one so it can't learn" — wrong; on step one $B$ has a gradient ($\partial L/\partial B \propto (Ax)^\top \ne 0$), only $\Delta W$ is numerically 0 at $t=0$.

</details>

<details>

<summary>Q3. What memory does LoRA save vs full FT? What doesn't it save?</summary>

- Saves: base optimizer states (Adam's $m,v$, 8 B) + gradient (2 B) + fp32 master (4 B) ≈ **14 B/param** (full-FT total ~16 B/param incl. 2 B trainable weight); the base bf16 weight 2 B/param is still resident
- Doesn't save: **activation memory** (still backprop through the frozen base), and the base weight itself (still resident)
- Saving activations needs gradient checkpointing

Saying "LoRA saves memory across the board so I can use a big batch" — activations aren't saved, big batch still OOMs.

</details>

<details>

<summary>Q4. What is the scaling factor $\alpha/r$ for?</summary>

- Decouple rank $r$ from learning rate, so changing $r$ needs no lr re-tuning
- Keep $\Delta W$ magnitude comparable across $r$
- Commonly $\alpha=2r$ or $\alpha=r$

Not knowing $\alpha$ is a hyperparameter, or thinking $\alpha$ must equal $r$.

</details>

<details>

<summary>Q5. Why does LoRA have no extra inference latency?</summary>

- After training, merge: $W' = W_0 + \frac{\alpha}{r}BA$
- Inference uses only $W'$, structurally identical to the original linear layer
- A linear bypass can be absorbed into the original weight (linear+linear=linear)

Saying "LoRA still computes $BA$ at inference" — that's the unmerged case; after merge it's zero.

</details>

<details>

<summary>Q6. How to compute the trainable param count? Give an example.</summary>

- Single layer $d\times k$: $r(d+k)$
- Whole model: $\sum_i r_i(d_i+k_i)$
- E.g. 7B adapting $q,k,v,o$ ($4096^2$), $r=8$, 32 layers ≈ 8.4M ≈ 0.12%

Computing it as $r\cdot d\cdot k$ (that's the element count of $\Delta W$, not the LoRA param count).

</details>

<details>

<summary>Q7. Which layers of a Transformer does LoRA usually go on?</summary>

- Original paper: attention $W_q, W_v$ (beats only $W_q$ at equal budget)
- Practice (since QLoRA): all linear layers (incl. FFN gate/up/down) is more stable
- target_modules names vary by model; wrong names mean "not attached"

Only knowing "on attention," not that modern practice often adapts all linear layers.

</details>

<details>

<summary>Q8. How to choose $r$? Is bigger always better?</summary>

- Simple tasks $r=4\sim8$, hard tasks $r=16\sim64$
- Not bigger-is-better: too large overfits, saves little, may hit $\alpha/r$ high-rank collapse
- For high-rank payoff use rsLoRA ($\alpha/\sqrt r$)

Saying "bigger $r$ is always better," ignoring overfitting and scaling collapse.

</details>

<details>

<summary>Q9. Should LoRA's learning rate be larger or smaller than full FT?</summary>

- Usually **an order of magnitude larger** ($1\text{e-}4\sim3\text{e-}4$ vs full FT $1\text{e-}5\sim2\text{e-}5$)
- Because it tunes only a low-rank subspace with few params
- LoRA+ further argues $B$ should use a larger lr than $A$

Applying full FT's small lr directly, making LoRA learn too slowly.

</details>

<details>

<summary>Q10. What do you save from a LoRA run?</summary>

- Only the adapter ($A,B$), tens of MB
- Load by attaching to the same base
- The base version must align, or $W_0$ mismatches and $\Delta W$ is void

Thinking you save the whole fine-tuned large model.

</details>

### L2 advanced (research / deep-engineering roles)

<details>

<summary>Q11. Why believe "weight updates are low-rank"? Theoretical basis?</summary>

- Aghajanyan 2020 (arXiv 2012.13255): pretrained models have a very low intrinsic dimension; low-dim subspace optimization approximates full FT
- Hu 2021 followed this: $\Delta W$ low-rank, parameterized as $BA$
- It's an **empirical hypothesis + evidence**, not a strict theorem

Treating "low rank works" as a proven theorem, or unable to state the source of intrinsic dimension.

</details>

<details>

<summary>Q12. Derive the first-step gradients for $A$, $B$ and explain the init.</summary>

- Let $g=\partial L/\partial h$. $\partial L/\partial B = \frac{\alpha}{r} g (Ax)^\top$, $\partial L/\partial A = \frac{\alpha}{r} B^\top g\, x^\top$
- $t=0$: $B=0$ → $\partial L/\partial A=0$ ($A$ frozen); $A\ne0$ → $\partial L/\partial B\ne0$ ($B$ moves)
- Once $B$ becomes nonzero, $A$ gets a gradient next step

Computing both gradients as nonzero, or saying "$B=0$ so both are zero."

</details>

<details>

<summary>Q13. What does rsLoRA fix? Why scaling $\alpha/\sqrt r$?</summary>

- Problem: $\alpha/r$ over-shrinks gradients at high rank, learning collapses
- $BA$'s output is a sum of $r$ terms, std $\propto\sqrt r$; dividing by $\sqrt r$ pulls magnitude back to a constant
- Dividing by $r$ over-suppresses, harder the larger $r$ is

Just memorizing "rsLoRA uses $\sqrt r$" without the variance/magnitude-stability reason.

</details>

<details>

<summary>Q14. What is NF4? Why better than INT4 for weight quantization?</summary>

- NF4 = 4-bit NormalFloat, 16 levels placed at standard-normal quantiles so bins carry equal probability mass
- Weights are ~$\mathcal{N}(0,\sigma^2)$; **under the normal assumption** equal-mass bins are info-theoretically optimal for that fixed distribution (quantile quantization)
- INT4 is equally spaced and wastes bins on normal tails; NF4 is also zero-preserving

Saying NF4 is plain 4-bit integer, or missing the key "equal probability mass / quantile."

</details>

<details>

<summary>Q15. How much does QLoRA's double quantization save? How?</summary>

- Re-quantize each block(64)'s fp32 absmax constant itself
- Single overhead $32/64=0.5$ bit/param → double $\approx0.127$ bit/param
- Saves ~0.37 bit/param on average (~3GB for 65B)

Not knowing "what's re-quantized is the scaling constant absmax," or confusing it with NF4 itself.

</details>

<details>

<summary>Q16. QLoRA's base is 4-bit during training — how do gradients flow?</summary>

- Base stored as NF4, dequantized to bf16 per block for the matmul
- Gradients flow only to the bf16 LoRA params; the NF4 base is frozen with no gradient
- So it saves the base's optimizer/gradient; activations remain

Thinking gradients are computed directly on 4-bit, or that the base is also updating.

</details>

<details>

<summary>Q17. What does DoRA change vs LoRA? Why might it be better?</summary>

- Decompose $W_0$ into magnitude $m=\lVert W_0\rVert_r$ (**per output row**, over the in dim `dim=1`, same as weight norm / HF PEFT) and direction $V/\lVert V\rVert_r$
- $m$ trainable, LoRA only updates direction: $W'=m\odot\frac{V+\Delta V}{\lVert V+\Delta V\rVert_r}$
- Unties magnitude and direction; update pattern closer to full FT, gains notable at low rank

Saying only "DoRA adds a magnitude," not the **per-output** normalization + direction-via-LoRA; or getting the magnitude axis wrong (per-input instead of per-output — DoRA is per-output).

</details>

<details>

<summary>Q18. Why does Adapter have inference latency but merged LoRA doesn't?</summary>

- Adapter serially inserts a nonlinear bottleneck module; inference must run it and it breaks operator fusion
- LoRA is a parallel linear bypass; linear+linear=linear, absorbable into the original weight
- A nonlinear module can't be merged this way

Saying both "add a small module so both have latency," missing the linear-mergeability point.

</details>

<details>

<summary>Q19. Essential difference between Prompt/Prefix Tuning and LoRA?</summary>

- Prompt/Prefix: inject trainable soft tokens / prefix K/V at the input or each layer, **occupying sequence length or KV**, with inference overhead
- LoRA: a low-rank bypass on the weights, **no sequence occupancy**, zero overhead after merge
- Prompt Tuning is weak on small models (Lester: matches full FT only at large scale)

Lumping LoRA in as "adding tokens," or not knowing Prompt methods occupy sequence length.

</details>

<details>

<summary>Q20. What do "init-changing" methods like PiSSA / LoRA-GA change?</summary>

- PiSSA: init $A,B$ with $W_0$'s principal singular vectors (aligning principal components), freeze the residual, faster convergence
- LoRA-GA: make LoRA's first-step gradient direction approximate full FT's gradient (gradient-alignment init)
- Common point: no structural change, only **initialization** to speed convergence / improve quality

Treating them as new structures, or unable to state the shared "changes initialization."

</details>

### L3 top-lab questions (deep end)

<details>

<summary>Q21. Why doesn't LoRA save activation memory automatically? How to save it?</summary>

- The bypass hangs on a frozen layer; backprop still flows through the whole network for $\partial L/\partial x$, storing intermediate activations
- LoRA saves optimizer states + gradients + master weights, unrelated to activations
- To save activations use gradient checkpointing (recompute for memory), or sequence parallelism etc.

Vaguely saying "LoRA saves memory," then unable to answer "which part" when pressed.

</details>

<details>

<summary>Q22. How does AdaLoRA implement "adaptive rank"?</summary>

- Parameterize $\Delta W$ in SVD form $P\Lambda Q$
- During training prune $\Lambda$ by singular-value importance, assigning different effective ranks per layer
- Dynamically route the limited rank budget to layers that need it, not uniformly

Thinking AdaLoRA "auto-selects a global $r$," missing the SVD parameterization + per-layer pruning.

</details>

<details>

<summary>Q23. For multi-task deployment, what are the two strategies for one base + many LoRAs? Their costs?</summary>

- Merge approach: merge an independent full weight per task, zero inference latency, but one full weight copy per task and no dynamic switching
- Bypass approach: keep the base, add each task's $\frac{\alpha}{r}B(Ax)$ online, hot-swappable / mix different adapters within a batch, at the cost of restoring a bit of inference overhead
- Industry mostly uses the bypass approach + batching optimizations; dedicated **multi-LoRA serving systems** (S-LoRA, Punica) serve hundreds–thousands of adapters concurrently on one base at high throughput (unified memory paging + custom batched kernels)

Only knowing merge, not "not merging to support dynamic multi-adapter."

</details>

<details>

<summary>Q24. (IA)³ and LoRA — essential difference? Why is (IA)³ more param-frugal?</summary>

- LoRA: adds a low-rank matrix $BA$, params $r(d+k)$
- (IA)³: learns three **scaling vectors** $l_k,l_v,l_{ff}$ applied element-wise to K/V/FFN activations, params only $O(d)$
- (IA)³ is "element-wise rescaling of activations" rather than "adding a matrix increment," so an order of magnitude fewer; T-Few uses it to beat ICL in few-shot

Treating (IA)³ as a low-rank matrix too, missing "scaling vector × activations."

</details>

<details>

<summary>Q25. Can LoRA fine-tuning learn "new knowledge"? Is low rank an expressiveness bottleneck?</summary>

- The low-rank bypass excels at "adjusting existing capabilities": style / format / task adaptation / alignment
- Injecting large amounts of **brand-new factual knowledge** may be limited by low rank — needs larger $r$, more layers, or continued pretraining / full FT
- Biderman et al. (2024, *LoRA Learns Less and Forgets Less*, arXiv 2405.09673) show empirically that on hard tasks (code / math) LoRA often learns less than full FT but **also forgets less** — the low-rank constraint is a double-edged sword
- This is exactly the motivation behind DoRA / high rank / PiSSA to break the expressiveness limit; rank is a capacity knob, not a free lunch

Absolutely saying "LoRA equals full FT" or "LoRA can't learn anything new" — both overreach; it depends on $r$, the number of adapted layers, and task type.

</details>

## §A Appendix: sanity check

Key invariants of `LoRALinear` (verifiable with a short script):

- **Start equivalence**: after injecting LoRA and before training, the output should equal the base elementwise (since $B=0 \Rightarrow \Delta W=0$).
- **Merge consistency**: outputs before/after `merge()` for the same input should match numerically (under fp32 / eval / dropout=0 the error is ~1e-6 from pure float accumulation; under bf16 or train-mode dropout the error is larger and this value shouldn't be promised).
- **Trainable set**: only `lora_A`, `lora_B` (and DoRA's `m`) have `requires_grad=True`, the base all `False`.
- **Parameter count**: adapting $q,v$, $r=8$, $d=4096$, $L$ layers gives trainable params $= L \times 2 \times 8 \times (4096+4096)$.

Below is the **real run** output of [`code/lora.py`](code/lora.py) ($\text{IN}=32, \text{OUT}=48, r=8, \alpha=16$) on **PyTorch 2.10 / CPU** (each line has an `assert`; the summary prints only if all pass):

```
[a] B=0 init: |out_lora - out_base| = 0.00e+00  OK
[b] after 1 step: ||dW|| = 4.768e-02, output changed  OK
[c] merge: |out_merged - out_unmerged| = 3.58e-07; weight += dW = True; unmerge restores base = True  OK
[d] trainable params = 640 (expect r*(in+out) = 640); base frozen = True  OK
[e] scaling: standard alpha/r = 2.0000, rsLoRA alpha/sqrt(r) = 5.6569  OK
[f] DoRA: m.requires_grad = True, base frozen, identity start |Δ| = 1.19e-07  OK

all LoRA / DoRA sanity checks passed ✓
```

Here $640 = r(\text{IN}+\text{OUT}) = 8\times(32+48)$ verifies the trainable-param formula of §4; the pure-float error of $3.58\text{e-}7$ across merge verifies the numerical equivalence of zero-latency merging; the DoRA identity start $\lvert\Delta\rvert=1.19\text{e-}7$ verifies $W'=W_0$ when $B=0$.

---

## 📜 Runnable Code

The LoRA / DoRA core implementation of this tutorial has a minimal runnable version at [`docs/tutorials/code/lora.py`](code/lora.py):

- [`lora.py`](code/lora.py) — `LoRALinear` ($B=0$ identity start · $\alpha/r$ and rsLoRA $\alpha/\sqrt{r}$ scaling · merge/unmerge) + `DoRALinear` (magnitude–direction decomposition), with 6 `assert` sanity checks (start equivalence / $\Delta W \ne 0$ after one step / merge consistent & reversible / param count $= r(\text{in}+\text{out})$ / rsLoRA scaling / DoRA frozen base).

Pure PyTorch, runs on CPU in seconds with no GPU: `python docs/tutorials/code/lora.py`. The §A output above is this script's real run result.

---

## 📚 References

- **LoRA** — Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, arXiv 2106.09685 (2021), ICLR 2022.
- **Intrinsic Dimension** — Aghajanyan et al., *Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning*, arXiv 2012.13255 (2020).
- **QLoRA** — Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs*, arXiv 2305.14314 (2023), NeurIPS 2023.
- **DoRA** — Liu et al., *DoRA: Weight-Decomposed Low-Rank Adaptation*, arXiv 2402.09353 (2024), ICML 2024.
- **rsLoRA** — Kalajdzievski, *A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA*, arXiv 2312.03732 (2023).
- **PiSSA** — Meng et al., *PiSSA: Principal Singular Values and Singular Vectors Adaptation*, arXiv 2404.02948 (2024), NeurIPS 2024.
- **LoRA-GA** — Wang et al., *LoRA-GA: Low-Rank Adaptation with Gradient Approximation*, arXiv 2407.05000 (2024).
- **AdaLoRA** — Zhang et al., *Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning*, arXiv 2303.10512 (2023), ICLR 2023.
- **LoRA+** — Hayou et al., *LoRA+: Efficient Low Rank Adaptation of Large Models*, arXiv 2402.12354 (2024).
- **(IA)³ / T-Few** — Liu et al., *Few-Shot Parameter-Efficient Fine-Tuning Is Better and Cheaper than In-Context Learning*, arXiv 2205.05638 (2022).
- **Adapter** — Houlsby et al., *Parameter-Efficient Transfer Learning for NLP*, arXiv 1902.00751 (2019), ICML 2019.
- **Prefix-Tuning** — Li & Liang, *Prefix-Tuning: Optimizing Continuous Prompts for Generation*, arXiv 2101.00190 (2021), ACL 2021.
- **Prompt Tuning** — Lester et al., *The Power of Scale for Parameter-Efficient Prompt Tuning*, arXiv 2104.08691 (2021), EMNLP 2021.
- **P-Tuning v2** — Liu et al., *P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks*, arXiv 2110.07602 (2021).
- **BitFit** — Ben-Zaken et al., *BitFit: Simple Parameter-efficient Fine-tuning for Transformer-based Masked Language-models*, arXiv 2106.10199 (2021), ACL 2022.
- **S-LoRA** — Sheng et al., *S-LoRA: Serving Thousands of Concurrent LoRA Adapters*, arXiv 2311.03285 (2023).
- **LoRA Learns Less and Forgets Less** — Biderman et al., arXiv 2405.09673 (2024), TMLR 2024.
