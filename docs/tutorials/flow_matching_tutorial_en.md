## §0 TL;DR

> 💡 **Flow Matching in 5 sentences** — one page covering the core points (full derivations in §1–§4).

1. **Goal**: learn a vector field $v_\theta(t, x)$ such that the ODE $\dot{x}_t = v_\theta(t, x_t)$ transports $x_0 \sim p_0$ (noise) into $x_1 \sim p_1$ (data).

2. **Training (CFM)**: $\mathcal{L}_\text{CFM}(\theta) = \mathbb{E}_{t, z, x_t \sim p_t(\cdot|z)} \|v_\theta(t, x_t) - u_t(x_t|z)\|^2$, **simulation-free** (no ODE solve needed to compute the loss).

3. **Key theorem**: $\nabla_\theta \mathcal{L}_\text{FM} = \nabla_\theta \mathcal{L}_\text{CFM}$ — so learning the conditional vector field is equivalent to learning the marginal one (Lipman et al. 2023).

4. **Simplest form (Rectified Flow / OT-CFM)**: $x_t = (1-t)x_0 + tx_1$, target $u_t = x_1 - x_0$. SD3 / FLUX / Lumina all use this.

5. **Sampling**: starting from $x_0 \sim p_0$, integrate with an ODE solver (Euler / Heun / RK4) until $t=1$.

## §1 Basic setup and intuition

Given a data distribution $p_1$ (the "target") and a simple prior $p_0$ (typically $\mathcal{N}(0, I)$), we want to construct a family of **probability paths** $\{p_t\}_{t \in [0,1]}$ smoothly interpolating from $p_0$ to $p_1$.

> ⚠️ **Convention (used throughout)** — notation summarized in the table below.

- $x_0 \sim p_0 = \mathcal{N}(0, I)$ (noise side) — $t=0$

- $x_1 \sim p_1$ (data side) — $t=1$

- Sampling direction: integrate from $t=0$ to $t=1$ (noise → data)

- Note: different papers use different conventions — Lipman et al. 2023 uses $x_0$=data, $x_1$=noise; Liu et al. 2022 (Rectified Flow) uses $x_0$=noise, $x_1$=data (which we follow here). The SD3 paper is also noise→data but with slightly different notation. **In interviews, disambiguate in your first sentence**.

A family of **time-varying vector fields** $u_t : [0,1] \times \mathbb{R}^d \to \mathbb{R}^d$ pushes particles from $p_0$ to $p_1$ via the ODE $\dot{x}_t = u_t(x_t)$. By the **continuity equation**:

$$\boxed{\;\frac{\partial p_t}{\partial t} + \nabla \cdot (p_t\, u_t) = 0\;}$$

Our goal: find a neural network $v_\theta(t, x) \approx u_t(x)$.

```

   p_0 (noise)             p_t (intermediate)       p_1 (data)
   ●●●●●          →        ●  ●  ●        →        ████
                            v_θ(t, x)
                           ─────────→
                          dx/dt = v_θ
```

Compared to diffusion:

- **Diffusion (SDE)**: $dx = f(x, t) dt + g(t) dW$, trained with score matching $s_\theta \approx \nabla \log p_t$

- **Flow matching (ODE)**: $dx = v_\theta(t, x) dt$, **no stochastic term**, training directly regresses the vector field

- The two are linked via the **probability flow ODE**: $v = f - \frac{1}{2} g^2 \nabla \log p_t$ (see §6)

## §2 Flow Matching Loss

### 2.1　Marginal Flow Matching (theoretical form)

If we knew $u_t$ (the marginal vector field), we could just regress against it:

$$\mathcal{L}_\text{FM}(\theta) = \mathbb{E}_{t \sim \mathcal{U}[0,1],\; x \sim p_t} \left\| v_\theta(t, x) - u_t(x) \right\|^2$$

**Problem**: $u_t(x)$ is a marginal obtained by integrating (weighted) all conditional paths — **not directly sampleable**.

### 2.2　Conditional Flow Matching (the practical training objective)

Introduce a **conditioning variable** $z$ (e.g. $z = x_1$, or $z = (x_0, x_1)$). Pick a conditional path $p_t(x | z)$ and conditional vector field $u_t(x | z)$ such that marginalizing over $z$ recovers the desired marginal:

$$p_t(x) = \int p_t(x | z) q(z)\, dz, \quad u_t(x) = \int u_t(x|z) \frac{p_t(x|z) q(z)}{p_t(x)} dz$$

Then the **Conditional FM loss** is:

$$\boxed{\;\mathcal{L}_\text{CFM}(\theta) = \mathbb{E}_{t,\; z \sim q,\; x \sim p_t(\cdot|z)} \left\| v_\theta(t, x) - u_t(x|z) \right\|^2\;}$$

Each term is **sampleable and computable**. $x \sim p_t(\cdot|z)$ is usually closed-form sampleable (e.g. linear interpolation below).

### 2.3　Key theorem (Lipman et al. 2023, Theorem 2)

> ✅ **Gradient equivalence theorem** — under appropriate regularity of $p_t$ and $u_t$, and $p_t > 0$:
$$\nabla_\theta \mathcal{L}_\text{FM}(\theta) = \nabla_\theta \mathcal{L}_\text{CFM}(\theta)$$
So **minimizing CFM ≡ minimizing FM**. The two losses differ by a $\theta$-independent constant under the above assumptions.

**Proof sketch**: expand the L2 norm $\|v_\theta\|^2 - 2 v_\theta^\top u_t + \|u_t\|^2$; the first two terms are equal under either loss (using the definition of $u_t$ to write the marginal as a conditional-weighted expectation); the third term is $\theta$-independent and vanishes under the gradient.

> 💡 **Interview bonus: marginal vector field is non-unique** — given $p_t$, the $u_t$ satisfying the continuity equation $\partial_t p_t + \nabla\cdot(p_t u_t) = 0$ is **not unique** — adding any divergence-free vector field still yields a valid choice. CFM automatically picks a "natural" $u_t$ via the conditional path (usually corresponding to the OT map or a score-based ODE). This is often a follow-up: "Is the marginal $u_t$ unique?"

## §3 Three conditional path choices

Let $z = (x_0, x_1)$, $x_0 \sim p_0$, $x_1 \sim p_1$. The conditional path $p_t(x | x_0, x_1)$ is generally a Dirac $\delta(x - \psi_t(x_0, x_1))$ (deterministic interpolation), with conditional vector field $\dot{\psi}_t(x_0, x_1)$.

| Path | $x_t = \psi_t(x_0, x_1)$ | Target $u_t$ | Used in |
| --- | --- | --- | --- |
| **Rectified Flow / OT-CFM** | $(1-t)x_0 + t\, x_1$ | $x_1 - x_0$ (constant) | SD3, FLUX, Lumina, MovieGen |
| **VP cosine** | $\cos\!\left(\frac{\pi t}{2}\right) x_0 + \sin\!\left(\frac{\pi t}{2}\right) x_1$ | $-\frac{\pi}{2}\sin\!\frac{\pi t}{2}\, x_0 + \frac{\pi}{2}\cos\!\frac{\pi t}{2}\, x_1$ | Same family as DDPM cosine schedule (under restrictions) |
| **VE** | $x_1 + \sigma(1{-}t)\, x_0$, $\sigma$ increasing | $-\sigma'(1{-}t)\, x_0$ | Same family as SMLD/EDM (prior variance must match $\sigma_{\max}^2$) |

### 3.1　Rectified Flow: simplest, most stable, most widely used

Linear interpolation: $x_t = (1-t) x_0 + t\, x_1$, so $\dot{x}_t = x_1 - x_0$ is **constant** (does not depend on $t$).

Training objective:

$$\mathcal{L}_\text{RF}(\theta) = \mathbb{E}_{t, x_0, x_1} \|v_\theta(t,\, (1-t)x_0 + t x_1) - (x_1 - x_0)\|^2$$

The name "OT-CFM" comes from: if $(x_0, x_1)$ is the optimal transport coupling (rather than independent samples), the learned vector field approximately realizes the OT map.

> ✅ **Reflow: Rectified Flow's killer feature** — use the learned $v_\theta$ to regenerate $(x_0, x_1)$ pairs (run the ODE from $x_0$ to obtain the corresponding $x_1$), then **train again**. The new trajectories are straighter, and **few-step sampling quality improves dramatically**, enabling 1-step / 2-step generation (InstaFlow et al.).

### 3.2　VP path (same family as DDPM)

With $\sigma(t) = \cos\!\frac{\pi t}{2}$ (noise coefficient) and $\alpha(t) = \sin\!\frac{\pi t}{2}$ (data coefficient), satisfying $\sigma^2 + \alpha^2 = 1$ (variance preserving):

$$x_t = \sigma(t)\, x_0 + \alpha(t)\, x_1, \quad u_t = \sigma'(t)\, x_0 + \alpha'(t)\, x_1$$

Boundaries: $x_t = x_0$ (noise) at $t=0$, $x_t = x_1$ (data) at $t=1$.

This path and DDPM's cosine schedule belong to **the same Gaussian-path family** (continuous limit + time reversal). But strictly speaking they are not "exactly equivalent" — DDPM (Nichol-Dhariwal) has details like $s=0.008$ offset, and DDPM uses the forward-noising convention ($t=0$ is data) while FM uses the reverse ($t=0$ is noise).

### 3.3　VE path (same family as SMLD/EDM)

Following Lipman et al. 2023's conditional VE path:

$$p_t(x | x_1) = \mathcal{N}\!\left(x \,\Big|\, x_1,\; \sigma(1-t)^2 I\right)$$

$\sigma(s)$ is monotonically increasing in forward time $s \in [0, 1]$ (e.g. $\sigma(s) = \sigma_\min (\sigma_\max/\sigma_\min)^s$). Reparameterizing gives

$$x_t = x_1 + \sigma(1-t)\, x_0, \quad u_t = -\sigma'(1-t)\, x_0$$

Boundaries: $x_t \approx x_1 + \sigma_\max\, x_0$ at $t=0$ (noise-dominated), $x_t \approx x_1 + \sigma_\min\, x_0 \approx x_1$ at $t=1$ (data).

> ⚠️ **VE deployment note** — strictly, the prior $p_0$ should be $\mathcal{N}(0, \sigma_\max^2 I)$ (so the marginal variance at $t=0$ matches); when using $\mathcal{N}(0, I)$, scale accordingly (e.g. $x_0 \leftarrow \sigma_\max \cdot \tilde{x}_0$). The code examples here are pedagogical; **for production VE, use EDM preconditioning** for stability.

## §4 Training code framework (PyTorch)

### 4.1　Probability Path abstraction

```python
import math
from dataclasses import dataclass
from typing import Callable, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class FlowPath:
    """ Conditional probability path abstraction """
    name: str
    sample_xt: Callable    # (t, x0, x1) -> x_t
    target_ut: Callable    # (t, x0, x1) -> u_t

def _broadcast_t(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """ t: [B], x: [B, ...] —— broadcast t to shape [B, 1, 1, ...] for elementwise ops """
    return t.view(-1, *([1] * (x.dim() - 1)))

def rectified_flow_path() -> FlowPath:
    """ x_t = (1-t)x_0 + t*x_1,  u_t = x_1 - x_0 """
    def sample_xt(t, x0, x1):
        tb = _broadcast_t(t, x0)
        return (1 - tb) * x0 + tb * x1
    def target_ut(t, x0, x1):
        return x1 - x0
    return FlowPath("rectified_flow", sample_xt, target_ut)

def vp_cosine_path() -> FlowPath:
    """ x_t = cos(π t/2) x_0 + sin(π t/2) x_1
        t=0: x_t = x_0 (noise);  t=1: x_t = x_1 (data)  [noise → data direction] """
    def sample_xt(t, x0, x1):
        tb = _broadcast_t(t, x0)
        sig = torch.cos(0.5 * math.pi * tb)    # noise coeff
        alp = torch.sin(0.5 * math.pi * tb)    # data coeff
        return sig * x0 + alp * x1
    def target_ut(t, x0, x1):
        tb = _broadcast_t(t, x0)
        d_sig = -0.5 * math.pi * torch.sin(0.5 * math.pi * tb)
        d_alp =  0.5 * math.pi * torch.cos(0.5 * math.pi * tb)
        return d_sig * x0 + d_alp * x1
    return FlowPath("vp_cosine", sample_xt, target_ut)

def ve_path(sigma_min: float = 0.01, sigma_max: float = 50.0) -> FlowPath:
    """ VE: x_t = x_1 + σ(1-t) · x_0,  σ(s) increasing in forward time s (log-linear)
        t=0: x_t = x_1 + σ_max·x_0 (large noise);  t=1: x_t ≈ x_1 (data)
        Note: strict VE requires prior p_0 ~ N(0, σ_max² I); this example uses N(0, I) for
              simplicity. Production code needs EDM-style preconditioning. """
    log_min, log_max = math.log(sigma_min), math.log(sigma_max)
    def sigma_fwd(s):                          # increasing in forward time s
        return torch.exp(log_min * (1 - s) + log_max * s)
    def d_sigma_fwd(s):                        # dσ/ds = σ · (log σ_max − log σ_min)
        return sigma_fwd(s) * (log_max - log_min)
    def sample_xt(t, x0, x1):
        tb = _broadcast_t(t, x0)
        return x1 + sigma_fwd(1 - tb) * x0
    def target_ut(t, x0, x1):
        tb = _broadcast_t(t, x0)
        # u_t = d/dt [σ(1-t)] x_0 = -σ'(1-t) · x_0
        return -d_sigma_fwd(1 - tb) * x0
    return FlowPath("ve", sample_xt, target_ut)
```

### 4.2　Vector field network (pedagogical MLP; production uses U-Net / DiT)

```python
class SinusoidalTimeEmbed(nn.Module):
    """ Time encoding isomorphic to Transformer positional embedding """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B] in [0, 1]
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

class VectorFieldMLP(nn.Module):
    """ v_θ(t, x) ——— simplified version for 2D toy / low-dim experiments
        Real generative models replace this with U-Net (image) or DiT (high-res / video) """
    def __init__(self, dim: int, hidden: int = 256, t_dim: int = 128):
        super().__init__()
        self.t_embed = nn.Sequential(
            SinusoidalTimeEmbed(t_dim),
            nn.Linear(t_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.net = nn.Sequential(
            nn.Linear(dim + hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden),       nn.SiLU(),
            nn.Linear(hidden, dim),
        )
    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # t: [B], x: [B, dim]
        return self.net(torch.cat([x, self.t_embed(t)], dim=-1))
```

### 4.3　CFM Loss

```python
def cfm_loss(
    model: nn.Module,
    path: FlowPath,
    x1: torch.Tensor,                              # [B, ...] data samples
    x0: Optional[torch.Tensor] = None,              # defaults to N(0, I)
    t_dist: str = "uniform",                        # "uniform" or "logitnormal"
    return_components: bool = False,
):
    """
    Conditional Flow Matching loss:
        L = E ‖v_θ(t, x_t) - u_t(x_t | x_0, x_1)‖²
    """
    B = x1.shape[0]
    device = x1.device
    if x0 is None:
        x0 = torch.randn_like(x1)

    # t sampling
    if t_dist == "uniform":
        t = torch.rand(B, device=device)
    elif t_dist == "logitnormal":
        # SD3 default: t = σ(z), z ~ N(0, 1). More concentrated around t≈0.5 (hardest middle region)
        t = torch.sigmoid(torch.randn(B, device=device))
    else:
        raise ValueError(f"unknown t_dist: {t_dist}")

    x_t = path.sample_xt(t, x0, x1)
    u_t = path.target_ut(t, x0, x1)
    v_pred = model(t, x_t)

    loss = F.mse_loss(v_pred, u_t)
    if return_components:
        return loss, {"v_pred_norm": v_pred.norm().item(), "u_norm": u_t.norm().item()}
    return loss
```

### 4.4　Minimal training loop

```python
def train_flow_matching(
    model: nn.Module,
    dataloader,                         # yields x1 batches
    path: FlowPath,
    total_steps: int = 50_000,
    lr: float = 3e-4,
    weight_decay: float = 0.0,
    device: str = "cuda",
    log_every: int = 200,
    ema_decay: float = 0.9999,           # EMA is essential for generative models
):
    model = model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    ema_model = _make_ema(model)        # see below

    step = 0
    while step < total_steps:
        for x1 in dataloader:
            x1 = x1.to(device, non_blocking=True)
            loss = cfm_loss(model, path, x1, t_dist="logitnormal")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            _update_ema(ema_model, model, ema_decay)

            if step % log_every == 0:
                print(f"[{step:6d}] {path.name} loss = {loss.item():.4f}")
            step += 1
            if step >= total_steps: break

    return model, ema_model

@torch.no_grad()
def _make_ema(model):
    import copy
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters(): p.requires_grad_(False)
    return ema

@torch.no_grad()
def _update_ema(ema, model, decay):
    for ep, p in zip(ema.parameters(), model.parameters()):
        ep.mul_(decay).add_(p.detach(), alpha=1 - decay)
```

## §5 ODE sampling

After training $v_\theta$, start from $x_0 \sim p_0$ and solve the ODE $\dot{x}_t = v_\theta(t, x_t)$ up to $t = 1$.

```python
@torch.no_grad()
def euler_sampler(model, x0, steps=50, t_start=0.0, t_end=1.0):
    """ First-order Euler: 1 NFE per step, simple but needs many steps """
    x = x0.clone()
    ts = torch.linspace(t_start, t_end, steps + 1, device=x0.device)
    for i in range(steps):
        t = ts[i].expand(x.shape[0])
        dt = ts[i + 1] - ts[i]
        x = x + dt * model(t, x)
    return x

@torch.no_grad()
def heun_sampler(model, x0, steps=50, t_start=0.0, t_end=1.0):
    """ Second-order Heun (improved Euler / RK2): 2 NFE per step, O(dt²) accuracy """
    x = x0.clone()
    ts = torch.linspace(t_start, t_end, steps + 1, device=x0.device)
    for i in range(steps):
        b = x.shape[0]
        t_i, t_next = ts[i], ts[i + 1]
        dt = t_next - t_i
        v1 = model(t_i.expand(b), x)
        x_euler = x + dt * v1
        v2 = model(t_next.expand(b), x_euler)
        x = x + dt * 0.5 * (v1 + v2)
    return x

@torch.no_grad()
def rk4_sampler(model, x0, steps=25, t_start=0.0, t_end=1.0):
    """ Fourth-order Runge-Kutta: 4 NFE per step, O(dt⁴) accuracy
        25 steps × 4 NFE = 100 NFE, but usually much more accurate than 100-step Euler """
    x = x0.clone()
    ts = torch.linspace(t_start, t_end, steps + 1, device=x0.device)
    for i in range(steps):
        b = x.shape[0]
        t_i, t_next = ts[i], ts[i + 1]
        dt = t_next - t_i
        k1 = model(t_i.expand(b),                       x)
        k2 = model((t_i + dt / 2).expand(b),  x + dt / 2 * k1)
        k3 = model((t_i + dt / 2).expand(b),  x + dt / 2 * k2)
        k4 = model(t_next.expand(b),          x + dt * k3)
        x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x
```

> 💡 **Sampler choice cheat sheet** — sorted by NFE / quality trade-off.

- **Euler**: 1 NFE/step, needs ≥50 steps for good images; debug baseline

- **Heun / RK2**: 2 NFE/step, ~25 steps already good; EDM default

- **RK4**: 4 NFE/step, 10-20 steps usually matches 100-step Euler

- **Adaptive (Dopri5 / dopri8)**: provided by torchdiffeq; auto error control but uncontrolled NFE

- **Rectified Flow after retraining**: after 1-2 reflow passes, 1-4 step Euler reaches near multi-step quality

## §6 Relationship to diffusion / score matching

For any SDE $dx = f(x, t) dt + g(t) dW$ (forward), there exists a corresponding **probability flow ODE** (Song et al. 2021):

$$dx = \underbrace{\left[ f(x, t) - \frac{1}{2} g^2(t)\, \nabla_x \log p_t(x) \right]}_{\text{vector field } u_t(x)} dt$$

This ODE has the same marginal distribution $p_t$ as the SDE at every time.

> ✅ **FM ↔ Score Matching bridge (with caveats)** — when the FM probability path arises from a non-degenerate noising SDE ($g(t) > 0$), learning the score $s_\theta \approx \nabla \log p_t$ and learning the vector field $v_\theta \approx u_t$ are **two parameterizations of the same information**:
$$v_\theta(t, x) = f(x, t) - \tfrac{1}{2} g^2(t)\, s_\theta(t, x)$$
So under VP/VE paths, FM can be viewed as a score matching equivalent in the ODE viewpoint. **But this fails for Rectified Flow / OT-CFM** (no standard SDE correspondence), where FM is more general vector-field regression.

### 6.1　Velocity ↔ Score ↔ Noise prediction interconversion (must know)

Under VP/VE paths, assuming $x_t = \alpha(t) x_1 + \sigma(t) x_0$ (with $x_0 \sim \mathcal{N}(0, I)$ as the noise direction), the three main prediction targets are linearly interconvertible:

$$
\begin{aligned}
\epsilon\text{-prediction} &:\quad \epsilon_\theta(t, x_t) \approx x_0 \\
x_0\text{-prediction} &:\quad x^0_\theta(t, x_t) \approx x_1 \\
v\text{-prediction (Salimans-Ho)} &:\quad v_\theta(t, x_t) \approx \alpha'(t) x_1 + \sigma'(t) x_0 \\
\text{score} &:\quad s_\theta(t, x_t) \approx -x_0 / \sigma(t)
\end{aligned}
$$

Given $x_t$ and any one prediction, the other three are algebraically recoverable. For example, under VP the $\epsilon$-score relation is:

$$s_\theta(t, x_t) = -\epsilon_\theta(t, x_t) / \sigma(t)$$

This is why DDPM (learning $\epsilon$) and score-based (learning $\nabla \log p_t$) are **equivalent parameterizations**. Flow matching learning $v = \alpha' x_1 + \sigma' x_0$ is one such choice, and under RF (linear) it degenerates to $v = x_1 - x_0$.

### 6.2　Correspondence between FM paths and diffusion

| FM Path | Equivalent diffusion / SDE | Typical noise schedule |
| --- | --- | --- |
| VP cosine | DDPM (cosine) | $\bar\alpha_t = \cos^2(\pi t/2)$ |
| VP linear | DDPM (linear β) | $\beta_t = \beta_0 + t(\beta_1 - \beta_0)$ |
| VE | SMLD / EDM | $\sigma_t \in [\sigma_\min, \sigma_\max]$ log-linear |
| Rectified Flow | No standard non-zero-diffusion noising SDE (except degenerate cases) | Path is a straight line, the "shortest" path |

### 6.3　Why Rectified Flow training / sampling is relatively "stable"

- **Constant target**: $u_t = x_1 - x_0$ does not explicitly depend on $t$ (given $x_0, x_1$), making it numerically easy to fit

- **Straight-line paths**: few-step ODE integration error is small

- **Loss conditioning**: RF training is more balanced than native DDPM; but **that does not mean reweighting is unnecessary** — SD3 still applies logit-normal $t$ sampling and similar reweighting on top of RF, with ablated gains

- **Reflow compresses NFE**: enables 1-step generation routes (InstaFlow / SD3-Turbo / Flux-Schnell)

## §7 Advanced topics

### 7.1　Reflow (Liu et al., ICLR 2023)

The reason Rectified Flow enables few-step generation is the **reflow algorithm**:

1. Train initially to obtain $v_\theta^{(1)}$ (using independent pairs $(x_0, x_1) \sim p_0 \otimes p_1$)

2. Use $v_\theta^{(1)}$ to run the ODE and generate **coupled** pairs $(x_0, x_1^{(1)})$, i.e. $x_1^{(1)} = \text{ODE}(x_0; v_\theta^{(1)})$

3. Train again on coupled pairs to obtain $v_\theta^{(2)}$ — the new trajectories are **straighter**

4. Repeat — Liu et al. 2022 prove that under suitable assumptions, the **convex transport cost** of the coupling is non-increasing (each reflow does not worsen total transport cost)

"Trajectories become straighter" is intuition + empirical observation; the rigorous theorem is monotonicity of transport cost. In practice 1-2 reflow passes make 4-step quality match 50-step (InstaFlow / SD3-Turbo / Flux-Schnell). The limit: completely straight → 1-step generation ($x_1 = x_0 + v_\theta(0, x_0)$).

### 7.2　Conditional Flow Matching (CFG)

For conditional generation (e.g. text-to-image), the model takes an extra condition $c$:

$$v_\theta(t, x, c)$$

During training, with probability $p_\text{drop}$ (typically 0.1), $c$ is replaced by a null token (e.g. null embedding), yielding an **unconditional head**.

At sampling time, use **Classifier-Free Guidance**:

$$v_\text{CFG}(t, x, c) = v_\theta(t, x, \emptyset) + s \cdot \left[v_\theta(t, x, c) - v_\theta(t, x, \emptyset)\right]$$

$s$ is the guidance scale (typically 1.5-7.5). $s > 1$ amplifies the conditional signal, improving text alignment but reducing diversity.

### 7.3　Logit-normal $t$ (SD3 default)

SD3 (Esser et al. 2024) finds that **$t \sim \mathcal{U}[0, 1]$ is not optimal**. The middle region ($t \approx 0.5$) has the most difficult target noise-signal ratio. Replace with:

$$t = \sigma(\tau), \quad \tau \sim \mathcal{N}(m, s^2)$$

i.e. Gaussian-sample $\tau$ then sigmoid-map back to $(0, 1)$. Tune $m, s$ to control which range of $t$ is emphasized. With default $m = 0, s = 1$, $t$ concentrates near 0.5. This is one of the key ablation wins in the SD3 paper.

## §8 Complete runnable example (2D toy)

Below is an end-to-end minimal runnable example: train a vector field to map $\mathcal{N}(0, I)$ to a 2D moon-shaped distribution.

```python
if __name__ == "__main__":
    # 1) Data (target distribution p_1): 2D moons
    from sklearn.datasets import make_moons

    def sample_moons(n: int) -> torch.Tensor:
        X, _ = make_moons(n_samples=n, noise=0.05)
        return torch.tensor(X, dtype=torch.float32) * 2.0  # scale

    # 2) Model + path
    model = VectorFieldMLP(dim=2, hidden=128)
    path = rectified_flow_path()

    # 3) "dataloader" (random generation)
    class MoonDataset:
        def __init__(self, batch=512, total=5000):
            self.batch = batch; self.total = total
        def __iter__(self):
            for _ in range(self.total):
                yield sample_moons(self.batch)

    # 4) Train
    train_flow_matching(
        model,
        MoonDataset(batch=512, total=2000),
        path=path,
        total_steps=2000,
        lr=3e-4,
        device="cuda" if torch.cuda.is_available() else "cpu",
        log_every=100,
    )

    # 5) Sample
    model.eval()
    device = next(model.parameters()).device
    x0 = torch.randn(2000, 2, device=device)
    x_samples = euler_sampler(model, x0, steps=50)

    # Overlay with real 2D moons for visual sanity check
    import matplotlib.pyplot as plt
    real = sample_moons(2000).numpy()
    fake = x_samples.cpu().numpy()
    plt.scatter(real[:, 0], real[:, 1], alpha=0.3, label="real")
    plt.scatter(fake[:, 0], fake[:, 1], alpha=0.3, label="generated")
    plt.legend(); plt.savefig("flow_matching_moons.png", dpi=120)
```

> ⚠️ **Production additions (not in this pedagogical version)** — engineering items to add before deployment.

- **EMA scheduler**: decay closer to 1 in late training (e.g. 0.9999 → 0.99995)

- **Gradient checkpointing**: U-Net / DiT memory optimization

- **Mixed precision**: fp16 / bf16 + GradScaler

- **Latent space**: high-resolution images run FM in VAE latent space (LDM / SD3 / FLUX)

- **Conditioning**: text encoder (T5 / CLIP) + cross-attention or token concat

- **Distributed**: DDP / FSDP for multi-GPU

- **Loss weighting**: SD3 implicitly reweights via logit-normal $t$; EDM uses explicit SNR weighting

**Flow Matching Quick Reference** · Main references: Lipman et al. 2023 (Flow Matching), Liu et al. 2022 (Rectified Flow), Esser et al. 2024 (SD3 / MM-DiT)

---

## 📜 Runnable Code

Minimal runnable PyTorch implementations of this tutorial's core concepts live in [`docs/tutorials/code/`](code/):

- [`flow_matching.py`](code/flow_matching.py) — Rectified Flow on 2D toy moons: train to regress `(x_1 - x_0)` velocity + Euler ODE sampling + trajectory viz

Runs on CPU in ~30 seconds (matplotlib optional) with built-in sanity checks. Full overview in [`code/README.md`](code/README.md).
