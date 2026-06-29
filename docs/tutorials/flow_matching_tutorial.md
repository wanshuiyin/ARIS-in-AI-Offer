## §0 TL;DR

> 💡 **5 句话搞定 Flow Matching** — 一页拿下核心要点（详见后文 §1–§4 推导）。

1. **目标**：学一个 vector field $v_\theta(t, x)$，使 ODE $\dot{x}_t = v_\theta(t, x_t)$ 把 $x_0 \sim p_0$（噪声）演化到 $x_1 \sim p_1$（数据）。

2. **训练 (CFM)**：$\mathcal{L}_\text{CFM}(\theta) = \mathbb{E}_{t, z, x_t \sim p_t(\cdot|z)} \|v_\theta(t, x_t) - u_t(x_t|z)\|^2$，**simulation-free**（不用解 ODE 算 loss）。

3. **关键定理**：$\nabla_\theta \mathcal{L}_\text{FM} = \nabla_\theta \mathcal{L}_\text{CFM}$——所以学 conditional vector field 等价于学 marginal 的（Lipman et al. 2023）。

4. **最简版 (Rectified Flow / OT-CFM)**：$x_t = (1-t)x_0 + tx_1$，target $u_t = x_1 - x_0$。SD3 / FLUX / Lumina 都用这个。

5. **采样**：从 $x_0 \sim p_0$ 出发，用 ODE solver (Euler / Heun / RK4) 积分到 $t=1$。

## §1 基本设定 & 直觉

给定数据分布 $p_1$（"目标"）和简单先验 $p_0$（一般 $\mathcal{N}(0, I)$），我们想构造一族**概率路径** $\{p_t\}_{t \in [0,1]}$ 从 $p_0$ 平滑过渡到 $p_1$。

> ⚠️ **Convention（全文统一）** — 后续公式记号见下表。

- $x_0 \sim p_0 = \mathcal{N}(0, I)$（噪声端）—— $t=0$

- $x_1 \sim p_1$（数据端）—— $t=1$

- 采样方向：从 $t=0$ 积分到 $t=1$（noise → data）

- 注：不同论文 convention 不同——Lipman et al. 2023 用 $x_0$=data, $x_1$=noise；Liu et al. 2022 (Rectified Flow) 用 $x_0$=noise, $x_1$=data（本文采用）。SD3 论文也是 noise→data 但记号略有差异。**面试时第一句先 disambiguate**。

一族**时变 vector field** $u_t : [0,1] \times \mathbb{R}^d \to \mathbb{R}^d$ 通过 ODE $\dot{x}_t = u_t(x_t)$ 把粒子从 $p_0$ 推到 $p_1$。由**连续性方程**：

$$\boxed{\;\frac{\partial p_t}{\partial t} + \nabla \cdot (p_t\, u_t) = 0\;}$$

我们的目标：找一个神经网络 $v_\theta(t, x) \approx u_t(x)$。

```

   p_0 (噪声)              p_t (中间)              p_1 (数据)
   ●●●●●          →        ●  ●  ●        →        ████
                            v_θ(t, x)
                           ─────────→
                          dx/dt = v_θ
```

对比 diffusion：

- **Diffusion (SDE)**：$dx = f(x, t) dt + g(t) dW$，训练用 score matching $s_\theta \approx \nabla \log p_t$

- **Flow matching (ODE)**：$dx = v_\theta(t, x) dt$，**无随机项**，训练直接回归 vector field

- 两者通过 **probability flow ODE** 关联：$v = f - \frac{1}{2} g^2 \nabla \log p_t$（见 §6）

## §2 Flow Matching Loss

### 2.1　Marginal Flow Matching（理论上）

如果我们知道 $u_t$（marginal vector field），直接回归即可：

$$\mathcal{L}_\text{FM}(\theta) = \mathbb{E}_{t \sim \mathcal{U}[0,1],\; x \sim p_t} \left\| v_\theta(t, x) - u_t(x) \right\|^2$$

**问题**：$u_t(x)$ 是 marginal，由所有 conditional path 加权（积分），**无法直接采样**。

### 2.2　Conditional Flow Matching（实际可训练）

引入 **conditioning variable** $z$（如 $z = x_1$，或 $z = (x_0, x_1)$）。选 conditional path $p_t(x | z)$ 和 conditional vector field $u_t(x | z)$，使得对 $z$ 边缘后等于 marginal：

$$p_t(x) = \int p_t(x | z) q(z)\, dz, \quad u_t(x) = \int u_t(x|z) \frac{p_t(x|z) q(z)}{p_t(x)} dz$$

那么 **Conditional FM loss**：

$$\boxed{\;\mathcal{L}_\text{CFM}(\theta) = \mathbb{E}_{t,\; z \sim q,\; x \sim p_t(\cdot|z)} \left\| v_\theta(t, x) - u_t(x|z) \right\|^2\;}$$

每一项都**可采样、可计算**。$x \sim p_t(\cdot|z)$ 一般是闭式可采样（如下面的线性插值）。

### 2.3　关键定理（Lipman et al. 2023, Theorem 2）

> ✅ **梯度等价定理** — 在 $p_t$ 与 $u_t$ 适当正则、$p_t > 0$ 等假设下：
$$\nabla_\theta \mathcal{L}_\text{FM}(\theta) = \nabla_\theta \mathcal{L}_\text{CFM}(\theta)$$
所以 **minimize CFM ≡ minimize FM**。两个 loss 在以上前提下相差一个不依赖 $\theta$ 的常数。

**证明草图**：展开二范数 $\|v_\theta\|^2 - 2 v_\theta^\top u_t + \|u_t\|^2$，前两项在两种 loss 下相等（用 $u_t$ 的定义把 marginal 写成 conditional 的加权期望），第三项不依赖 $\theta$，求梯度时消掉。

> 💡 **面试加分：marginal vector field 非唯一** — 给定 $p_t$，满足连续性方程 $\partial_t p_t + \nabla\cdot(p_t u_t) = 0$ 的 $u_t$ **不唯一**——任意 divergence-free 向量场加上去仍是合法的。CFM 通过 conditional path 自动选了一个"自然的" $u_t$（通常对应 OT map 或 score-based ODE）。这点常被追问 "marginal $u_t$ 唯一吗？"。

## §3 三种 Conditional Path 选择

conditioning 取 $z = (x_0, x_1)$，$x_0 \sim p_0$, $x_1 \sim p_1$。conditional path $p_t(x | x_0, x_1)$ 一般是 Dirac $\delta(x - \psi_t(x_0, x_1))$（确定性插值），conditional vector field 为 $\dot{\psi}_t(x_0, x_1)$。

| Path | $x_t = \psi_t(x_0, x_1)$ | Target $u_t$ | 用在哪 |
| --- | --- | --- | --- |
| **Rectified Flow / OT-CFM** | $(1-t)x_0 + t\, x_1$ | $x_1 - x_0$（常数） | SD3, FLUX, Lumina, MovieGen |
| **VP cosine** | $\cos\!\left(\frac{\pi t}{2}\right) x_0 + \sin\!\left(\frac{\pi t}{2}\right) x_1$ | $-\frac{\pi}{2}\sin\!\frac{\pi t}{2}\, x_0 + \frac{\pi}{2}\cos\!\frac{\pi t}{2}\, x_1$ | 与 DDPM cosine schedule 同族（在限制下） |
| **VE** | $x_1 + \sigma(1{-}t)\, x_0$，$\sigma$ 递增 | $-\sigma'(1{-}t)\, x_0$ | 与 SMLD/EDM 同族（需 prior 方差匹配 $\sigma_{\max}^2$） |

### 3.1　Rectified Flow：最简、最稳、最常用

线性插值：$x_t = (1-t) x_0 + t\, x_1$，所以 $\dot{x}_t = x_1 - x_0$ 是**常数**（不依赖 $t$）。

训练目标：

$$\mathcal{L}_\text{RF}(\theta) = \mathbb{E}_{t, x_0, x_1} \|v_\theta(t,\, (1-t)x_0 + t x_1) - (x_1 - x_0)\|^2$$

"OT-CFM" 名字来自：如果 $(x_0, x_1)$ 是 optimal transport coupling（不是独立采样），则学到的 vector field 实现近似 OT map。

> ✅ **Reflow：Rectified Flow 的杀手锏** — 用学到的 $v_\theta$ 重新生成 $(x_0, x_1)$ pair（用 $x_0$ 跑 ODE 得到对应 $x_1$），**再训一次**。新的 trajectory 更直，**少步数采样质量大幅提升**，可以做到 1-step / 2-step 生成（InstaFlow 等）。

### 3.2　VP path（与 DDPM 同族）

用 $\sigma(t) = \cos\!\frac{\pi t}{2}$（噪声系数）, $\alpha(t) = \sin\!\frac{\pi t}{2}$（数据系数），满足 $\sigma^2 + \alpha^2 = 1$（variance preserving）：

$$x_t = \sigma(t)\, x_0 + \alpha(t)\, x_1, \quad u_t = \sigma'(t)\, x_0 + \alpha'(t)\, x_1$$

边界：$t=0$ 时 $x_t = x_0$（噪声），$t=1$ 时 $x_t = x_1$（数据）。

这种 path 和 DDPM 的 cosine schedule **属于同一族 Gaussian path**（连续极限 + 时间反向）。但严格意义上不能说"完全等价"——DDPM 原文 (Nichol-Dhariwal) 还有 $s=0.008$ offset 等细节，且 DDPM 是 forward noising convention（$t=0$ 数据），FM 是反向（$t=0$ 噪声）。

### 3.3　VE path（与 SMLD/EDM 同族）

沿用 Lipman et al. 2023 的 conditional VE path：

$$p_t(x | x_1) = \mathcal{N}\!\left(x \,\Big|\, x_1,\; \sigma(1-t)^2 I\right)$$

$\sigma(s)$ 在 forward time $s \in [0, 1]$ 上单调递增（如 $\sigma(s) = \sigma_\min (\sigma_\max/\sigma_\min)^s$）。reparameterize 得

$$x_t = x_1 + \sigma(1-t)\, x_0, \quad u_t = -\sigma'(1-t)\, x_0$$

边界：$t=0$ 时 $x_t \approx x_1 + \sigma_\max\, x_0$（大噪声主导），$t=1$ 时 $x_t \approx x_1 + \sigma_\min\, x_0 \approx x_1$（数据）。

> ⚠️ **VE 部署注意** — prior $p_0$ 严格意义上应是 $\mathcal{N}(0, \sigma_\max^2 I)$（让 $t=0$ 的边缘方差匹配）；用 $\mathcal{N}(0, I)$ 时要相应缩放（如 $x_0 \leftarrow \sigma_\max \cdot \tilde{x}_0$）。本文代码示例以教学为主，**实战 VE 用 EDM preconditioning** 更稳。

## §4 训练代码框架（PyTorch）

### 4.1　Probability Path 抽象

```python
import math
from dataclasses import dataclass
from typing import Callable, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class FlowPath:
    """ Conditional probability path 抽象 """
    name: str
    sample_xt: Callable    # (t, x0, x1) -> x_t
    target_ut: Callable    # (t, x0, x1) -> u_t

def _broadcast_t(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """ t: [B], x: [B, ...] —— 把 t 扩成可广播 x 的 shape [B, 1, 1, ...] """
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
        t=0: x_t = x_0 (noise);  t=1: x_t = x_1 (data)  [noise → data 方向] """
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
    """ VE: x_t = x_1 + σ(1-t) · x_0,  σ(s) 在 forward time s 上递增 (log-linear)
        t=0: x_t = x_1 + σ_max·x_0 (大噪声);  t=1: x_t ≈ x_1 (数据)
        注: 严格 VE 需 prior p_0 ~ N(0, σ_max² I)，本示例为简化用 N(0, I)，
            实战需要 EDM-style preconditioning。 """
    log_min, log_max = math.log(sigma_min), math.log(sigma_max)
    def sigma_fwd(s):                          # 在 forward time s 上递增
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

### 4.2　Vector Field 网络（教学版 MLP；实际用 U-Net / DiT）

```python
class SinusoidalTimeEmbed(nn.Module):
    """ 与 Transformer positional embedding 同构的时间编码 """
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
    """ v_θ(t, x) ——— 简化版，用于 2D toy / low-dim 实验
        实际生成模型把这里换成 U-Net (image) 或 DiT (high-res / video) """
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
    x1: torch.Tensor,                              # [B, ...] 数据样本
    x0: Optional[torch.Tensor] = None,              # 默认 N(0, I)
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

    # t 采样
    if t_dist == "uniform":
        t = torch.rand(B, device=device)
    elif t_dist == "logitnormal":
        # SD3 默认: t = σ(z), z ~ N(0, 1). 更集中在 t≈0.5（最难学的中间区）
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

### 4.4　Minimal Training Loop

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
    ema_decay: float = 0.9999,           # EMA 对生成模型必加
):
    model = model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    ema_model = _make_ema(model)        # 见下

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

## §5 ODE 采样

训练完 $v_\theta$ 后，从 $x_0 \sim p_0$ 出发，解 ODE $\dot{x}_t = v_\theta(t, x_t)$ 到 $t = 1$。

```python
@torch.no_grad()
def euler_sampler(model, x0, steps=50, t_start=0.0, t_end=1.0):
    """ 一阶 Euler：每步 1 NFE，简单但需要较多步数 """
    x = x0.clone()
    ts = torch.linspace(t_start, t_end, steps + 1, device=x0.device)
    for i in range(steps):
        t = ts[i].expand(x.shape[0])
        dt = ts[i + 1] - ts[i]
        x = x + dt * model(t, x)
    return x

@torch.no_grad()
def heun_sampler(model, x0, steps=50, t_start=0.0, t_end=1.0):
    """ 二阶 Heun (improved Euler / RK2)：每步 2 NFE，精度 O(dt²) """
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
    """ 四阶 Runge-Kutta：每步 4 NFE，精度 O(dt⁴)
        25 步 × 4 NFE = 100 NFE，但通常比 100 步 Euler 准很多 """
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

> 💡 **Sampler 选择 cheat sheet** — 按 NFE / 质量 trade-off 排序如下。

- **Euler**：1 NFE/step，需 ≥50 步才出好图；调试 baseline 用

- **Heun / RK2**：2 NFE/step，~25 步质量已不错；EDM 默认

- **RK4**：4 NFE/step，10-20 步通常等同 Euler 100 步

- **Adaptive (Dopri5 / dopri8)**：torchdiffeq 提供；自动控制误差，但 NFE 不可控

- **Rectified Flow 重训后**：经 1-2 次 reflow，1-4 步 Euler 即可达到接近多步质量

## §6 与 Diffusion / Score Matching 的关系

对任意 SDE $dx = f(x, t) dt + g(t) dW$（forward），存在对应的 **probability flow ODE**（Song et al. 2021）：

$$dx = \underbrace{\left[ f(x, t) - \frac{1}{2} g^2(t)\, \nabla_x \log p_t(x) \right]}_{\text{vector field } u_t(x)} dt$$

这个 ODE 在每一时刻的边缘分布 $p_t$ 与 SDE 完全一样。

> ✅ **FM ↔ Score Matching 的桥梁（带前提）** — 当 FM 的概率路径源自一个非退化的 noising SDE ($g(t) > 0$) 时，学 score $s_\theta \approx \nabla \log p_t$ 与学 vector field $v_\theta \approx u_t$ 是**同一信息的两种参数化**：
$$v_\theta(t, x) = f(x, t) - \tfrac{1}{2} g^2(t)\, s_\theta(t, x)$$
所以在 VP/VE path 下，FM 可被视为 score matching 在 ODE 视角下的等价参数化。**但对 Rectified Flow / OT-CFM 不成立**（没有标准 SDE 对应），那里 FM 是更一般的 vector-field regression。

### 6.1　Velocity ↔ Score ↔ Noise prediction 互换（必考）

VP/VE path 下，假设 $x_t = \alpha(t) x_1 + \sigma(t) x_0$（$x_0 \sim \mathcal{N}(0, I)$ 是噪声方向），三种主流 prediction target 之间是线性可逆转换：

$$
\begin{aligned}
\epsilon\text{-prediction} &:\quad \epsilon_\theta(t, x_t) \approx x_0 \\
x_0\text{-prediction} &:\quad x^0_\theta(t, x_t) \approx x_1 \\
v\text{-prediction (Salimans-Ho)} &:\quad v_\theta(t, x_t) \approx \alpha'(t) x_1 + \sigma'(t) x_0 \\
\text{score} &:\quad s_\theta(t, x_t) \approx -x_0 / \sigma(t)
\end{aligned}
$$

已知 $x_t$ 和任一 prediction，可代数恢复其他三种。例如 VP 下 $\epsilon$ 与 score 关系：

$$s_\theta(t, x_t) = -\epsilon_\theta(t, x_t) / \sigma(t)$$

这就是为什么 DDPM (学 $\epsilon$) 与 score-based (学 $\nabla \log p_t$) 是**等价参数化**。Flow matching 学 $v = \alpha' x_1 + \sigma' x_0$ 也是其中一种，且在 RF (linear) 下退化为 $v = x_1 - x_0$。

### 6.2　几种 path 与 diffusion 的对应表

| FM Path | 等价 Diffusion / SDE | 典型 noise schedule |
| --- | --- | --- |
| VP cosine | DDPM (cosine) | $\bar\alpha_t = \cos^2(\pi t/2)$ |
| VP linear | DDPM (linear β) | $\beta_t = \beta_0 + t(\beta_1 - \beta_0)$ |
| VE | SMLD / EDM | $\sigma_t \in [\sigma_\min, \sigma_\max]$ 对数线性 |
| Rectified Flow | 没有标准非零扩散 noising SDE 对应（退化情形除外） | 路径本身是直线，"最短" path |

### 6.3　为什么 Rectified Flow 训练 / 采样相对"稳"

- **常数 target**：$u_t = x_1 - x_0$ 不显式依赖 $t$（在给定 $x_0, x_1$ 后），数值上易拟合

- **直线 path**：少步数 ODE 积分误差小

- **Loss conditioning**：RF 本身比 native DDPM 训练更平衡；但**不是说不需要 reweighting**——SD3 在 RF 之上仍做 logit-normal $t$ sampling 等 reweighting 并 ablate 出涨点

- **Reflow 可压缩 NFE**：1-step 生成路线（InstaFlow / SD3-Turbo / Flux-Schnell）

## §7 高级话题

### 7.1　Reflow（Liu et al., ICLR 2023）

Rectified Flow 之所以能少步数生成，关键是 **reflow 算法**：

1. 第一次训练得到 $v_\theta^{(1)}$（用独立配对 $(x_0, x_1) \sim p_0 \otimes p_1$）

2. 用 $v_\theta^{(1)}$ 跑 ODE 生成**coupled** pair $(x_0, x_1^{(1)})$，即 $x_1^{(1)} = \text{ODE}(x_0; v_\theta^{(1)})$

3. 用 coupled pair 重新训练得到 $v_\theta^{(2)}$，新的 trajectory 更**直**

4. 重复——Liu et al. 2022 证明在适当假设下，配对的 **convex transport cost** 非增（每次 reflow 不会让总传输成本变差）

"trajectory 变直"是直觉与经验观察；具体严格定理是 transport cost 单调性。实际 1-2 次 reflow 就能做到 4-step 媲美 50-step（InstaFlow / SD3-Turbo / Flux-Schnell）。极限：完全直线 → 1-step 生成（$x_1 = x_0 + v_\theta(0, x_0)$）。

### 7.2　Conditional Flow Matching (CFG)

对条件生成（如 text-to-image），模型接收额外条件 $c$：

$$v_\theta(t, x, c)$$

训练时以概率 $p_\text{drop}$（一般 0.1）把 $c$ 替换成空（如 null embedding），得到 **无条件 head**。

采样时用 **Classifier-Free Guidance**：

$$v_\text{CFG}(t, x, c) = v_\theta(t, x, \emptyset) + s \cdot \left[v_\theta(t, x, c) - v_\theta(t, x, \emptyset)\right]$$

$s$ 是 guidance scale（一般 1.5-7.5）。$s > 1$ 时放大 conditional 信号，提升文本对齐但损失多样性。

### 7.3　Logit-normal $t$（SD3 默认）

SD3 (Esser et al. 2024) 发现，**$t \sim \mathcal{U}[0, 1]$ 不是最优**。中间区域（$t \approx 0.5$）的 target 噪声-信号比最难学。改成：

$$t = \sigma(\tau), \quad \tau \sim \mathcal{N}(m, s^2)$$

即 $\tau$ 高斯采样后 sigmoid 映射回 $(0, 1)$，可调 $m, s$ 控制 $t$ 分布偏重哪段。默认 $m = 0, s = 1$ 时 $t$ 集中在 0.5 附近。这是 SD3 论文中 ablation 涨点的关键之一。

## §8 完整可运行示例（2D toy）

下面是端到端最小可运行示例：训练 vector field 学习把 $\mathcal{N}(0, I)$ 映到一个 2D 月亮形分布。

```python
if __name__ == "__main__":
    # 1) 数据 (target distribution p_1): 2D 月亮形
    from sklearn.datasets import make_moons

    def sample_moons(n: int) -> torch.Tensor:
        X, _ = make_moons(n_samples=n, noise=0.05)
        return torch.tensor(X, dtype=torch.float32) * 2.0  # scale

    # 2) 模型 + path
    model = VectorFieldMLP(dim=2, hidden=128)
    path = rectified_flow_path()

    # 3) "dataloader" (随机生成)
    class MoonDataset:
        def __init__(self, batch=512, total=5000):
            self.batch = batch; self.total = total
        def __iter__(self):
            for _ in range(self.total):
                yield sample_moons(self.batch)

    # 4) 训练
    train_flow_matching(
        model,
        MoonDataset(batch=512, total=2000),
        path=path,
        total_steps=2000,
        lr=3e-4,
        device="cuda" if torch.cuda.is_available() else "cpu",
        log_every=100,
    )

    # 5) 采样
    model.eval()
    device = next(model.parameters()).device
    x0 = torch.randn(2000, 2, device=device)
    x_samples = euler_sampler(model, x0, steps=50)

    # 与真实 2D 月亮叠图，可视化检验
    import matplotlib.pyplot as plt
    real = sample_moons(2000).numpy()
    fake = x_samples.cpu().numpy()
    plt.scatter(real[:, 0], real[:, 1], alpha=0.3, label="real")
    plt.scatter(fake[:, 0], fake[:, 1], alpha=0.3, label="generated")
    plt.legend(); plt.savefig("flow_matching_moons.png", dpi=120)
```

> ⚠️ **Production 化要补的（教学版未含）** — 上线前需补的工程项如下。

- **EMA scheduler**：训练后期 decay 更接近 1（如 0.9999 → 0.99995）

- **Gradient checkpointing**：U-Net / DiT 显存优化

- **Mixed precision**：fp16 / bf16 + GradScaler

- **Latent space**：高分辨率图像在 VAE latent 里跑 FM（LDM / SD3 / FLUX）

- **Conditioning**：text encoder (T5 / CLIP) + cross-attention 或 token concat

- **Distributed**：DDP / FSDP for multi-GPU

- **Loss weighting**：SD3 用 logit-normal $t$ 已隐式 reweighting；EDM 显式 SNR weighting

**Flow Matching Quick Reference** · 主要参考：Lipman et al. 2023 (Flow Matching), Liu et al. 2022 (Rectified Flow), Esser et al. 2024 (SD3 / MM-DiT)

---

## 📜 Runnable Code

本 tutorial 的核心概念在 [`docs/tutorials/code/`](code/) 里有最小可跑的 PyTorch 实现：

- [`flow_matching.py`](code/flow_matching.py) — Rectified Flow on 2D toy moons：训练 `(x_1 - x_0)` velocity + Euler ODE 采样 + 轨迹可视化

CPU 30 秒左右跑完（matplotlib 可选），自带 sanity 检查。完整说明见 [`code/README.md`](code/README.md)。
