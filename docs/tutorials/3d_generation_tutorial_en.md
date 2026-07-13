## §0 TL;DR Cheat Sheet

> 💡 **9 sentences to nail 3D Generation** — interview core for Embodied AI / AR / VR roles (see §1–§11 for derivations).

1. **Three representations**: **NeRF** (implicit neural field + volume rendering), **3DGS** (explicit Gaussian point cloud + rasterization), **Mesh / SDF** (explicit surface / implicit distance field). Sweet spot for quality vs speed: 3DGS (Kerbl 2023 SIGGRAPH Best Paper).

2. **NeRF core equation**: $C(\mathbf{r}) = \int_{t_n}^{t_f} T(t)\sigma(\mathbf{r}(t))\mathbf{c}(\mathbf{r}(t),\mathbf{d})\,dt$, where $T(t) = \exp\!\left(-\int_{t_n}^{t}\sigma(\mathbf{r}(s))\,ds\right)$ is the transmittance. Discretization yields $\alpha$-compositing: $C \approx \sum_i T_i (1-e^{-\sigma_i\delta_i})\mathbf{c}_i$.

3. **Instant-NGP** (Müller 2022 SIGGRAPH): **multi-resolution hash grid** + tiny MLP, roughly a 4+ order-of-magnitude speedup; hash collisions are automatically disambiguated by the MLP over colliding entries (suppressed jointly by the loss and multi-scale redundancy).

4. **3DGS core**: scene represented as a set of 3D Gaussians $\{\mu_i, \Sigma_i, \alpha_i, c_i(\mathbf{d})\}$, **differentiable rasterization** projects the 3D covariance to 2D via Jacobian $J$: $\Sigma' = J W \Sigma W^\top J^\top$, then performs front-to-back alpha-blending after depth sorting.

5. **DreamFusion SDS** (Poole et al. 2022 arXiv → ICLR 2023 Outstanding): supervise a 3D representation using a pretrained 2D diffusion: $\nabla_\theta \mathcal{L}_\text{SDS} = \mathbb{E}_{t,\epsilon}[w(t)(\epsilon_\phi(x_t;y,t)-\epsilon)\,\partial x/\partial \theta]$, **deliberately dropping the U-Net Jacobian** of $x_t$ to make training simulation-free. Price: mode-seeking → over-saturation / Janus.

6. **VSD** (Wang 2023 NeurIPS, ProlificDreamer): treats the 3D parameters $\theta$ as a random variable $\mu(\theta)$ and **minimizes the KL between the noised rendered image distributions**: $\mathbb{E}_t\big[D_\text{KL}\big(q_\mu^t(x_t|y)\,\|\,p_\phi^t(x_t|y)\big)\big]$. The gradient form is a **relative score** $\nabla_\theta \approx (\epsilon_\phi(x_t;y,t) - \epsilon_\psi(x_t;y,t,\pi))\,\partial x/\partial\theta$, where $\epsilon_\psi$ is a LoRA-finetuned auxiliary score. CFG can drop from 100 to 7.5.

7. **Single-image / Few-view 3D**: Zero-1-to-3 (Liu 2023 ICCV) uses viewpoint-conditioned diffusion; SyncDreamer / MVDream learn joint multi-view consistency; TripoSR / InstantMesh / Stable Fast 3D push image-to-mesh to the seconds regime (TripoSR ~0.5 s, InstantMesh ~10 s).

8. **3D Foundation Models (2024-25 open source)**: **Trellis** (Microsoft 2024) uses structured latent + flow matching; **Hunyuan3D-2** (Tencent 2025) is shape→texture two-stage; **CLAY** (Zhang 2024 SIGGRAPH, arXiv:2406.13897) is large-scale latent diffusion = multi-resolution VAE + latent DiT.

9. **Embodied AI key applications**: Sim2Real asset generation, NeRF/3DGS as differentiable simulators, language-conditioned 3D affordance. **Common interview crossovers**: NeRF SLAM, Gaussian-Splat scene editing, 3D physics consistency.

## §1 Intuitive comparison of the three representations

The first multiple-choice question in 3D generation is **representation** — pick the wrong one and the entire downstream pipeline is wasted.

|  | NeRF (Implicit Field) | 3DGS (Explicit Point) | Mesh / SDF |
| --- | --- | --- | --- |
| **Storage** | MLP weights $f_\theta(\mathbf{x},\mathbf{d}) \to (\sigma, \mathbf{c})$ | A pile of 3D Gaussians $\{\mu_i, \Sigma_i, \alpha_i, c_i\}$ | Triangle mesh / signed distance |
| **Rendering** | Ray marching + volume integration (hundreds of ms/frame on GPU) | Differentiable rasterization (a few ms/frame on GPU) | Rasterization (real-time) |
| **Training** | Hundreds of views, hours (vanilla) | Tens of views, 10-30 min | Requires mesh + texture optim |
| **Quality** | SOTA for view synthesis | On par with or better than NeRF (higher PSNR) | Limited by polygon resolution |
| **Editing** | Hard (neural field is not interpretable) | Easy (points can be moved, deleted, merged) | Easy (standard DCC pipelines) |
| **Mesh export** | Hard (needs NeuS / Poisson) | Medium (2DGS / GSDF / SuGaR) | Already a mesh |
| **Downstream fit** | Unfriendly for physical simulation | Needs mesh conversion first (no collision geometry/materials/physics params); standard robot/physics-sim pipelines still run on mesh | Standard robot / AR/VR pipeline |

> 💡 **Interview intuition** — Embodied AI leans toward mesh / 3DGS (simulator-friendly); AR/VR depends on scene scale (mesh for small foreground objects, 3DGS for large scenes); SOTA visual reconstruction uses 3DGS. NeRF is now more of a research baseline; industrial deployment is dominated by 3DGS.

## §2 NeRF: derivation of volume rendering (must-know)

### 2.1　Continuous volume-rendering equation

NeRF (Mildenhall 2020 ECCV **Best Paper Honorable Mention**) represents a scene as a **5D neural field** $f_\theta : (\mathbf{x}, \mathbf{d}) \to (\sigma, \mathbf{c})$:

- Input: 3D position $\mathbf{x} \in \mathbb{R}^3$ + view direction $\mathbf{d} \in \mathbb{S}^2$
- Output: volume density $\sigma \ge 0$ (direction-independent) + color $\mathbf{c} \in \mathbb{R}^3$ (direction-dependent, captures specular reflection)

For camera ray $\mathbf{r}(t) = \mathbf{o} + t\mathbf{d}$, integrate along $t \in [t_n, t_f]$ to get pixel color:

$$\boxed{\;C(\mathbf{r}) = \int_{t_n}^{t_f} T(t)\,\sigma(\mathbf{r}(t))\,\mathbf{c}(\mathbf{r}(t),\mathbf{d})\,dt\;}$$

where **transmittance** (the probability that the ray has not been blocked between $t_n$ and $t$) is

$$\boxed{\;T(t) = \exp\!\left(-\int_{t_n}^{t}\sigma(\mathbf{r}(s))\,ds\right)\;}$$

### 2.2　Why this form? — derivation from physics

Consider a light ray traveling through a participating medium. Over $[t, t+dt]$:

- Probability of absorption / scattering out of the ray: $\sigma(\mathbf{r}(t))\,dt$
- Color contribution emitted by the medium at this point: $\mathbf{c}(\mathbf{r}(t),\mathbf{d})$

Let $T(t)$ be the survival probability of the ray from $t_n$ to $t$. From $t \to t + dt$, the survival probability changes as

$$T(t+dt) = T(t)\,(1 - \sigma\,dt) \;\Rightarrow\; \frac{dT}{dt} = -\sigma(t)\,T(t)$$

This is a first-order ODE with initial value $T(t_n) = 1$, solving to

$$T(t) = \exp\!\left(-\int_{t_n}^{t}\sigma(\mathbf{r}(s))\,ds\right)$$

The color contribution to the pixel at each depth $t$ = **survival probability × absorption probability × local color**:

$$dC = T(t)\,\sigma(t)\,\mathbf{c}(t)\,dt$$

Integrating gives $C(\mathbf{r})$.

### 2.3　Discretization: $\alpha$-compositing (**must-derive**)

Continuous integration is impossible in practice. Slice $[t_n, t_f]$ into $N$ segments with $\delta_i = t_{i+1} - t_i$, and assume $\sigma, \mathbf{c}$ are constants $\sigma_i, \mathbf{c}_i$ within each segment.

**Within-segment transmittance decay**: on $[t_i, t_{i+1}]$, $T$ satisfies $dT/dt = -\sigma_i T$, so

$$\frac{T(t_{i+1})}{T(t_i)} = e^{-\sigma_i \delta_i}$$

This gives **inter-segment transmittance**:

$$T_i := T(t_i) = \prod_{j=1}^{i-1} e^{-\sigma_j \delta_j} = \exp\!\Big(\!-\!\sum_{j=1}^{i-1}\sigma_j\delta_j\Big)$$

**Within-segment color contribution** (integral, not a simple rectangle):

$$\int_{t_i}^{t_{i+1}} T(t)\sigma_i\mathbf{c}_i\,dt = T_i\,\mathbf{c}_i \int_0^{\delta_i} \sigma_i e^{-\sigma_i s}\,ds = T_i\,\mathbf{c}_i\,(1 - e^{-\sigma_i\delta_i})$$

Let $\alpha_i := 1 - e^{-\sigma_i \delta_i}$ (**segment opacity**). Combining yields the discrete NeRF equation:

$$\boxed{\;C(\mathbf{r}) \approx \sum_{i=1}^{N} T_i\,\alpha_i\,\mathbf{c}_i,\quad T_i = \prod_{j<i}(1-\alpha_j),\quad \alpha_i = 1 - e^{-\sigma_i\delta_i}\;}$$

This is exactly the graphics **front-to-back alpha-compositing** formula. **Key point**: $\alpha_i = 1 - e^{-\sigma_i\delta_i}$, not $\sigma_i\delta_i$; they are approximately equal when $\sigma_i\delta_i$ is small (first-order Taylor), but differ noticeably when large (saturating to 1 vs diverging linearly).

> ✅ **Physical consistency** — $\sigma \ge 0$ and $\alpha = 1 - e^{-\sigma\delta} \in [0, 1)$ guarantee that composited color **always lies in [0, 1]**; and regardless of how the ray traverses, $\sum T_i\alpha_i \le 1$ (the remaining $T_{N+1}$ goes to background).

### 2.4　Positional encoding $\gamma(p)$: representing high-frequency detail

MLPs default to a low-frequency bias (NTK analysis); fitting $f(\mathbf{x})$ directly produces blur. NeRF uses **positional encoding** to lift the frequency:

$$\gamma(p) = \big(\sin(2^0\pi p),\cos(2^0\pi p),\sin(2^1\pi p),\cos(2^1\pi p),\dots,\sin(2^{L-1}\pi p),\cos(2^{L-1}\pi p)\big)$$

Use $L=10$ for $\mathbf{x}$ (60 dims) and $L=4$ for $\mathbf{d}$ (24 dims). Tancik et al. 2020 "Fourier Features" later gave an NTK explanation: high-frequency $\sin/\cos$ slow the kernel decay, allowing the MLP to learn high frequencies.

### 2.5　Hierarchical sampling: coarse → fine

- **Coarse network**: uniformly sample $N_c = 64$ points, render to get weights $w_i = T_i \alpha_i$
- **Fine network**: normalize $w$ to a PDF and importance-sample $N_f = 128$ new points (**importance sampling**: regions near the surface have larger weights and should be densely sampled)
- Composite the final color with all $N_c + N_f$ coarse + fine points
- Loss: $\mathcal{L} = \|C_c - C_\text{gt}\|^2 + \|C_f - C_\text{gt}\|^2$ (supervise both networks; the coarse one provides a well-defined gradient for the sampler)

### 2.6　NeRF training code (core 30 lines)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def positional_encoding(x: torch.Tensor, L: int) -> torch.Tensor:
    """ x: [..., D]; returns [..., D*2*L]; NeRF γ(p) does not include the raw x """
    freqs = 2.0 ** torch.arange(L, device=x.device, dtype=x.dtype) * torch.pi
    args = x.unsqueeze(-1) * freqs                # [..., D, L]
    pe = torch.stack([torch.sin(args), torch.cos(args)], dim=-1)  # [..., D, L, 2]
    return pe.flatten(-3)                          # [..., D*2*L]

def volume_render(sigma: torch.Tensor, color: torch.Tensor, t_vals: torch.Tensor,
                  ray_d: torch.Tensor):
    """
    NeRF discrete α-compositing
        sigma:   [B, N]        volume density (>= 0; usually after softplus / ReLU)
        color:   [B, N, 3]     color
        t_vals:  [B, N]        t values of sampled points along the ray (monotonically increasing)
        ray_d:   [B, 3]        ray direction (used to convert Δt → true distance)
    Returns C: [B, 3], weights: [B, N], depth: [B]
    """
    # δ_i = t_{i+1} - t_i; pad the last segment with 1e10 (absorbs the remainder out to infinity)
    deltas = t_vals[..., 1:] - t_vals[..., :-1]
    delta_far = torch.full_like(deltas[..., :1], 1e10)
    deltas = torch.cat([deltas, delta_far], dim=-1)            # [B, N]
    deltas = deltas * torch.norm(ray_d[:, None, :], dim=-1)    # convert t-spacing to real Euclidean distance

    alpha = 1.0 - torch.exp(-sigma * deltas)                   # [B, N]
    # T_i = ∏_{j<i} (1 - α_j) — use cumprod, shifted by one so T_1 = 1
    T = torch.cumprod(torch.cat([torch.ones_like(alpha[..., :1]),
                                 1.0 - alpha + 1e-10], dim=-1), dim=-1)[..., :-1]
    weights = T * alpha                                        # [B, N]
    C = (weights[..., None] * color).sum(dim=-2)               # [B, 3]
    depth = (weights * t_vals).sum(dim=-1)                     # [B]
    return C, weights, depth
```

> ⚠️ **Numerical footgun** — `1 - alpha` underflows to 0 when alpha approaches 1, after cumprod the entire T is zeroed; the `+ 1e-10` prevents `log(0)` during backward. The final `δ → 1e10` forces background transmittance to 0, otherwise rays through unsampled regions pick up color contamination.

### 2.7　Mip-NeRF / Mip-NeRF 360 (anti-aliasing)

Vanilla NeRF aliases badly at low resolution / when zoomed (the same pixel corresponds to cones of different scales, but NeRF treats them as rays).

- **Mip-NeRF** (Barron 2021 ICCV): treats the ray as a cone (view frustum), uses **Integrated Positional Encoding (IPE)** — closed-form Gaussian expectation of PE over the cone segment $\mathbb{E}_{\mathbf{x}\sim\mathcal{N}(\mu,\Sigma)}[\gamma(\mathbf{x})]$. For frequency vector $\boldsymbol{\omega} = 2^k\pi\,\mathbf{e}$, $\mathbb{E}_{\mathbf{x}\sim\mathcal{N}(\boldsymbol{\mu},\Sigma)}[\sin(\boldsymbol{\omega}^\top\mathbf{x})] = \sin(\boldsymbol{\omega}^\top\boldsymbol{\mu})\,e^{-\frac{1}{2}\boldsymbol{\omega}^\top\Sigma\,\boldsymbol{\omega}}$ (the scalar case is $e^{-\frac{1}{2}\omega^2\sigma^2}$); **high-frequency coefficients are automatically attenuated by the cone covariance $\Sigma$ via $e^{-\frac{1}{2}\boldsymbol{\omega}^\top\Sigma\boldsymbol{\omega}}$**, naturally achieving multi-scale behavior.
- **Mip-NeRF 360** (Barron 2022 CVPR): for unbounded scenes, applies contraction $f(x) = (2 - 1/\|x\|)\,x/\|x\|$ for $\|x\| > 1$, squashing infinity into a ball; adds distortion / proposal MLP losses.

### 2.8　NeuS / VolSDF: volume rendering + SDF (key to mesh extraction)

NeRF is density-based; mesh extraction requires choosing a $\sigma$ threshold (unstable). **NeuS** (Wang 2021 NeurIPS) replaces density with **SDF $d(\mathbf{x})$**:

$$\sigma(t) = \max\!\left(\frac{-\frac{d}{dt}\Phi_s(d(\mathbf{r}(t)))}{\Phi_s(d(\mathbf{r}(t)))},\; 0\right),\quad \Phi_s(d) = (1 + e^{-sd})^{-1}$$

where $\Phi_s$ is the sigmoid and $s$ is a learnable "sharpness". Properties: the weight peaks at the surface ($d=0$); Marching Cubes can directly extract the mesh (the mesh is $\{d = 0\}$).

**VolSDF** (Yariv 2021 NeurIPS) uses Laplace CDF $\sigma = \alpha\,\Phi(-d/\beta)$, with a similar idea.

## §3 Instant-NGP: roughly 4+ orders of magnitude faster (must-know)

Training a vanilla NeRF on one scene takes 1-2 days. **Instant-NGP** (Müller 2022 SIGGRAPH **Best Paper**) fits a simple scene in 5 seconds.

### 3.1　Core idea: multi-resolution hash grid

Replace "dense grid vs big MLP" with "**sparse hash grid + tiny MLP**".

- $L$ resolution levels (e.g. $L = 16$); the $\ell$-th level has $N_\ell = \lfloor N_\min \cdot b^\ell \rfloor$ grid points, geometric progression ($b \approx 1.38$; typical paper values $N_\min = 16$, $N_\max \in [512, 2048]$ depending on scene size)
- Each level uses a **hash function** to map grid-point coordinates into a fixed-size feature table ($T = 2^{14}$–$2^{24}$, **typically $T = 2^{19} = 524288$**)
- Query point $\mathbf{x}$: trilinear-interpolate at the 8 corner points per level → concatenate to a $L \times F$-dim feature ($F = 2$)
- Feed into a **tiny MLP** (2 layers, hidden 64) to output $\sigma, \mathbf{c}$

### 3.2　Hash function

$$\text{hash}(\mathbf{x}) = \bigg(\bigoplus_{i=1}^{d} x_i \cdot \pi_i\bigg) \bmod T$$

$\pi_i$ are large coprime constants ($\pi_1 = 1, \pi_2 = 2654435761, \pi_3 = 805459861$ — note $\pi_1 = 1$ is not prime; it is set deliberately in the paper so that the first dimension maps directly without scrambling). $\oplus$ is XOR. This is a **spatial hash**, commonly used in physics-simulation BVHs.

### 3.3　How are hash collisions disambiguated? (**L3 high-frequency follow-up**)

When $N_\ell^d > T$ (inevitable at fine levels), multiple grid points map to the same entry → collision. Why does it still work?

1. **Multi-resolution redundancy**: features at coarse levels are unique ($N_\ell^d \le T$); fine levels add detail. The MLP can recover structure from coarse features, with fine levels only responsible for detail.
2. **Sparsity prevails**: most of space is empty (most voxels in a NeRF scene are background), and meaningful queries concentrate near surfaces, so "valid colliding grid-point pairs" are rare.
3. **Gradients auto-disambiguate**: early in training, sample points in empty regions with non-zero density still receive a gradient through the volume-rendering photometric loss (gradient magnitude $\propto \delta_i e^{-\sigma_i\delta_i}$, pushing $\sigma$ toward 0); as $\sigma\to 0$ that gradient magnitude naturally decays, and once the occupancy grid updates it skips sampling there entirely — so it's not that empty-region entries "never get a gradient signal" from the start.
4. **MLP post-processing**: the tiny MLP learns a classification/regression on the $L \times F$ concatenated features; surface points with collisions can be disambiguated via **non-colliding features from other levels**.

> 💡 **Interview gold answer** — "Hash collisions seem to break uniqueness, but the **effective region is actually sparse** (a scene's thin surface occupies a tiny fraction of the voxel total), and colliding regions are mostly unsupervised background; even at colliding surface entries, the non-colliding features from other multi-resolution levels + the tiny MLP can still learn a consistent output. This is **'lazy collision resolution'**: rather than pay the cost of perfect hashing, use redundancy + data-driven disambiguation."

### 3.4　Instant-NGP training equations

Parameters: hash table $\theta_\text{hash} \in \mathbb{R}^{L \times T \times F}$ + MLP weights $\theta_\text{MLP}$. The loss is still photometric MSE, but the 5-seconds-vs-1-day gap comes from:

- **Tiny MLP**: 100× fewer parameters, ~50× faster forward
- **Hash table**: sparse activations, cache-friendly
- **Fused CUDA kernels**: tiny-cuda-nn fuses forward + backward
- **Occupancy grid**: a coarse occupancy grid skips sampling in empty regions, avoiding wasted queries

### 3.5　Plenoxels / TensoRF (contemporary explicit methods)

**Plenoxels** (Fridovich-Keil 2022 CVPR): pure voxel grid + spherical harmonics SH coefficients + density, **no MLP at all**, directly gradient-descend on voxels; speed similar to Instant-NGP but heavy on VRAM. **TensoRF** (Chen 2022 ECCV): compresses the 3D grid field via **VM / CP decomposition**, reducing parameters from $O(N^3)$ to $O(N)$ for CP or $O(N^2)$ for VM.

## §4 3D Gaussian Splatting: explicit differentiable rasterization (**current workhorse**)

**3DGS** (Kerbl 2023 SIGGRAPH **Best Paper**) addressed NeRF's two big pain points: slow rendering and hard editing.

### 4.1　Scene representation

Scene = a set of 3D Gaussians $\{G_i\}$, each:

- **Mean** $\mu_i \in \mathbb{R}^3$ (position)
- **Covariance** $\Sigma_i \in \mathbb{R}^{3\times 3}$ (shape), decomposed as $\Sigma = R S S^\top R^\top$ (rotation $R$ + diagonal scaling $S$)
- **Opacity** $\alpha_i \in [0, 1]$
- **Color** $c_i(\mathbf{d})$ via SH coefficients ($\ell = 3$, 16 coefficients per channel, 48 parameters total)

Why $R S S R^\top$ and not learn $\Sigma$ directly? — to ensure $\Sigma$ is positive definite. Learning $\Sigma$ as a raw matrix under gradient descent escapes the positive-semidefinite cone; the decomposition only requires $R$ to be orthogonal (parameterized via quaternion $q$) and $S$ to be positive (parameterized via $\exp(s)$), so it is satisfied automatically.

### 4.2　3D → 2D projection Jacobian (**L3 must-derive**)

To splat a 3D Gaussian to the screen for rasterization, the 3D covariance $\Sigma$ must be projected to a 2D covariance $\Sigma'$.

**Step 1**: World → Camera: rigid transform $W \in SE(3)$. $\Sigma_\text{cam} = W \Sigma W^\top$ (where $W$ takes the rotation part; translation does not affect covariance).

**Step 2**: Camera → Screen: perspective projection is **nonlinear**:

$$\pi(\mathbf{x}) = \begin{pmatrix} f_x\,x/z \\ f_y\,y/z \end{pmatrix}$$

The covariance of a nonlinear map is approximated via first-order Taylor. Compute the Jacobian at the mean $\mu_\text{cam} = (x, y, z)$:

$$J = \frac{\partial \pi}{\partial \mathbf{x}}\bigg|_{\mu_\text{cam}} = \begin{pmatrix}\dfrac{f_x}{z} & 0 & -\dfrac{f_x\,x}{z^2}\\[2pt] 0 & \dfrac{f_y}{z} & -\dfrac{f_y\,y}{z^2}\end{pmatrix} \in \mathbb{R}^{2\times 3}$$

**Step 3**: 2D covariance (**core formula**):

$$\boxed{\;\Sigma' = J\,W\,\Sigma\,W^\top\,J^\top \in \mathbb{R}^{2\times 2}\;}$$

Derivation: if $\mathbf{x} \sim \mathcal{N}(\mu, \Sigma)$, then first-order $\pi(\mathbf{x}) \approx \pi(\mu) + J(\mathbf{x} - \mu)$, so $\text{Cov}[\pi(\mathbf{x})] \approx J\,\Sigma_\text{cam}\,J^\top = J W \Sigma W^\top J^\top$. This is the classic corollary of EWA splatting (Zwicker 2001).

### 4.3　Differentiable rasterization: tile-based front-to-back alpha-blending

The color at pixel $\mathbf{p}$:

$$C(\mathbf{p}) = \sum_{i \in \mathcal{N}(\mathbf{p})}\,c_i\,\alpha_i\,G_i'(\mathbf{p}) \prod_{j < i}\big(1 - \alpha_j\,G_j'(\mathbf{p})\big)$$

where $G_i'(\mathbf{p}) = \exp\!\big(-\tfrac{1}{2}(\mathbf{p} - \mu_i')^\top \Sigma_i'^{-1} (\mathbf{p} - \mu_i')\big)$ is the value of the 2D Gaussian at the pixel, and $\mathcal{N}(\mathbf{p})$ are the Gaussians covering $\mathbf{p}$ sorted by depth.

**Key engineering**:

1. **Tile partitioning**: split the screen into $16\times 16$ tiles, sort Gaussians by depth within each tile, render in parallel
2. **GPU sort**: radix sort, with composite key `(tile_id, depth)`
3. **Front-to-back early stop**: exit when accumulated $\prod(1 - \alpha G') < 10^{-4}$
4. **CUDA kernel**: the authors release `diff-gaussian-rasterization`; both forward and backward are manual derivatives

### 4.4　3DGS forward pass (PyTorch reference implementation)

```python
def quat_to_rot(q: torch.Tensor) -> torch.Tensor:
    """ q: [N, 4] (w, x, y, z) already normalized; returns R: [N, 3, 3] """
    w, x, y, z = q.unbind(-1)
    R = torch.stack([
        1 - 2*(y*y + z*z),   2*(x*y - w*z),     2*(x*z + w*y),
        2*(x*y + w*z),       1 - 2*(x*x + z*z), 2*(y*z - w*x),
        2*(x*z - w*y),       2*(y*z + w*x),     1 - 2*(x*x + y*y),
    ], dim=-1).reshape(-1, 3, 3)
    return R

def gaussian_splat_forward(
    means3D: torch.Tensor,        # [N, 3]  Gaussian centers (world)
    scales: torch.Tensor,          # [N, 3]  log-scale (take exp for true scale)
    quats: torch.Tensor,           # [N, 4]  quaternion (will be normalized)
    opacities: torch.Tensor,       # [N, 1]  σ(logit) → α
    colors: torch.Tensor,          # [N, 3]  (simplified to RGB here, no SH expansion)
    viewmat: torch.Tensor,         # [4, 4]  world→camera
    K: torch.Tensor,               # [3, 3]  intrinsics (fx, fy, cx, cy)
    H: int, W: int,
):
    """ Pedagogical forward: no tile sort / CUDA, just shows the math.
        Real production uses gsplat / diff-gaussian-rasterization. """
    N = means3D.shape[0]
    device = means3D.device

    # --- 1. World → Camera ---
    homo = torch.cat([means3D, torch.ones(N, 1, device=device)], dim=-1)
    mu_cam = (homo @ viewmat.T)[:, :3]                          # [N, 3]
    z = mu_cam[:, 2].clamp(min=1e-4)                            # guard against div-by-zero

    # --- 2. Covariance (3D) ---
    q = quats / quats.norm(dim=-1, keepdim=True)
    R = quat_to_rot(q)                                          # [N, 3, 3]
    S = torch.diag_embed(torch.exp(scales))                     # [N, 3, 3]
    cov3D = R @ S @ S.transpose(-1, -2) @ R.transpose(-1, -2)   # [N, 3, 3]

    # Apply the World→Cam rotation part W_rot (3x3) to the covariance
    W_rot = viewmat[:3, :3]
    cov_cam = W_rot @ cov3D @ W_rot.T                           # [N, 3, 3]

    # --- 3. Projection Jacobian J (2x3) ---
    fx, fy = K[0, 0], K[1, 1]
    x_c, y_c, z_c = mu_cam[:, 0], mu_cam[:, 1], z
    J = torch.zeros(N, 2, 3, device=device)
    J[:, 0, 0] = fx / z_c
    J[:, 0, 2] = -fx * x_c / z_c**2
    J[:, 1, 1] = fy / z_c
    J[:, 1, 2] = -fy * y_c / z_c**2

    # --- 4. 2D covariance Σ' = J W Σ W^T J^T ---
    cov2D = J @ cov_cam @ J.transpose(-1, -2)                   # [N, 2, 2]
    cov2D = cov2D + 0.3 * torch.eye(2, device=device)           # low-pass filter (anti-aliasing)

    # --- 5. 2D center (pixel coordinates) ---
    cx, cy = K[0, 2], K[1, 2]
    mu2D = torch.stack([fx * x_c / z_c + cx, fy * y_c / z_c + cy], dim=-1)  # [N, 2]

    # --- 6. Depth sort (front to back) ---
    depth = z_c
    order = depth.argsort()                                     # ascending z
    mu2D, cov2D = mu2D[order], cov2D[order]
    colors_o = colors[order]
    alphas = torch.sigmoid(opacities[order]).squeeze(-1)        # [N]

    # --- 7. Pixel loop (pedagogical full-image loop; real impl uses tile + CUDA) ---
    yy, xx = torch.meshgrid(torch.arange(H, device=device),
                            torch.arange(W, device=device), indexing='ij')
    pix = torch.stack([xx, yy], dim=-1).float()                 # [H, W, 2]

    img = torch.zeros(H, W, 3, device=device)
    T_acc = torch.ones(H, W, device=device)
    inv_cov = torch.linalg.inv(cov2D)                           # [N, 2, 2]

    for i in range(mu2D.shape[0]):
        diff = pix - mu2D[i]                                    # [H, W, 2]
        # 2D Gaussian: exp(-0.5 (diff)^T Σ^-1 (diff))
        G = torch.exp(-0.5 * (diff @ inv_cov[i] * diff).sum(-1))  # [H, W]
        contrib = alphas[i] * G                                 # [H, W]
        contrib = contrib.clamp(max=0.99)                       # numerical safety
        img = img + (T_acc * contrib).unsqueeze(-1) * colors_o[i]
        T_acc = T_acc * (1 - contrib)
        if (T_acc < 1e-4).all():                                # early stop
            break

    return img
```

> ⚠️ **Pedagogical vs production gap** — the above is $O(N \cdot HW)$; 30k Gaussians + an 800×800 image would already take seconds. Real `gsplat` is (a) tile-based: each tile only processes Gaussians "touching" it; (b) GPU radix sort with composite keys; (c) the whole forward / backward is manual CUDA, achieving ≤ 10ms at 1080p.

### 4.5　Adaptive density control (**frequent interview topic**)

3DGS initializes from sparse SfM (COLMAP) point clouds, but training must "spread" more Gaussians.

| Trigger | Action | Intuition |
| --- | --- | --- |
| **Large gradient + small scale** | **clone** (duplicate in place: new_xyz = original xyz, no explicit offset along the gradient; the two overlapping Gaussians separate naturally through subsequent independent gradient descent) | "Under-reconstruction" — this region lacks detail |
| **Large gradient + large scale** | **split** (the step that actually samples a new position: draw via torch.normal with the original Gaussian's scale as the std + rotation, then divide scale by 1.6, splitting into 2 smaller Gaussians) | "Over-reconstruction" — a big Gaussian covers what it shouldn't |
| **Opacity near 0** | **prune** (delete) | This Gaussian contributes nothing, wasting VRAM |
| **Every 3k iter** | reset opacities to 0.01 (0.005 is the min_opacity threshold used by prune below — not the same number) | Force the model to relearn opacity, prevent floaters |

Heuristic conditions: `gradient norm > τ_pos` (e.g. $2 \times 10^{-4}$), `scale > τ_scale` (1% of scene scale).

```python
def densify_and_prune(gaussians, grad_thresh=2e-4, scale_thresh=0.01,
                      max_screen_size=None):
    """ Pedagogical densify decisions (simplified; real gsplat also has screen-size triggers).
        Assume gaussians exposes the following 1D fields (N = current Gaussian count):
          xyz_grad_accum: [N]  ‖accumulated xyz gradient norm‖
          denom:          [N]  accumulation count (avoid /0)
          scales:         [N, 3]  log-scale
          opacities:      [N]  ∈ (0, 1) after sigmoid
          screen_size:    [N]  last-render screen-projected size (optional)
    """
    grad_norm  = gaussians.xyz_grad_accum / gaussians.denom.clamp(min=1)      # [N]
    mean_scale = gaussians.scales.exp().max(dim=-1).values                    # [N]

    # ⚠️ Compute BOTH the clone / split masks on the original N Gaussians first,
    #    before any append/delete — otherwise once clone_at appends and the array
    #    grows, the N-length split_mask is misaligned with the new array
    #    (mask-length mismatch / acting on the wrong objects).
    clone_mask = (grad_norm > grad_thresh) & (mean_scale <= scale_thresh)     # [N]
    split_mask = (grad_norm > grad_thresh) & (mean_scale >  scale_thresh)     # [N]

    # CLONE: high gradient + small scale — duplicate in place (new_xyz = original xyz, no
    # positional offset); keep original (append at the end), and the two overlapping
    # Gaussians separate naturally through subsequent independent gradient descent
    gaussians.clone_at(clone_mask)

    # SPLIT: high gradient + large scale — this is the step that actually samples a new
    # position: draw via torch.normal with the original Gaussian's scale as the std +
    # rotation, then divide scale by 1.6, splitting into 2 children, removing original at the end
    # split_at applies explicitly to the original N only: pass original_n, do not rely on implicit alignment with the post-append array length
    gaussians.split_at(split_mask, n=2, scale_div=1.6, original_n=split_mask.shape[0])

    # PRUNE: low opacity / too large on screen / already marked by split
    # ⚠️ Note: new Gaussians from clone are appended to the end; length has changed. The mask only applies to the original N.
    prune_mask = (gaussians.opacities[:split_mask.shape[0]] < 0.005) | split_mask
    if max_screen_size is not None:
        prune_mask = prune_mask | (gaussians.screen_size[:split_mask.shape[0]] > max_screen_size)
    gaussians.remove_original(prune_mask)   # delete only the originals flagged among the first N

    gaussians.reset_grad_accum()
    return gaussians
```

> 💡 **Typical hyperparameters** — Kerbl 2023 paper: densify every 100 iters; max Gaussian count 5e6; 30k iters total; from ~30 minutes to a few hours.

### 4.6　2DGS / Surfels (surface-aligned)

3DGS ellipsoids are not surface-aware; mesh extraction needs SuGaR / GSDF post-processing. **2DGS** (Huang 2024 SIGGRAPH) **degenerates the 3D ellipsoid into a 2D disk** (one axis = 0), directly aligning with the surface, which works better with normal/depth supervision and produces clearly higher-quality meshes.

### 4.7　Dynamic 4DGS

**Dynamic 3DGS** (Luiten 2024 3DV, arXiv:2308.09713) uses a set of **persistent Gaussians**: color / opacity / size stay fixed across frames, only the position $\mu(t)$ and rotation move over time, with a **local-rigidity** regularizer constraining neighborhood rigidity; **4DGS** (Wu 2024 CVPR / Yang 2024 ICLR) writes $\mu(t), \Sigma(t)$ as functions of time (MLP or spline); **SC-GS** (Huang 2024) drives dense Gaussians from sparse control points (analogous to LBS).

## §5 Mesh extraction: Marching Cubes / DMTet

After NeRF / 3DGS reconstruction, downstream (simulator, AR, 3D printing) often needs a mesh.

### 5.1　Marching Cubes (**classic must-know**)

Input: 3D scalar field $f(\mathbf{x})$ (density / SDF) + threshold $\tau$. Output: triangle mesh of the level set $\{f = \tau\}$.

**Algorithm skeleton**:

1. Voxelize space (8 corners per voxel)
2. For each voxel, **binarize the 8 corners** ($f > \tau$ as 1, else 0) → 256 possible configurations
3. Look up table: each configuration has predefined isosurface triangle patches + vertex positions on edges
4. **Linear interpolation** for precise vertices: between edge endpoints $\mathbf{a}, \mathbf{b}$, interpolate $t = (\tau - f(\mathbf{a})) / (f(\mathbf{b}) - f(\mathbf{a}))$, vertex $= \mathbf{a} + t(\mathbf{b} - \mathbf{a})$
5. Merge all voxel triangles → full mesh

```python
def marching_cubes_sketch(density: torch.Tensor, threshold: float):
    """ Real implementations use mcubes / scikit-image / pytorch3d; this shows the idea """
    from skimage.measure import marching_cubes
    # density: [Nx, Ny, Nz]; detach() cuts the autograd graph, cpu().numpy() moves to host
    verts, faces, normals, _ = marching_cubes(
        density.detach().cpu().numpy(),
        level=threshold,
        spacing=(1.0, 1.0, 1.0),
        gradient_direction='descent',  # normal direction; descent = surface faces low density
    )
    return verts, faces, normals
```

> ⚠️ **NeRF mesh-extraction footgun** — vanilla NeRF has no notion of "surface"; the threshold $\tau$ is hard to pick when extracting mesh, and floaters get extracted too. **Use NeuS / VolSDF for stable mesh extraction** (the 0 level set of the SDF is well defined).

### 5.2　Differentiable: DMTet / FlexiCubes (end-to-end mesh learning)

Marching Cubes is not differentiable (the lookup table is discrete).

- **DMTet** (Shen 2021 NeurIPS / Munkberg 2022 CVPR): uses a **deformable tetrahedral grid**, each tetrahedron has 4 vertex SDFs + position offsets that are differentiable. **Marching Tetrahedra** replaces MC; topology is determined by SDF signs and geometry by vertex positions: within a region of fixed topology, geometry (vertex positions) is **differentiable** with respect to the SDF values and offsets; but at the instant a vertex SDF crosses zero and triggers a switch in that tetrahedron's look-up case, the topology itself is **not differentiable** (a measure-zero set, akin to the kink in ReLU at 0) — it should not be described unconditionally as "fully differentiable."
- **FlexiCubes** (Shen 2023 SIGGRAPH): generalizes dual marching cubes by introducing extra learnable parameters (dual vertex offsets / interpolation weights) to fix quality artifacts.

**Typical use**: Magic3D and Fantasia3D after DreamFusion use DMTet to learn meshes + textures under SDS supervision.

## §6 SDS Loss: supervising 3D with 2D diffusion (DreamFusion family)

### 6.1　Problem setup

We want to generate 3D assets but **have no 3D training data** — 3D data is scarce (ShapeNet ~50k objects, Objaverse-XL 10M but uneven quality). **Pretrained 2D diffusion** (Stable Diffusion, Imagen) is abundant. Can we use 2D diffusion as a teacher to supervise 3D?

**DreamFusion** (Poole et al. 2022 arXiv → **ICLR 2023 Outstanding Paper**) proposed **Score Distillation Sampling (SDS)**.

### 6.2　Setup

- 3D representation $\theta$ (NeRF parameters / DMTet vertices / 3DGS point cloud)
- Differentiable renderer $g(\theta, \pi) \to x$, where $x$ is an image ($\pi$ is the camera viewpoint)
- Pretrained 2D diffusion $\epsilon_\phi(x_t; y, t)$ ($y$ is the text prompt)

**Goal**: make $g(\theta, \pi)$ look like a "photo of $y$", i.e. $g(\theta, \pi)$ lies on the data manifold learned by the diffusion model.

### 6.3　SDS gradient derivation (**L3 must-know**)

Intuition: backprop the diffusion training loss into $\theta$. **Naive idea**: treat the rendered image $x = g(\theta, \pi)$ as a training sample and minimize

$$\mathcal{L}_\text{diff}(\theta) = \mathbb{E}_{t, \epsilon}\Big[w(t)\big\|\epsilon_\phi(x_t; y, t) - \epsilon\big\|^2\Big],\quad x_t = \alpha_t x + \sigma_t \epsilon$$

Take the gradient w.r.t. $\theta$ (chain rule):

$$\nabla_\theta \mathcal{L}_\text{diff} = \mathbb{E}\Big[w(t)\,2\big(\epsilon_\phi(x_t; y, t) - \epsilon\big)\,\underbrace{\frac{\partial \epsilon_\phi(x_t;y,t)}{\partial x_t}}_{\text{U-Net Jacobian}}\,\alpha_t\,\underbrace{\frac{\partial x}{\partial \theta}}_{\text{renderer Jacobian}}\Big]$$

**Problem**: the U-Net Jacobian $\partial \epsilon_\phi / \partial x_t$ is expensive to compute and numerically poor (the diffusion model is large and not trained for second-order stability).

**SDS trick: drop the U-Net Jacobian entirely**, giving

$$\boxed{\;\nabla_\theta \mathcal{L}_\text{SDS} \;=\; \mathbb{E}_{t, \epsilon}\Big[w(t)\,\big(\epsilon_\phi(x_t; y, t) - \epsilon\big)\,\frac{\partial x}{\partial \theta}\Big]\;}$$

(The original DreamFusion paper writes it as $\partial L/\partial \theta$; $\alpha_t$ and the constant 2 are absorbed into $w(t)$.)

### 6.4　Why does dropping the Jacobian still work?

**First explanation (original DreamFusion, score view)**: $\epsilon_\phi(x_t; y, t)/\sigma_t \approx -\nabla_{x_t}\log p_\phi(x_t|y)$ (score). The SDS gradient = `(predicted score - noise) × renderer Jacobian`, i.e. it pushes the rendered image toward high-probability regions.

**Second explanation (mode-seeking)**: SDS is equivalent to a mode-seeking form of $\mathbb{E}_t[D_\text{KL}(q(x_t|\theta) \,\|\, p_\phi(x_t|y))]$: drive toward high-probability regions of $p_\phi(\cdot|y)$.

### 6.5　SDS side effects: over-saturation / mode collapse / Janus

- **Over-saturation**: oversaturated colors, excessive contrast ("plastic-y look")
- **Over-smoothing**: blurred details
- **Mode collapse**: objects converge to "canonical" single forms
- **Janus problem**: 3D objects grow a "front face" in every view (faces on the back of heads; animals with heads on both sides)

**Root cause**: SDS is equivalent to a mode-seeking KL, and **only large CFG (DreamFusion default 100) can escape the mean-mode**. CFG=100 sharpens the distribution to the extreme → over-saturation.

> ⚠️ **Interview high score point** — the SDS formula "drops the Jacobian" to save compute, but **at the cost** of implicitly becoming a mode-seeking KL that needs huge CFG to escape mean-seeking blur; huge CFG in turn causes over-saturation. This is an **information-theoretic trade-off**: simulation-free + computationally cheap = mode-seeking artifact.

### 6.6　SDS code (core 30 lines)

```python
def sds_loss(
    renderer,                 # θ → x (B, 3, H, W)
    theta,                    # 3D parameters (NeRF / 3DGS / DMTet)
    prompt_emb,               # text embedding (cond) [B, L, D]
    uncond_emb,               # text embedding (uncond / null) [B, L, D]
    unet,                     # frozen 2D diffusion U-Net (e.g. SD); returns noise pred Tensor
    alpha_cumprod: torch.Tensor,  # [T_max] precomputed bar-alpha schedule
    cfg_scale: float = 100.0,
    t_range: tuple = (0.02, 0.98),
):
    """ Score Distillation Sampling loss (DreamFusion).
        Convention: unet(x_t, t, encoder_hidden_states=emb) -> [B, 3, H, W] noise pred.
        If using diffusers UNet2DConditionModel, wrap to take .sample.
        Returns a grad surrogate, can be backward'd directly. """
    x = renderer(theta)                                  # [B, 3, H, W]
    B = x.shape[0]
    device, dtype = x.device, x.dtype
    T_max = alpha_cumprod.shape[0]                       # usually 1000

    # 1. Sample t and noise, forward add noise
    t = torch.randint(int(t_range[0] * T_max), int(t_range[1] * T_max),
                      (B,), device=device)
    noise = torch.randn_like(x)
    abar = alpha_cumprod.to(device=device, dtype=dtype)[t].view(B, 1, 1, 1)
    x_t = abar.sqrt() * x + (1 - abar).sqrt() * noise

    # 2. U-Net predicts noise (cond / uncond), CFG combination; key: no autograd through diffusion
    with torch.no_grad():
        eps_uncond = unet(x_t, t, encoder_hidden_states=uncond_emb)
        eps_cond   = unet(x_t, t, encoder_hidden_states=prompt_emb)
        eps_pred = eps_uncond + cfg_scale * (eps_cond - eps_uncond)

    # 3. SDS gradient: w(t)(ε_pred - ε) · ∂x/∂θ; w(t) = σ_t² is a common choice
    grad = ((1 - abar) * (eps_pred - noise)).detach()
    # backward of (grad · x) gives grad · ∂x/∂θ
    return (grad * x).sum() / B
```

> 💡 **Training loop** — each iter, randomly sample viewpoint $\pi$, render $x$, compute SDS loss, backprop to $\theta$. For NeRF representations, train 10k-100k steps (hours on GPU); for 3DGS representations (GaussianDreamer / DreamGaussian), minutes to one hour.

### 6.7　VSD: variational SDS (**ProlificDreamer**, NeurIPS 2023 Spotlight)

**VSD** (Wang 2023 NeurIPS) views SDS as a special case of "point estimation for a single $\theta$" and generalizes to **variational inference over a distribution $\mu(\theta)$ on $\theta$**.

#### Setup
- Treat the 3D parameters $\theta$ as a latent random variable with distribution $\mu(\theta)$
- Goal: align the rendered-image **distribution** with the diffusion-learned prior **distribution** (not mode alignment)

#### Objective and gradient

ProlificDreamer writes the objective as a KL:

$$\min_{\mu}\; D_\text{KL}\!\Big(q_\mu^t(x_t|y)\;\Big\|\;p_\phi^t(x_t|y)\Big),\quad t\sim\mathcal{U}[0,1]$$

where $q_\mu^t$ is the distribution induced by "rendering from $\theta\sim\mu$ + adding noise to time $t$". The **variational gradient** (Wang et al. 2023, Theorem 2, abbreviated) gives the update direction in $\theta$ as a **relative score**:

$$\boxed{\;\nabla_\theta \mathcal{L}_\text{VSD} \;=\; \mathbb{E}_{t,\epsilon}\Big[\,w(t)\,\big(\epsilon_\phi(x_t;y,t) \;-\; \epsilon_\psi(x_t;y,t,\pi)\big)\,\frac{\partial x}{\partial \theta}\,\Big]\;}$$

Compared to SDS: replace the raw noise $\epsilon$ with an **auxiliary score** $\epsilon_\psi$. $\epsilon_\psi$ is a **LoRA-finetuned** score network that online-minimizes a score-matching loss to track the score of the current $q_\mu^t$; it is the necessary term corresponding to the $q$ distribution's own score (an entropy term) that arises from differentiating the KL objective $D_\text{KL}(q_\mu^t\|p_\phi^t)$ with respect to $\theta$ — it is **not** a zero-mean, variance-only-reducing control variate. Dropping it degrades the gradient objective from "distribution matching" back to SDS's mode-seeking (it changes the expected gradient direction, not just the noise), so it cannot be equated with a value baseline in RL actor-critic, which only reduces variance without changing the expected gradient. **Note**: the above is the gradient form, not a squared-loss form; the paper does not have an implementable form "write a scalar loss $\|\epsilon_\phi-\epsilon_\psi\|^2$ and then differentiate" — $\epsilon_\psi$ depends on $\mu$, which would drop the key term of the KL.

#### Intuition for why over-saturation is mitigated
- SDS: pushes the rendered $x$ toward modes of the prior $p_\phi(\cdot|y)$ (mean-mode → needs CFG=100 → over-saturation)
- VSD: the auxiliary score tracks the score of the current rendered distribution; the update direction = from "where I am now" to "where the prior is" (**relative gradient**), without needing huge CFG. CFG can drop to 7.5 (the standard diffusion default), avoiding extreme sharpening
- Empirically: VSD has more natural colors and more complex geometry, and can maintain multiple modes simultaneously (ProlificDreamer reports 50k-step training yielding photorealistic Buddha statues, etc.)

> ✅ **Key insight VSD vs SDS** — SDS is "**single-point + mode-seeking**"; VSD is "**particle / variational + relative score**". The latter's $\epsilon_\psi$ is the necessary term corresponding to the $q$ distribution's own score from differentiating the KL objective with respect to $\theta$; dropping it degrades the gradient objective back to SDS's mode-seeking — it is not a value baseline in RL actor-critic that only reduces variance without changing the expected gradient.

### 6.8　SDS derivative family: mesh / 3DGS + SDS

| Method | Representation / Stages | Key point |
| --- | --- | --- |
| **DreamFusion** | NeRF + SDS @ low-res | Original, hard to extract mesh |
| **Magic3D** (Lin 2023 CVPR) | Instant-NGP @ 64px → DMTet + SDS @ 512px | **Two-stage**: coarse structure → high-resolution end-to-end mesh |
| **Fantasia3D** (Chen 2023 ICCV) | DMTet geometry + PBR material | normal-as-input + physical material BRDF |
| **DreamGaussian** (Tang 2024 ICLR) | 3DGS + SDS, ~2 min / object | GPU speed advantage; mesh export + UV-Net texturing |
| **GaussianDreamer** (Yi 2024 CVPR) | Point-E / Shap-E init → 3DGS + SDS | Alleviates from-scratch geometric chaos |

## §7 Single-image / Few-view 3D generation

A more practical setting: **given one image, generate 3D**.

### 7.1　Zero-1-to-3 paradigm (novel view via diffusion)

**Zero-1-to-3** (Liu 2023 ICCV): finetune Stable Diffusion on Objaverse so it accepts (input view, target camera) → output novel view.

- Input: single image $x$ + relative camera pose $\Delta R, \Delta T$
- Diffusion conditioning: image embedding (CLIP) + camera embedding (sinusoidal)
- Output: the image from the $\Delta R, \Delta T$ viewpoint

**Usage**: given one input view, sample 16-32 novel views, then reconstruct via NeRF / 3DGS.

**Derivatives**:
- **Zero-1-to-3++** (Shi 2023): fixed generation of 6 anchor views (azimuth uniformly spaced every 60° at 30°/90°/150°/210°/270°/330°, elevation alternating between -10° and 20° — i.e. interleaved elevation + uniform azimuth, not "one north-pole view + four eye-level views + one top view"), reducing randomness
- **SyncDreamer** (Liu 2024 ICLR): **joint** prediction of multiple views in latent space (cross-attention lets views see each other), ensuring 3D consistency
- **MVDream** (Shi 2024 ICLR): text-to-multi-view, generates 4 views simultaneously; followed by SDS refinement

### 7.2　One-2-3-45 / InstantMesh / TripoSR / Stable Fast 3D

| Method | Input | Output | Speed | Key |
| --- | --- | --- | --- | --- |
| **One-2-3-45** (Liu 2023 NeurIPS) | Single image | mesh | 45 s | Zero-1-to-3 → SparseNeuS |
| **One-2-3-45++** (Liu 2024) | Single image | mesh | 60 s | Multi-view + SDF |
| **TripoSR** (Tochilkin 2024, Stability+Tripo) | Single image | NeRF/mesh | 0.5-2 s | LRM-style (Large Reconstruction Model) transformer |
| **InstantMesh** (Xu 2024, arXiv:2404.07191) | Single image | mesh | ~10 s | Zero-1-to-3++ multi-view → sparse-view recon transformer |
| **Stable Fast 3D** (SF3D, Stability 2024) | Single image | textured mesh | ~0.5 s | TripoSR successor; adds illumination disentangle + UV unwrap |

**LRM (Hong et al. 2023 arXiv → ICLR 2024) setting**: treat the image as tokens + Plucker ray embedding, transformer outputs a NeRF triplane. This is the parent model of TripoSR / InstantMesh.

### 7.3　LRM Triplane representation (**high-frequency interview topic**)

- **Triplane** (Chan 2022 EG3D): 3 axis-aligned 2D planes (XY, YZ, XZ), total $3 \times C \times N \times N$ dim
- Query 3D point $(x, y, z)$: bilinearly interpolate on each plane → **element-wise sum of the three plane features** (EG3D uses sum, not concat) → small MLP → $(\sigma, \mathbf{c})$
- Advantages: less VRAM than a voxel grid ($O(N^2)$ vs $O(N^3)$), denser than a hash grid making it suitable as transformer output
- LRM / TripoSR / InstantMesh all let the transformer directly regress triplane tokens

## §8 3D Foundation Models (the 2024 open-source wave)

### 8.1　Trellis (Microsoft 2024, open source)

**Trellis** (Xiang 2024 arXiv) is the first attempt at "Stable Diffusion for 3D" in the open-source space.

- **Structured Latent (SLAT)**: encode the 3D asset onto a sparse latent grid over voxels — preserving spatial structure (suitable for sparse conv / sparse attention) while being compact (only active voxels store latents)
- **3D VAE**: mesh + texture (derived from signed distance field) → SLAT
- **Flow matching prior**: rectified flow on SLAT, conditioned on text/image
- **Multiple decoders**: decode SLAT into NeRF / 3DGS / mesh representations (same latent, choice of output format)
- **Training data**: a subset of Objaverse-XL + internal high-quality set
- **Effects**: text-to-3D / image-to-3D, seconds to tens of seconds, quality surpassing the SDS family

### 8.2　Hunyuan3D-1 / -2 (Tencent 2024-25, open source)

**Hunyuan3D** follows the **shape-then-texture** two-stage route.

- **Hunyuan3D-1** (Yang 2024 arXiv):
  - Stage 1: text/image → multi-view image (Zero-1-to-3 family)
  - Stage 2: multi-view → 3D mesh (LRM-like reconstructor)
  - Outputs textured mesh in seconds to tens of seconds
- **Hunyuan3D-2** (Tencent 2025, arXiv 2501.12202):
  - **Hunyuan3D-DiT**: geometry-only DiT generating mesh on SDF latents
  - **Hunyuan3D-Paint**: multi-view PBR texture diffusion, UV-space refinement
  - High-quality PBR texture (production-ready for game / VR assets)
- **Open source**: complete weights + inference code on HuggingFace

### 8.3　CLAY (Zhang 2024 SIGGRAPH, arXiv:2406.13897)

- **Multi-resolution VAE + latent DiT**: a 3D shape VAE encodes the mesh into a neural-field latent, then a DiT diffusion runs on that latent (distinct from 3DShape2VecSet, Zhang et al. 2023, arXiv:2301.11445, which is a separate vector-set representation work)
- Large-scale training (Objaverse-XL + internal curated set)
- Output SDF → marching cubes → mesh
- Adds a PBR texture stage (similar to Hunyuan3D-2)

**Rodin** (Microsoft 2023, commercial): early production-grade text-to-3D-avatar system, diffusion on triplane, focused on characters / avatars.

### 8.4　Comparison table

| Method | Representation | Prior | Training scale | Open source |
| --- | --- | --- | --- | --- |
| **Trellis** | Structured Latent (SLAT) + multi decoders | Rectified Flow | Objaverse-XL subset | ✅ |
| **Hunyuan3D-2** | SDF latent (Shape DiT) + UV texture diff | Diffusion | Internal large-scale set | ✅ |
| **CLAY** | Multi-resolution VAE latent (latent DiT) | Diffusion | Objaverse-XL + internal | Partial |
| **Rodin** | Triplane | Diffusion | Commercial internal | ❌ |
| **TripoSR / SF3D** | NeRF/mesh feedforward | No prior, pure regression | Objaverse-class | ✅ |

> 💡 **Architecture-choice intuition** — large scenes / general objects use **Trellis-style SLAT** (preserves spatial structure); high-quality single meshes use **CLAY-style latent DiT** (multi-resolution VAE latent, global attention); fast inference uses **LRM/TripoSR feedforward** (no diffusion, direct regression).

## §9 Complexity / resource comparison

| Method | Training | Inference (one frame) | VRAM (training) | VRAM (model) |
| --- | --- | --- | --- | --- |
| NeRF vanilla | 1-2 days | several seconds | 8 GB | <10 MB MLP |
| Instant-NGP | 5 seconds - 5 min | 30 fps+ | 4-12 GB | 100-500 MB hash |
| 3DGS | 10-30 min | 100 fps+ | 6-24 GB | 100 MB - 1 GB Gaussians |
| 2DGS | Close to 3DGS | Close to 3DGS | Similar | Similar |
| DreamFusion (NeRF+SDS) | 2 hr / object | — | 12 GB | NeRF itself |
| DreamGaussian (3DGS+SDS) | 2 min / object | — | 8-16 GB | — |
| ProlificDreamer (VSD) | 3-6 hr / object | — | 24 GB | — |
| TripoSR feedforward | 50 GPU-days training | 0.5 s (A100) | 6 GB inference | 1.5 GB |
| Trellis | 100+ GPU-days training | several seconds | 16 GB inference | a few GB |
| Hunyuan3D-2 | Training on large cluster | tens of seconds | 24+ GB inference | combination of multiple models |

## §10 Comparison with related methods & Embodied AI applications

### 10.1　Key differences between 3D and 2D generation

| Dimension | 2D generation (Stable Diffusion) | 3D generation |
| --- | --- | --- |
| **Data scale** | LAION-5B 5B images | Objaverse-XL 10M objects (500× smaller) |
| **Data format** | Image (uniform RGB) | mesh / SDF / point cloud / NeRF / 3DGS (**fragmented**) |
| **Training prior** | Train diffusion directly | Distill from 2D diffusion (SDS / Zero-1-to-3) **or** 3D-native diffusion (Trellis / CLAY) |
| **Evaluation** | FID, CLIP score | Chamfer / IoU / PSNR (recon) + perceptual + user study |
| **Downstream** | Output image directly | Output asset → rendering / simulation / editing |

### 10.2　Embodied AI / AR / VR practical routes

| Task | Recommended representation | Key toolchain / constraints |
| --- | --- | --- |
| **Sim2Real assets** | mesh (PBR) | Trellis / Hunyuan3D-2 → IsaacSim / MuJoCo |
| **Large indoor scenes** | 3DGS | COLMAP → 3DGS (chunk-wise with VastGS / CityGS) |
| **NeRF/3DGS as simulator** | NeRF / 3DGS + physics | DreamGaussian-Sim / Splatting Physics |
| **3D affordance / manipulation** | point cloud / 3DGS feature | OpenScene / LERF / RVT / 3D Diffuser Actor |
| **AR object scanning** | 3DGS (realistic lighting + real-time) | mobile compute (PostShot / Luma), pruning / quantization |
| **VR large scenes** | 3DGS (large-scale) | 60 fps stereo + 6DoF |
| **Avatar** | mesh + LBS or 3DGS avatar | real-time expressions / hair |
| **Object insertion** | mesh + PBR | consistent environment lighting (IBL) |

> ⚠️ **Embodied AI interview follow-up example** — "Biggest challenge in making NeRF a physics simulator?" Key points: NeRF is radiance, no mass / friction → physics priors must be added manually; mesh extraction has floaters → collision detection is hard; differentiable but slow backward; **industry mostly uses 3DGS / mesh rather than vanilla NeRF**.

## §11 Engineering practice & common footguns

### 11.1　COLMAP / SfM preprocessing (essential for reconstruction)

Input multi-view → output intrinsics $K$ + extrinsics $\{R_i, t_i\}$ + sparse point cloud; standard pipeline SIFT → matching → incremental SfM → bundle adjustment. **Common pitfalls**: SfM fails on texture-less / specular objects; dynamic objects pollute extrinsics.

### 11.2　Numerical stability (general NeRF/3DGS)

| Issue | Symptom | Fix |
| --- | --- | --- |
| Sigma blowup | Floaters fill the space | Use softplus or truncated $\sigma$; occupancy grid skip |
| Alpha saturation | 1-α underflow → T all 0 | `(1-α).clamp(min=1e-10)` or log-space cumprod |
| Gaussian degeneration | Extremely small scale / extreme anisotropy | Clamp scale lower bound; regularize anisotropy |
| Densify explosion | Gaussian count skyrockets to memory limit | Add max gaussian count; periodic prune; reset opacity |
| SDS Janus | Faces / heads in multiple views | Add view-conditioning ("front view" / "back view"); MVDream |
| SDS over-saturation | Saturated colors | Lower CFG; switch to VSD; or negative prompt |

### 11.3　Multi-machine distributed & evaluation metrics

**Distributed**: NeRF / Instant-NGP / 3DGS are single-GPU standard; large 3DGS scenes use chunk-wise (VastGaussian, CityGaussian); SDS/VSD runs 2 SD forwards per iter, 8×A100 gives significant speedup; Trellis / Hunyuan3D training is large-scale multi-node DDP.

| Evaluation metric | Use | Algorithm |
| --- | --- | --- |
| **PSNR / SSIM / LPIPS** | View synthesis (reconstruction) | Compare with real views |
| **Chamfer Distance** | Mesh geometry | Average nearest-neighbor distance between two point clouds |
| **F-Score (3D)** | Mesh / point | Precision + recall under threshold |
| **CLIP Score / CLIP-R-Prec** | Text-to-3D alignment | Render → CLIP similarity / distinguish distractor prompts |
| **User study** | Final quality | MTurk / lab-internal |

## §12 25 frequently-asked interview questions

Sorted into 3 tiers by difficulty (L1 must-know / L2 advanced / L3 top labs). Each question links to answer points + footguns.

### L1 must-know (asked at every 3D / vision role)

<details>

<summary>Q1. NeRF volume-rendering equation?</summary>

- $C(\mathbf{r}) = \int T(t)\sigma(\mathbf{r}(t))\mathbf{c}(\mathbf{r}(t),\mathbf{d})dt$

- $T(t) = \exp(-\int_{t_n}^t\sigma\,ds)$ is the transmittance

- Discretization → $\alpha$-compositing: $C \approx \sum T_i\alpha_i \mathbf{c}_i$, $\alpha_i = 1 - e^{-\sigma_i\delta_i}$

Writing only $\sum \alpha_i \mathbf{c}_i$ misses $T_i$; or writing $\alpha_i$ as $\sigma_i\delta_i$ (first-order approximation, strictly wrong).

</details>

<details>

<summary>Q2. Why does NeRF need positional encoding?</summary>

- MLPs default to a low-frequency bias (NTK analysis)

- $\gamma(p) = (\sin 2^k\pi p, \cos 2^k\pi p)_{k=0}^{L-1}$ provides a high-frequency basis

- Learning $(x,y,z) \to (\sigma,\mathbf{c})$ directly produces blur; adding PE recovers high-frequency detail

Thinking PE just "adds position to the MLP" (actually it adds the spatial frequency spectrum); or flipping the frequency levels for $\mathbf{x}$ vs $\mathbf{d}$ ($L=10$ vs $L=4$).

</details>

<details>

<summary>Q3. What is NeRF's hierarchical sampling?</summary>

- Two networks: coarse + fine

- Coarse samples 64 points uniformly, renders to get weights $w_i = T_i\alpha_i$

- Normalize $w$ to a PDF, importance-sample 128 fine points (dense sampling near the surface)

- Loss supervises both networks

Saying "just sample more densely once" misses the importance-sampling core.

</details>

<details>

<summary>Q4. Why is Instant-NGP roughly 4+ orders of magnitude faster than NeRF?</summary>

- **Hash grid replaces dense grid**: fixed-size $T$ hash table, cache-friendly

- **Tiny MLP** (2 layers hidden 64) replaces large MLP (NeRF 8 layers 256)

- **Multi-resolution cascade** + **occupancy grid** skips empty-region sampling

- **CUDA fused kernels** (tiny-cuda-nn)

Only saying "uses a hash" misses the combined contribution of multi-resolution + tiny-MLP + occupancy skip.

</details>

<details>

<summary>Q5. How is a "Gaussian" in 3DGS defined?</summary>

- Each Gaussian $G_i = (\mu_i, \Sigma_i, \alpha_i, c_i(\mathbf{d}))$

- $\mu \in \mathbb{R}^3$ position, $\Sigma \in \mathbb{R}^{3\times 3}$ covariance

- $\Sigma = R S S^\top R^\top$ decomposition ($R$ via quaternion, $S$ via diagonal + $\exp$), ensures positive semi-definite

- $c(\mathbf{d})$ via SH coefficients ($\ell = 3$, 48 parameters)

Just saying "Gaussian distribution" misses the covariance-parameterization trick + SH color.

</details>

<details>

<summary>Q6. How is 3DGS rendered?</summary>

- Project 3D Gaussians to 2D ($\Sigma' = JW\Sigma W^\top J^\top$)

- Sort by depth

- Front-to-back alpha-blending (same origin as NeRF $\alpha$-compositing)

- Actually tile-based + CUDA radix sort

Just saying "rasterization" without mentioning the projection Jacobian / sorting / alpha-blend.

</details>

<details>

<summary>Q7. How is 3DGS densification done?</summary>

- High gradient + small scale → **clone** (under-reconstruction)

- High gradient + large scale → **split** (over-reconstruction)

- Low opacity or excessive screen-size → **prune**

- Periodic opacity reset to prevent floaters

Flipping clone and split; or forgetting the reset step.

</details>

<details>

<summary>Q8. NeRF vs 3DGS comparison?</summary>

- **NeRF**: implicit (MLP), slow rendering (ray march), hard editing

- **3DGS**: explicit (point cloud), fast rendering (rasterize), easy editing

- **Quality**: 3DGS PSNR is usually ≥ NeRF; NeRF is better on volumetric effects (smoke / translucency)

- **Industry trend**: 3DGS dominant, NeRF research-only

Treating them as incomparable different things — actually both are volumetric scene reps; 3DGS is the explicit version of NeRF.

</details>

<details>

<summary>Q9. What is Marching Cubes?</summary>

- Input: 3D scalar field + threshold; output: triangle mesh

- Each voxel binarizes the 8 corners (above/below threshold) → 256 lookup table

- Linear interpolation on edges defines vertex positions

- Not differentiable (discrete lookup)

Saying "find a contour" — MC is 3D; contours belong to 2D Marching Squares.

</details>

<details>

<summary>Q10. What is SDS roughly?</summary>

- Supervise a 3D representation using pretrained 2D diffusion (Stable Diffusion)

- Render $x = g(\theta, \pi)$, add noise to $x_t$, ask diffusion "is this a photo of $y$?"

- gradient $\propto (\epsilon_\phi(x_t; y) - \epsilon)\cdot \partial x/\partial \theta$

- Proposed by DreamFusion (Poole et al. 2022 arXiv → ICLR 2023 Outstanding Paper)

Just saying "use SD to train NeRF" misses the special form of the SDS gradient (drop the U-Net Jacobian).

</details>

### L2 advanced (research-oriented roles)

<details>

<summary>Q11. Derive NeRF's continuous integration → discrete $\alpha$-compositing.</summary>

- $T$ satisfies $dT/dt = -\sigma T$; on a segment with constant $\sigma$ → $T(t_{i+1})/T(t_i) = e^{-\sigma_i\delta_i}$

- Within-segment color contribution $\int_0^{\delta_i} T_i e^{-\sigma_i s}\sigma_i \mathbf{c}_i\,ds = T_i\mathbf{c}_i(1 - e^{-\sigma_i\delta_i})$

- Let $\alpha_i = 1 - e^{-\sigma_i\delta_i}$, then $C \approx \sum T_i\alpha_i \mathbf{c}_i$, $T_i = \prod_{j<i}(1 - \alpha_j)$

Writing $\alpha_i$ as $\sigma_i\delta_i$ instead of $1 - e^{-\sigma_i\delta_i}$; or skipping the ODE solution.

</details>

<details>

<summary>Q12. Derive the 3D→2D projection Jacobian for 3DGS.</summary>

- Perspective projection $\pi(\mathbf{x}) = (f_x x/z, f_y y/z)$ is nonlinear

- First-order Taylor: $\pi(\mathbf{x}) \approx \pi(\mu) + J(\mathbf{x}-\mu)$

- $J = \partial\pi/\partial\mathbf{x}|_\mu = \begin{pmatrix} f_x/z & 0 & -f_x x/z^2 \\ 0 & f_y/z & -f_y y/z^2 \end{pmatrix}$

- $\Sigma' = JW\Sigma W^\top J^\top$ ($W$ is the world→cam rotation)

Plugging into the "covariance projection" formula without deriving; or forgetting the $W$ step (World→Cam rotation).

</details>

<details>

<summary>Q13. How are Instant-NGP hash collisions disambiguated?</summary>

- **Multi-resolution redundancy**: coarse level $N_\ell^d \le T$ has no collision; only fine levels collide; the MLP can infer from coarse + fine jointly

- **Sparse activation**: effective supervision concentrates near surfaces; colliding entries in empty regions with non-zero density still get gradient (pushing $\sigma$ toward 0) until it decays and the occupancy grid skips those samples

- **MLP post-processing**: learns nonlinear fusion over the $L\times F$ concatenated features, can disambiguate

- No explicit collision resolution; relies on "lazy resolution by sparsity + redundancy"

Thinking there is hash chaining or similar traditional disambiguation — actually it's data-driven implicit disambiguation.

</details>

<details>

<summary>Q14. Which Jacobian does SDS gradient drop? Why?</summary>

- Naive diffusion training grad: $(\epsilon_\phi - \epsilon)\cdot \partial \epsilon_\phi/\partial x_t \cdot \alpha_t \cdot \partial x/\partial \theta$

- SDS drops the $\partial \epsilon_\phi/\partial x_t$ **U-Net Jacobian**

- Intuition: (1) expensive; (2) U-Net is not trained for second-order stability → noisy Jacobian

- Cost: SDS becomes mode-seeking KL, requires huge CFG (100) to escape the mean-mode → over-saturation

Saying only "for simplification" without the consequences. Or not realizing that mode-seeking is determined by the KL direction.

</details>

<details>

<summary>Q15. How does VSD alleviate SDS over-saturation?</summary>

- SDS: pulls toward the modes of prior $p_\phi$; needs CFG=100 → over-saturation

- **VSD**: treats the 3D parameters $\theta$ as a random variable $\mu(\theta)$, minimizes KL(rendered dist || prior)

- Introduces an **auxiliary score** $\epsilon_\psi$ (LoRA-finetuned SD) tracking the score of the current $\mu$

- gradient = $(\epsilon_\phi - \epsilon_\psi)\cdot \partial x/\partial \theta$ — **relative score**, no huge CFG required

- $\epsilon_\psi$ is the necessary term from differentiating the KL objective w.r.t. $\theta$, corresponding to the $q$ distribution's own score (dropping it degrades back to SDS) — not an RL actor-critic value baseline that only reduces variance

Saying VSD "uses variational inference" but not explaining $\epsilon_\psi$ replacing raw noise.

</details>

<details>

<summary>Q16. Differences between Zero-1-to-3 / SyncDreamer / MVDream?</summary>

- **Zero-1-to-3** (Liu 2023 ICCV): input view + camera $\Delta R, \Delta T$ → single novel view; sampled independently each time

- **Zero-1-to-3++** (Shi 2023): fixed 6 anchor views, multiple in one shot (reduces randomness)

- **SyncDreamer** (Liu 2024 ICLR): **joint** prediction of multi-view in latent space, cross-attention so views see each other → better consistency

- **MVDream** (Shi 2024 ICLR): text-to-multi-view (no input image needed), 4 views generated together + SDS refinement

Saying only "they're all novel views" misses the independent vs joint vs text-only main thread.

</details>

<details>

<summary>Q17. How does Mip-NeRF anti-alias?</summary>

- Vanilla NeRF treats pixels as rays; at different resolutions the same pixel corresponds to different scales → aliasing

- **Mip-NeRF** treats pixels as cones (view frustums), approximating cone segments as anisotropic Gaussians

- **IPE (Integrated Positional Encoding)**: $\mathbb{E}_{\mathbf{x}\sim\mathcal{N}(\mu,\Sigma)}[\gamma(\mathbf{x})]$ has a closed-form solution

- High-frequency coefficients are attenuated by $\Sigma$ → automatic multi-scale smoothing

Just saying "uses cones" without explaining IPE's high-frequency attenuation.

</details>

<details>

<summary>Q18. Difference between NeuS and vanilla NeRF for mesh extraction?</summary>

- Vanilla NeRF: density has no explicit surface; extracting mesh requires picking a $\sigma$ threshold (unstable)

- **NeuS** (Wang 2021 NeurIPS): replaces density with **SDF $d(\mathbf{x})$**, defines $\sigma$ via the sigmoid derivative

- Surface = $\{d = 0\}$, **well-defined**

- Marching Cubes runs directly on the SDF, quality is much better

Saying "use SDF" without explaining how NeuS plugs SDF into NeRF volume rendering.

</details>

<details>

<summary>Q19. Core of the LRM family (TripoSR / InstantMesh)?</summary>

- **Triplane** representation: 3 axis-aligned 2D planes, $O(N^2)$ VRAM

- Transformer maps image tokens + Plucker ray embeddings → regress triplane tokens

- Inference is feedforward (no SDS / no iterative optimization), **seconds to a 3D output** (TripoSR ~0.5 s, InstantMesh ~10 s)

- TripoSR (Stability+Tripo 2024) / InstantMesh (Xu 2024) / SF3D (2024) all belong to this family

Treating them as the SDS family — wrong, LRM is fully feedforward; it is not distillation.

</details>

<details>

<summary>Q20. How is mesh extracted from 3DGS?</summary>

- Vanilla 3DGS is unfriendly (ellipsoids are not surfaces)

- **SuGaR** (Guédon 2024 CVPR): surface-alignment loss + Poisson reconstruction

- **2DGS** (Huang 2024 SIGGRAPH): degenerate ellipsoids to 2D disks, align surface → MC mesh extraction is more stable

- **GSDF** (Yu 2024): jointly train an SDF head with 3DGS

Saying "just run MC" — 3DGS has no density field, MC doesn't work directly; surface alignment is required first.

</details>

### L3 top-lab (top-conference / industry research roles)

<details>

<summary>Q21. Manually derive NeRF's discrete $\alpha$-compositing.</summary>

- ODE $dT/dt = -\sigma(t) T(t)$, initial value $T(t_n) = 1$ → $T(t) = \exp(-\int_{t_n}^t \sigma\,ds)$

- On segment $[t_i, t_{i+1}]$ $\sigma$ is constant $= \sigma_i$, so $T(t_{i+1}) = T(t_i)e^{-\sigma_i\delta_i}$

- Inter-segment accumulation $T_i = T(t_i) = \prod_{j<i} e^{-\sigma_j\delta_j} = \prod_{j<i}(1 - \alpha_j)$, where $\alpha_j = 1 - e^{-\sigma_j\delta_j}$

- Within-segment color contribution $\int_{t_i}^{t_{i+1}} T(t)\sigma_i\mathbf{c}_i\,dt = \mathbf{c}_i T_i \int_0^{\delta_i}\sigma_i e^{-\sigma_i s}ds = T_i\mathbf{c}_i(1 - e^{-\sigma_i\delta_i}) = T_i\alpha_i\mathbf{c}_i$

- Compositing: $C \approx \sum_i T_i\alpha_i \mathbf{c}_i$

- **Key**: $\alpha_i = 1 - e^{-\sigma_i\delta_i}$ exactly, vs $\alpha_i \approx \sigma_i\delta_i$ first-order approximation (consistent when $\sigma\delta \ll 1$)

Skipping the ODE derivation and going straight to the conclusion; or using $\sigma_i\delta_i$ to substitute for $\alpha_i$ when $\sigma\delta$ is large, which is wrong.

</details>

<details>

<summary>Q22. How are Instant-NGP hash collisions automatically disambiguated by the MLP?</summary>

- **When collisions happen**: at fine levels, grid-point count $N_\ell^d > T$ (hash table size), multiple grid points map to the same entry

- **Sparse activation**: most scene voxels are background, **only voxels near the surface have non-zero supervised gradients** — two colliding "background entries" never receive signals and don't pollute each other

- **Multi-resolution redundancy**: at coarse levels $N_\ell^d \le T$ guarantees uniqueness; fine levels add detail. Even if a fine level collides, the non-colliding features at coarse levels already uniquely identify the point

- **MLP post-processing**: the tiny MLP learns nonlinear fusion over the $L\times F$ concatenated features; on colliding entries, it can use **non-colliding features from other levels to disambiguate**

- **Auto-regulated gradients**: during training high gradients naturally concentrate at surface entries; if colliding entries are simultaneously on the surface (rare), the loss pushes them to a compromise position (averaging multiple samples)

- **Physical intuition**: rather than pay the cost of perfect hashing, allow collisions and use data-driven implicit disambiguation ("lazy collision resolution")

Saying "hash collisions are solved by the MLP" without explaining which mechanisms (sparsity + multi-scale + MLP nonlinearity) act together.

</details>

<details>

<summary>Q23. Derive the 3DGS 3D→2D covariance projection Jacobian.</summary>

- World → Camera: rigid transform $\mathbf{x}_\text{cam} = W\mathbf{x} + t$; covariance is only affected by rotation, $\Sigma_\text{cam} = W\Sigma W^\top$

- Camera → Screen: perspective projection $\pi(x, y, z) = (f_x x/z, f_y y/z)$ is nonlinear

- First-order Taylor at the mean $\mu_\text{cam}$: $\pi(\mathbf{x}) \approx \pi(\mu_\text{cam}) + J(\mathbf{x} - \mu_\text{cam})$, $J = \partial\pi/\partial\mathbf{x}|_{\mu_\text{cam}}$

- $J = \begin{pmatrix} f_x/z & 0 & -f_x x/z^2 \\ 0 & f_y/z & -f_y y/z^2 \end{pmatrix} \in \mathbb{R}^{2\times 3}$

- $\text{Cov}[\pi(\mathbf{x})] = J\Sigma_\text{cam} J^\top = JW\Sigma W^\top J^\top \in \mathbb{R}^{2\times 2}$

- This is the classic corollary of EWA splatting (Zwicker 2001); 3DGS adopts it directly

- Real implementations also add a $0.3 I$ low-pass filter (anti-aliasing)

Unable to do the first-order Taylor linearization of the nonlinear projection; or missing the World→Cam step.

</details>

<details>

<summary>Q24. Which Jacobian does SDS gradient drop? Why does it still "work"?</summary>

- **Naive diffusion training gradient**:

  $\nabla_\theta \mathcal{L}_\text{diff} = \mathbb{E}[w(t)\cdot 2(\epsilon_\phi - \epsilon)\cdot \underbrace{\partial \epsilon_\phi/\partial x_t}_{\text{U-Net Jacobian}}\cdot \alpha_t \cdot \partial x/\partial\theta]$

- **SDS** drops the U-Net Jacobian $\partial \epsilon_\phi/\partial x_t$:

  $\nabla_\theta \mathcal{L}_\text{SDS} = \mathbb{E}[w(t)(\epsilon_\phi - \epsilon)\cdot \partial x/\partial\theta]$

- **Why dropping it is reasonable**:
  - The U-Net Jacobian is expensive (H×W×3 input → H×W×3 output second-order)
  - U-Net is not trained for second-order stability, the Jacobian is numerically poor
  - $(\epsilon_\phi - \epsilon)$ itself is already a proxy for the score ($\epsilon_\phi/\sigma_t \approx -\nabla_{x_t}\log p_\phi$); dropping the Jacobian amounts to using a first-order score signal

- **Cost**: SDS mathematically becomes a mode-seeking KL (toward modes of the prior), needing large CFG (100) to escape mean-blur

- **Symptoms**: over-saturation (saturated colors) + Janus (same face in multiple views) + over-smoothing (blurry details)

Saying only "drop the Jacobian for simplicity" without explaining the mode-seeking consequence + why large CFG is needed.

</details>

<details>

<summary>Q25. Why can VSD avoid over-saturation under small CFG?</summary>

- **SDS view**: $\theta$ is a point estimate; the gradient pulls toward modes of $p_\phi(\cdot|y)$; large CFG sharpens the modes further → over-saturation

- **VSD view**: $\theta$ is a random variable $\mu(\theta)$; **minimizes the KL between noised rendered-image distributions**: $\mathbb{E}_t[D_\text{KL}(q_\mu^t(x_t|y) \,\|\, p_\phi^t(x_t|y))]$ (not a direct KL against a 3D prior in the $\theta$ domain)

- Introduces an **auxiliary score** $\epsilon_\psi$ (LoRA-finetuned Stable Diffusion) tracking the score of the current rendered distribution

- **VSD gradient**:

  $\nabla_\theta \mathcal{L}_\text{VSD} = \mathbb{E}[w(t)(\epsilon_\phi - \epsilon_\psi)\cdot \partial x/\partial \theta]$

  i.e. **relative score** (target prior score $-$ current rendered score)

- **Geometric intuition**: points from "where I am now" to "where the prior is" — a local "gradient direction" rather than a global mode; no large CFG sharpening needed

- **Not an RL baseline**: $\epsilon_\psi$ is the necessary term corresponding to the $q$ distribution's own score from differentiating the KL objective w.r.t. $\theta$; dropping it degrades the gradient objective back to SDS's mode-seeking (changing the expected gradient direction), unlike an RL actor-critic value baseline, which only reduces variance without changing the expected gradient

- **Effects (ProlificDreamer)**: CFG can drop to 7.5, natural colors; finer geometry; can maintain multiple modes (diversity)

Saying only "VSD introduces variational inference" without explaining the role of $\epsilon_\psi$ + the relative-gradient view.

</details>

## §A Appendix: full code skeleton + references

### A.1　Complete from-scratch code includes

`volume_render()` (NeRF α-compositing with numerical stability) · `positional_encoding()` (γ(p) Fourier features) · `gaussian_splat_forward()` (3DGS pedagogical forward + projection Jacobian) · `densify_and_prune()` (3DGS densification heuristics) · `sds_loss()` (SDS gradient surrogate) · `marching_cubes_sketch()` (mesh extraction interface using scikit-image).

### A.2　Key papers reading list

- **NeRF family**: Mildenhall 2020 ECCV (HM); Müller **Instant-NGP** SIGGRAPH 2022 Best; Barron **Mip-NeRF** / **360** ICCV 2021 / CVPR 2022; Wang **NeuS** + Yariv **VolSDF** NeurIPS 2021; Fridovich-Keil **Plenoxels** CVPR 2022; Chen **TensoRF** ECCV 2022.
- **3DGS family**: Kerbl **3D Gaussian Splatting** SIGGRAPH 2023 Best; Huang **2D Gaussian Splatting** SIGGRAPH 2024; Luiten **Dynamic 3DGS** 3DV 2024; Wu **4DGS** CVPR 2024; Guédon **SuGaR** CVPR 2024.
- **Mesh / SDF**: Shen **DMTet** NeurIPS 2021 / **FlexiCubes** SIGGRAPH 2023.
- **SDS family**: Poole **DreamFusion** arXiv 2022.09 → ICLR 2023 Outstanding; Wang **ProlificDreamer (VSD)** NeurIPS 2023 Spotlight; Lin **Magic3D** CVPR 2023; Chen **Fantasia3D** ICCV 2023; Tang **DreamGaussian** ICLR 2024; Yi **GaussianDreamer** CVPR 2024.
- **Single-image 3D**: Liu **Zero-1-to-3** ICCV 2023 / **One-2-3-45** NeurIPS 2023 / **SyncDreamer** ICLR 2024; Shi **Zero-1-to-3++** arXiv 2023 / **MVDream** ICLR 2024; Hong **LRM** arXiv 2023.11 → ICLR 2024; Tochilkin **TripoSR** arXiv 2024; Xu **InstantMesh** arXiv 2024; Boss **Stable Fast 3D** arXiv 2024.
- **3D Foundation Models**: Xiang **Trellis** arXiv 2024 (Microsoft); Tencent **Hunyuan3D-2** arXiv 2501.12202 (2025); Zhang **CLAY** SIGGRAPH 2024.

### A.3　Common Embodied AI / AR / VR follow-ups

3DGS connected to a physics engine → first extract mesh via 2DGS / SuGaR → IsaacSim / MuJoCo; dynamic NeRF → 4DGS / D-NeRF / K-Planes; real-time AR 3DGS → mobile-friendly (PostShot, Luma) + pruning / quantization; insufficient 3D data → Objaverse-XL (Trellis), 2D distillation (DreamFusion family), or multi-view heuristics (MVDream).

---

**3D Generation Quick Reference** · Main references: Mildenhall 2020 (NeRF), Müller 2022 (Instant-NGP), Kerbl 2023 (3DGS), Poole 2022/ICLR 2023 (DreamFusion), Wang 2023 (VSD), Xiang 2024 (Trellis), Tencent 2025 (Hunyuan3D-2). Covers: NeRF volume-rendering derivation, Instant-NGP hash grid, 3DGS projection Jacobian, SDS / VSD gradient derivation, single-image 3D, 3D foundation models. Essential for Embodied AI / AR / VR.
