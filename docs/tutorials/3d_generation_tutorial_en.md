## §0 TL;DR Cheat Sheet

> 💡 **11 sentences to nail 3D Generation** — interview core for Embodied AI / AR / VR roles (see §1–§12 for derivations).

1. **Three representations**: **NeRF** (implicit neural field + volume rendering), **3DGS** (explicit Gaussian point cloud + rasterization), **Mesh / SDF** (explicit surface / implicit distance field). Sweet spot for quality vs speed: 3DGS (Kerbl 2023 SIGGRAPH Best Paper).

2. **NeRF core equation**: $C(\mathbf{r}) = \int_{t_n}^{t_f} T(t)\sigma(\mathbf{r}(t))\mathbf{c}(\mathbf{r}(t),\mathbf{d})\,dt$, where $T(t) = \exp\!\left(-\int_{t_n}^{t}\sigma(\mathbf{r}(s))\,ds\right)$ is the transmittance. Discretization yields $\alpha$-compositing: $C \approx \sum_i T_i (1-e^{-\sigma_i\delta_i})\mathbf{c}_i$.

3. **Instant-NGP** (Müller 2022 SIGGRAPH): **multi-resolution hash grid** + tiny MLP, roughly a 4+ order-of-magnitude speedup; hash collisions are automatically disambiguated by the MLP over colliding entries (suppressed jointly by the loss and multi-scale redundancy).

4. **3DGS core**: scene represented as a set of 3D Gaussians $\{\mu_i, \Sigma_i, \alpha_i, c_i(\mathbf{d})\}$, **differentiable rasterization** projects the 3D covariance to 2D via Jacobian $J$: $\Sigma' = J W \Sigma W^\top J^\top$, then performs front-to-back alpha-blending after depth sorting.

5. **DreamFusion SDS** (Poole et al. 2022 arXiv → ICLR 2023 Outstanding): supervise a 3D representation using a pretrained 2D diffusion: $\nabla_\theta \mathcal{L}_\text{SDS} = \mathbb{E}_{t,\epsilon}[w(t)(\epsilon_\phi(x_t;y,t)-\epsilon)\,\partial x/\partial \theta]$, **deliberately dropping the U-Net Jacobian** of $x_t$ to make training simulation-free. Price: mode-seeking → over-saturation / Janus.

6. **VSD** (Wang 2023 NeurIPS, ProlificDreamer): treats the 3D parameters $\theta$ as a random variable $\mu(\theta)$ and **minimizes the KL between the noised rendered image distributions**: $\mathbb{E}_t\big[D_\text{KL}\big(q_\mu^t(x_t|y)\,\|\,p_\phi^t(x_t|y)\big)\big]$. The gradient form is a **relative score** $\nabla_\theta \approx (\epsilon_\phi(x_t;y,t) - \epsilon_\psi(x_t;y,t,\pi))\,\partial x/\partial\theta$, where $\epsilon_\psi$ is a LoRA-finetuned auxiliary score. CFG can drop from 100 to 7.5.

7. **Single-image / Few-view 3D**: Zero-1-to-3 (Liu 2023 ICCV) uses viewpoint-conditioned diffusion; SyncDreamer / MVDream learn joint multi-view consistency; TripoSR / InstantMesh / Stable Fast 3D push image-to-mesh to the seconds regime (TripoSR ~0.5 s, InstantMesh ~10 s).

8. **Feed-forward reconstruction**: **DUSt3R** (arXiv:2312.14132) replaces SfM with a single forward pass — it regresses pointmaps directly, both images' 3D points landing in the first image's camera frame, so **pose becomes an output rather than an input**; the loss divides by the mean distance of valid points to the origin, hence up-to-scale by default. **VGGT** (2503.11651, CVPR 2025 Best Paper) draws four heads (camera / depth / point map / track) off one backbone. COLMAP thereby drops from "mandatory" to "accuracy reference" (§7, §12.1).

9. **3D Foundation Models (2024-26 open source)**: TRELLIS's **SLAT** = sparse voxel coordinates + per-voxel latent ($N=64$, $L\approx 20\text{K}$ active voxels), two-stage rectified flow, one latent decoding into 3DGS / field / mesh; **Hunyuan3D** iterates 2.0 → 2.1 → 2.5 → Omni / Studio / Buffalo (**there is no 3.0**), shape→texture two-stage; **CLAY** (arXiv:2406.13897) is multi-resolution VAE + latent DiT.

10. **Native mesh generation and the latent-representation axis**: whether the mesh comes from isosurface extraction (marching cubes, FlexiCubes) or from a direct O-Voxel conversion as in TRELLIS.2, **none of these outputs is guaranteed to have edge flow suited to editing, rigging and deformation** — that, not raw quality, is why artists reject it. **MeshGPT → MeshAnything V2 (AMT) → BPT → TreeMeshGPT** compresses tokens against a baseline of 9 per face, while **Meshtron instead answers with architecture** (hourglass + sliding window). Latent representations (SLAT / sparse voxel / VecSet / triplane) can all be decoded by coordinate query; what they decide is not "how many output formats" but **where compute and memory go**.

11. **Embodied AI key applications**: Sim2Real asset generation, NeRF/3DGS as differentiable simulators, language-conditioned 3D affordance. **Common interview crossovers**: NeRF SLAM, Gaussian-Splat scene editing, 3D physics consistency.

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

## §7 Feed-forward reconstruction: from COLMAP to pointmap regression

Before 2024, the first step of "multi-view → 3D" was almost always SfM: COLMAP running SIFT → matching → incremental SfM → bundle adjustment, tens of minutes to hours, and frequently failing on texture-less surfaces, low overlap, and dynamic objects. **DUSt3R** (Wang 2024 CVPR, arXiv:2312.14132, Naver) replaced that entire pipeline with a single network forward pass: no explicit matching, no triangulation, no bundle adjustment — it **regresses pointmaps directly**. By the time VGGT won CVPR 2025 Best Paper, this line had become the default question direction for 3D vision roles.

### 7.1　DUSt3R: pointmaps and the confidence-weighted loss (**must-derive**)

A **pointmap** $X \in \mathbb{R}^{H\times W\times 3}$ is a dense "one 3D point per pixel" map whose points are expressed **in one designated camera frame**. It constrains geometry, intrinsics and pose at once, but each is recovered differently:

- **Intrinsics**: pair the 3D points of $X^{1,1}$ with their own pixel coordinates and fit a projection model.
- **Pose**: pair the 3D points of $X^{2,1}$ with the pixel coordinates of $I^2$ and solve **PnP** — it rests on 2D-3D correspondence.
- **Depth**: the $z$ component of $X^{2,1}$ is **not** $I^2$'s depth, because those points live in $I^1$'s frame; to get $I^2$'s depth you must first transform the points back into $I^2$'s camera frame and then take $z$.

Given two images $I^1, I^2$, DUSt3R encodes them with weight-shared ViT encoders, lets two decoders cross-attend to each other, and outputs two pointmaps $X^{1,1}$ and $X^{2,1}$. **The superscripts are the point: both pointmaps are expressed in the camera frame of $I^1$.** Both images' pixels now live in one coordinate system, so **pose stops being an input and becomes something you can solve for from the output**. That one sentence is the origin of the whole feed-forward reconstruction family.

> ⚠️ **Do not say "rigidly align the two pointmaps to get the pose"** — they are already in the same frame, so there is nothing to register. The pose comes from 2D-3D correspondences between $X^{2,1}$ and $I^2$'s pixels (PnP), which is a different operation.

**Regression loss.** For valid pixels $i \in \mathcal{D}^v$ of view $v \in \{1, 2\}$ (pixels where GT is defined), the per-point Euclidean error is

$$\ell_{\text{regr}}(v,i)=\Big\lVert \tfrac{1}{z}X^{v,1}_i-\tfrac{1}{\bar z}\bar X^{v,1}_i\Big\rVert$$

where $\bar X$ is the GT. Each side is divided by a scale factor — $z$ from the prediction, $\bar z$ from the GT — both defined identically:

$$z=\text{norm}(X^1,X^2)=\frac{1}{\lvert\mathcal{D}^1\rvert+\lvert\mathcal{D}^2\rvert}\sum_{v}\sum_{i\in\mathcal{D}^v}\lVert X^v_i\rVert$$

that is, the **mean distance of all valid points to the origin**. After dividing by it the loss is invariant to global scaling: the network is only asked to get the shape right, scale is free. **This is exactly where "DUSt3R-style training is up-to-scale by default" comes from.**

Getting absolute scale is done in more than one way, and **MapAnything** is one of them: it keeps this scale-invariant geometric supervision and **separately predicts a global scale with its own loss**. Normalisation itself also differs — **VGGT** normalizes only the GT, without DUSt3R's prediction-side normalization, but what it learns is still the normalised scale, not a metric route. The `use_metric` switch in the code below (setting $z=\bar z$) is the most direct teaching variant, pressing absolute scale straight into the regression term — easy to reason about, but not what either paper does.

**Confidence weighting.** Real data contains sky, specularities, transparency and moving objects — pixels where the GT itself is untrustworthy. DUSt3R has the network emit an extra confidence map $C^{v,1}$, turning the loss into

$$\boxed{\;\mathcal{L}_{\text{conf}}=\sum_{v}\sum_{i\in\mathcal{D}^v}\Big[\,C^{v,1}_i\,\ell_{\text{regr}}(v,i)\;-\;\alpha\log C^{v,1}_i\,\Big]\;},\qquad C=1+\exp(\check C)\gt 1$$

where $\check C$ is the raw output of the confidence head and the $1+\exp(\cdot)$ parameterization guarantees $C\gt 1$ (every pixel is counted at least once). The paper calls $-\alpha\log C$ a **regularization term**.

**Why $-\alpha\log C$ is mandatory** (this is the question). Look at one pixel's loss as a function of $C$ on the domain $C\gt 1$: $f(C)=C\ell-\alpha\log C$.

- **Drop the $\log$ term**: $f(C)=C\ell$ is non-decreasing in $C$ (since $\ell \ge 0$), so the optimum pushes $C$ to the lower end of the domain, $C\to 1$, and the per-pixel loss falls back to the plain regression loss $\ell$. **Geometry is still learned** — what degenerates is only the confidence head: it becomes a constant carrying no information. Blocking that degeneracy is exactly what the regularizer is for.
- **Keep the $\log$ term**: $f'(C)=\ell-\alpha/C$. For $\ell\lt \alpha$ the stationary point $C^\star=\alpha/\ell\gt 1$ lies inside the domain; for $\ell\ge\alpha$, $f'(C)\gt 0$ throughout and the infimum is still at $C\to 1^+$.
- **Eliminate $C$** (taking the infimum over $C$ — an **optimization elimination**, not a probabilistic marginalization):

$$\inf_{C\gt 1} f(C)=\begin{cases}\alpha+\alpha\log(\ell/\alpha), & 0\lt \ell\lt \alpha\\[2pt] \ell, & \ell\ge\alpha\end{cases}$$

- **So what does it actually do**: the $\ell\lt \alpha$ branch is reshaped into a logarithm, while the $\ell\ge\alpha$ branch **stays linear in $\ell$**. It is **not** an outlier suppressor that squashes large residuals into a log — large residuals are still counted linearly. $\alpha$ is where the boundary sits: raising $C$ only pays off once a pixel's residual drops below $\alpha$ (buying a negative $\alpha\log(\ell/\alpha)$). So $C$ learns "can I get this pixel right", and on sky, specularities and moving objects — where it cannot — $C$ sits near 1 and earns no reward.

> 💡 **$\alpha$ is the only knob** — it is simultaneously the threshold for "how small a residual is worth committing to" and the strength of the confidence reward. Raise $\alpha$ and more pixels enter the logarithmic branch, making the confidence map more outspoken; at a fixed residual, lowering $\alpha$ moves more pixels into the $C\to 1$ branch, collapsing the mechanism back to plain regression. Naming the boundary at $\ell=\alpha$ puts this answer a tier above reciting the formula.

```python
import torch

def dust3r_conf_loss(pred_pts, gt_pts, conf_raw, valid, alpha=0.2, use_metric=False):
    """ DUSt3R confidence-weighted pointmap loss (pedagogical version).
        pred_pts: [B, V, H, W, 3]  pointmaps from both heads, in the I^1 camera frame
        gt_pts:   [B, V, H, W, 3]  GT in the same frame (V = 2); invalid pixels may be NaN
        conf_raw: [B, V, H, W]     raw confidence-head output Ĉ (pre-activation)
        valid:    [B, V, H, W]     bool, the set of pixels D^v where GT is defined
        use_metric: teaching variant setting z = z̄ to press absolute scale into the
                    regression term (not what the paper does)
        Note: the paper's L_conf sums over valid pixels; this returns a per-sample mean
        over valid pixels, then a batch mean, so values compare across batches.
    """
    eps = 1e-8
    vm = valid.unsqueeze(-1)                                       # [B, V, H, W, 1]
    zero = torch.zeros((), dtype=pred_pts.dtype, device=pred_pts.device)
    # select before computing: invalid GT depth is often NaN, and NaN * 0 is still NaN
    pred_v = torch.where(vm, pred_pts, zero)
    gt_v = torch.where(vm, gt_pts, zero)

    n = valid.flatten(1).sum(1).to(pred_pts.dtype).clamp(min=1.0)  # [B]  |D^1| + |D^2|
    # norm(.) = mean distance of valid points to the origin; one for pred, one for gt
    z_bar = gt_v.norm(dim=-1).flatten(1).sum(1) / n                # [B]  z̄
    z = z_bar if use_metric else pred_v.norm(dim=-1).flatten(1).sum(1) / n

    bshape = (-1,) + (1,) * (pred_pts.dim() - 1)                   # [B, 1, 1, 1, 1]
    l_regr = (pred_v / z.view(bshape).clamp(min=eps)
              - gt_v / z_bar.view(bshape).clamp(min=eps)).norm(dim=-1)   # [B, V, H, W]

    C = 1.0 + conf_raw.exp()                                       # C > 1, every pixel counted
    per_pix = C * l_regr - alpha * C.log()                         # drop the log term -> C collapses to 1
    per_pix = torch.where(valid, per_pix, torch.zeros((), dtype=per_pix.dtype,
                                                      device=per_pix.device))
    return per_pix.flatten(1).sum(1).div(n).mean()
```

> ⚠️ **Two easy mistakes** — first, the scale normalization must be computed over **both images jointly** ($\lvert\mathcal{D}^1\rvert+\lvert\mathcal{D}^2\rvert$ is the denominator); normalizing separately erases the relative scale between the two images. Second, $C$ must not be detached — gradients have to reach the confidence head, otherwise the "only raise $C$ once the residual is below $\alpha$" mechanism never forms.

**MASt3R** (2406.09756) adds a dense local-feature head on top of DUSt3R, fusing "regress a pointmap" and "do pixel-level matching" into one model with much better matching accuracy; the follow-up MASt3R-SfM uses it to replace COLMAP's matching front end.

### 7.2　The shared recipe and VGGT's four heads

The recipe across this family is broadly uniform, but **every line has exceptions and a good answer names them**:

1. **Views exchange information through attention**, rather than hand-crafted matching followed by geometric solving. But "one forward pass consumes all views" is not universal: DUSt3R itself is a **pairwise** model whose multi-view use still needs a global alignment stage (and does not require running every pair); Fast3R is the one that truly emits all pointmaps in a single pass, though its **camera parameters are recovered from the pointmaps afterwards**.
2. **Outputs land in one common frame**, usually the camera frame of some reference image. **π³ is the exception**: its raw outputs are per-view local pointmaps in each view's own camera frame plus the corresponding poses, and the common frame is assembled from those poses.
3. **Pose is usually an output rather than a required input** — no SfM initialization needed.
4. **Scale depends on how it was trained**: DUSt3R-style two-sided (prediction + GT) normalization → up-to-scale; VGGT normalizes only the GT. Metric output has more than one route — MapAnything obtains it by predicting a separate global scale, and you can also simply train on metric geometric supervision (CUT3R does exactly that; MASt3R is likewise trained on metric data).

**VGGT** (Wang 2025 CVPR **Best Paper**, arXiv:2503.11651, Oxford + Meta) is the current reference implementation. It processes N images with alternating frame-wise / global attention and draws four heads off **one shared backbone**:

| Head | Output | Downstream use |
| --- | --- | --- |
| **Camera** | Extrinsics + intrinsics per image (quaternion + translation + FOV) | Replaces SfM pose estimation |
| **Depth** | Dense depth per image (DPT-style) | Depth completion / fusion |
| **Point map** | Dense 3D points in the common frame | Point cloud straight out |
| **Track** | Given query pixels, the **2D position of the same physical point in every image** | Correspondence / tracking and matching |

> 💡 **VGGT's most counter-intuitive observation** — the paper reports that **at inference, combining the depth head and camera head into a point cloud is more accurate than using the point map head directly**. Note this is an empirical observation, not a proof that "the benefit of multi-task training does not live in that head"; what it does establish is that **which supervision targets you train on** and **which inference path you take** are two separable decisions.

**Why VGGT displaced "COLMAP first" as the interview answer**: seconds instead of hours; no dependence on pre-calibrated intrinsics and far more tolerance for limited overlap; four quantities out in one pass, directly usable downstream (3DGS initialization, SLAM, robot mapping). But this is not "COLMAP is dead" — see §12.1 for accuracy and benchmark GT.

### 7.3　Branch structure: which directions the family is growing in

- **Fast3R** (2501.13928, Meta, CVPR 2025): DUSt3R is pairwise, so multi-view use means "run some image pairs, then globally align" — two stages. Fast3R pushes **all N images through one forward pass** of a transformer straight to pointmaps — demonstrated up to 1000+ images — dropping the global alignment; camera parameters are then recovered from the pointmaps in post-processing.
- **CUT3R** (2501.12387, Berkeley): goes **recurrent** — it maintains a persistent state, updating it and emitting that frame's pointmap as each frame arrives. That makes it **online / streaming**, able to build while capturing and to accept a single image; its pointmaps and poses are **metric**.
- **π³** (2507.13347): the methods above all nominate a **reference view** as the coordinate origin, so results depend on which one was picked and a degenerate reference drags everything with it. π³ is **permutation-equivariant**: the output is equivariant to input ordering, so **no single view is privileged**.
- **MapAnything** (2509.13414, Meta + CMU, 3DV 2026): unifies a pile of tasks with a **factored metric representation** (ray directions + depth + pose + scale), and its **inputs are optional** — use intrinsics if you have them, poses if you have them, run with nothing if you have neither. Scale comes from a separately predicted global factor.
- **Depth Anything 3** (2511.10647, ByteDance): subtraction in the other direction — no bespoke architecture, no four heads, just **a plain transformer with a single depth-ray prediction target**. On the authors' own benchmark the paper reports **average relative improvements** over VGGT of 44.3% on camera pose and 25.1% on geometry. **Open weights.**
- **VGGT-Ω** (2605.15195, CVPR 2026 Oral): cuts VGGT's **training memory** to 30% and scales the **supervised training data** to 15× (two separate facts, not an equal-budget exchange), extending coverage to **dynamic scenes** (the VGGT line originally assumes static ones). It predicts depth and cameras. **Weights unverified.**

| Model | Input views | Main outputs | Pose input needed? | Metric? | Weights |
| --- | --- | --- | --- | --- | --- |
| **DUSt3R** (2312.14132) | 2 (multi-view via global alignment) | pointmap + confidence | No | No (up-to-scale) | Open weights |
| **MASt3R** (2406.09756) | 2 | pointmap + dense local features | No | Yes (trained on metric data) | Open weights |
| **VGGT** (2503.11651) | N, one forward pass | camera / depth / pointmap / track | No | No | Open weights (non-commercial license) |
| **Fast3R** (2501.13928) | N, demonstrated 1000+ | pointmap (cameras recovered in post) | No | No | Open weights |
| **CUT3R** (2501.12387) | streaming frame-by-frame (incl. single image) | pointmap + pose + persistent state | No | Yes | Open weights |
| **π³** (2507.13347) | N, permutation-equivariant | pointmap + camera | No | No | Open weights |
| **MapAnything** (2509.13414) | N, other inputs optional | factored metric representation | Optional (used if present) | Yes | Open weights |
| **Depth Anything 3** (2511.10647) | N | depth-ray → pose / geometry | No | — | Open weights |
| **VGGT-Ω** (2605.15195) | N, incl. dynamic scenes | depth + camera | No | No | Weights unverified |

**MV-DUSt3R+** (2412.06974) is another transition route from DUSt3R to multi-view (multi-view decoder blocks + cross-reference-view fusion); a systematic survey of the family is arXiv:2507.08448.

### 7.4　Feed-forward 3DGS

The line above regresses geometry; a parallel line regresses a **directly renderable 3DGS** — replacing the 10-30 minutes of per-scene optimization from §4 with a single forward pass. **Note that this line is not uniform about needing input poses**:

- **LGM** (2402.05054, Tang 2024 ECCV): an asymmetric U-Net takes 4 multi-view images and **emits one 3D Gaussian per pixel**, producing an asset in seconds. Those 4 images come from **preset, fixed orbit camera poses** that the upstream multi-view generator renders to.
- **GS-LRM** (2404.19702): swaps LRM's transformer output for Gaussian parameters — **per-pixel Gaussians**, patch tokens in, Gaussians out, from **2-4 posed views**, in **0.23 s on an A100**. This is the pivot from "LRM outputs a triplane" to "LRM outputs 3DGS" (LRM itself is in §8.3).
- **Long-LRM** (2410.12781): pushes the input to **32 posed views at 960×540**, reconstructing a room-scale scene in roughly **1 second** on an A100 — about **800×** faster than per-scene optimization, at comparable or better quality. Followed by Long-LRM++ (2512.10267).
- **DepthSplat** (2410.13862, CVPR 2025): feeds pretrained monocular depth features into multi-view Gaussian prediction so depth and splatting reinforce each other — the depth prior supplies exactly the constraint that sparse views lack most. Its inputs are **calibrated** multi-view images.
- **NoPoSplat** (2410.24207): **the Gaussian network itself needs no pose input** — it **defines the first input image's camera frame as canonical space**, emits all Gaussians there, and injects intrinsics as tokens. The "pose-free" claim is scoped to that network: rendering target views and evaluating still involve a separate pose-estimation procedure.
- **AnySplat** (2505.23716): an uncalibrated image collection goes in and **Gaussians + intrinsics + extrinsics** come out — the point where the geometry-regression line and the 3DGS-regression line merge.

> ⚠️ **Feed-forward 3DGS does not replace 3DGS** — what it replaces is the per-scene optimization step, not the representation; the output is still Gaussians, still rasterized per §4.3, and **still optimizable afterwards**. Which is higher quality is conditional: Long-LRM reports comparable or better quality than per-scene optimization in its room-scale setting, while optimization usually stays ahead when views are dense. The trade-off is Q15.

## §8 Single-image / Few-view 3D generation

A more practical setting: **given one image, generate 3D**.

### 8.1　Zero-1-to-3 paradigm (novel view via diffusion)

**Zero-1-to-3** (Liu 2023 ICCV): finetune Stable Diffusion on Objaverse so it accepts (input view, target camera) → output novel view.

- Input: single image $x$ + relative camera pose $\Delta R, \Delta T$
- Diffusion conditioning: image embedding (CLIP) + camera embedding (sinusoidal)
- Output: the image from the $\Delta R, \Delta T$ viewpoint

**Usage**: given one input view, sample 16-32 novel views, then reconstruct via NeRF / 3DGS.

**Derivatives**:
- **Zero-1-to-3++** (Shi 2023): fixed generation of 6 anchor views (azimuth uniformly spaced every 60° at 30°/90°/150°/210°/270°/330°, elevation alternating between -10° and 20° — i.e. interleaved elevation + uniform azimuth, not "one north-pole view + four eye-level views + one top view"), reducing randomness
- **SyncDreamer** (Liu 2024 ICLR): **joint** prediction of multiple views in latent space (cross-attention lets views see each other), ensuring 3D consistency
- **MVDream** (Shi 2024 ICLR): text-to-multi-view, generates 4 views simultaneously; followed by SDS refinement

### 8.2　One-2-3-45 / InstantMesh / TripoSR / Stable Fast 3D

| Method | Input | Output | Speed | Key |
| --- | --- | --- | --- | --- |
| **One-2-3-45** (Liu 2023 NeurIPS) | Single image | mesh | 45 s | Zero-1-to-3 → SparseNeuS |
| **One-2-3-45++** (Liu 2024) | Single image | mesh | 60 s | Multi-view + SDF |
| **TripoSR** (Tochilkin 2024, Stability+Tripo) | Single image | NeRF/mesh | 0.5-2 s | LRM-style (Large Reconstruction Model) transformer |
| **InstantMesh** (Xu 2024, arXiv:2404.07191) | Single image | mesh | ~10 s | Zero-1-to-3++ multi-view → sparse-view recon transformer |
| **Stable Fast 3D** (SF3D, Stability 2024) | Single image | textured mesh | ~0.5 s | TripoSR successor; adds illumination disentangle + UV unwrap |

**LRM (Hong et al. 2023 arXiv → ICLR 2024) setting**: treat the image as tokens + Plucker ray embedding, transformer outputs a NeRF triplane. This is the parent model of TripoSR / InstantMesh.

### 8.3　LRM Triplane representation (**high-frequency interview topic**)

- **Triplane** (Chan 2022 EG3D): 3 axis-aligned 2D planes (XY, YZ, XZ), total $3 \times C \times N \times N$ dim
- Query 3D point $(x, y, z)$: bilinearly interpolate on each plane → **element-wise sum of the three plane features** (EG3D uses sum, not concat) → small MLP → $(\sigma, \mathbf{c})$
- Advantages: less VRAM than a voxel grid ($O(N^2)$ vs $O(N^3)$), denser than a hash grid making it suitable as transformer output
- LRM / TripoSR / InstantMesh all let the transformer directly regress triplane tokens

## §9 3D Foundation Models (the 2024–26 open-source wave)

### 9.1　TRELLIS: SLAT and two-stage rectified flow

**TRELLIS** (Xiang 2024, arXiv:2412.01506, Microsoft) is the most complete open attempt at a "Stable Diffusion for 3D", and the one worth explaining precisely in an interview — because its latent design directly determines its engineering shape.

**SLAT (Structured LATents)** is defined as

$$\boldsymbol z=\{(\boldsymbol z_i,\boldsymbol p_i)\}_{i=1}^{L},\qquad \boldsymbol p_i\in\{0,1,\dots,N-1\}^3,\quad N=64$$

a set of "**latent vector + the voxel coordinate it sits at**". $N=64$ is the voxel resolution ($64^3\approx 2.6\times10^5$ cells), and $L\approx 20\text{K}$ is the number of occupied active voxels — roughly 8% occupancy.

**Get the division of labour right**: $\boldsymbol p_i$ only fixes the **sparse support** — which cells contain material, at $64^3$ granularity. The actual **fine geometry and appearance are both encoded in $\boldsymbol z_i$**, since a 64-resolution voxel grid could never express sharp edges and smooth surfaces on its own. $\boldsymbol z_i$ comes from projecting each active voxel into multi-view renderings, sampling and aggregating from **DINOv2 feature maps**, then compressing with a sparse VAE. So SLAT encodes **geometry and appearance jointly** from the start.

**Two rectified-flow stages** (the order cannot be swapped):

1. **Structure stage**: generate *which* voxels are active, i.e. the sparse structure $\{\boldsymbol p_i\}$ itself.
2. **Latent stage**: generate $\{\boldsymbol z_i\}$ on the already-determined active voxels.

The direct payoff of splitting is that the second stage runs flow over ~20K tokens instead of the $64^3\approx262\text{K}$ cells — **fix the skeleton first, then fill in content, and the attention and forward compute on the 92% empty cells simply never happens**.

**Multiple decoders**: three decoders were trained on the same SLAT, producing **3DGS / Radiance Field / Mesh** (the mesh path goes through FlexiCubes, not naive marching cubes — see §5.2).

Model sizes are **342M / 1.1B / 2B**, trained on roughly **500K** assets filtered from four public datasets (Objaverse(-XL), ABO, 3D-FUTURE, HSSD). **Open weights.**

**TRELLIS.2** (2512.14692) is its successor. Its Sparse Compression VAE takes **O-Voxel** as its **native representation** — sparse voxels carrying geometry and PBR material together; the VAE compresses O-Voxel into a tighter latent, **three flow models** split the generation work on that compressed latent, and meshes come out through its own dual-grid conversion. **Weights unverified.**

### 9.2　The Hunyuan3D family: 2.0 → 2.1 → 2.5 → Omni / Studio / Buffalo

Hunyuan3D takes the **shape-then-texture** two-stage route and is the fastest-iterating line in the Chinese open-source ecosystem.

- **Hunyuan3D-1** (Yang 2024): stage 1 text/image → multi-view (Zero-1-to-3 family), stage 2 multi-view → mesh (LRM-like).
- **Hunyuan3D 2.0** (2501.12202): **Hunyuan3D-DiT** — ShapeVAE encodes a point cloud into a **vecset shape latent**, **flow matching** runs on that latent, and the decoder produces an SDF; **Hunyuan3D-Paint** does multi-view texture diffusion plus UV-space refinement, outputting RGB textures. **Open weights.**
- **Hunyuan3D 2.1** (2506.15442): **PBR material generation enters the main line at this version**, along with the released **training code** — the latter matters far more than the version number for anyone reproducing it.
- **Hunyuan3D 2.5** (2506.16504): the geometry model becomes **LATTICE, 10B parameters**, with clearly better detail and sharp edges. **Weights unverified.**
- **Hunyuan3D-Omni** (2509.21245): controllable generation — beyond image/text it accepts **pose / bounding box / voxel** control signals. **Weights unverified.**
- **Hunyuan3D Studio** (2509.12815): a production-oriented asset toolchain (retopology, UV, texture integration). **Weights unverified.**
- **Hunyuan3D Buffalo 1.0** (2608.02711): unifies **generation, understanding and editing** in one model. **Weights unverified.**

> ⚠️ **There is no "Hunyuan3D 3.0"** — after 2.0 → 2.1 → 2.5 the line forks straight into the Omni / Studio / Buffalo naming scheme. Answering "3.0" when asked for the latest version reveals immediately that the answer came from a second-hand summary.

### 9.3　The latent-representation axis: CLAY / Direct3D / Dora / Step1X-3D

Sorting these by release date teaches nothing; sorting them by **latent representation and decoding target** is the structure an interview wants.

- **CLAY** (Zhang 2024 SIGGRAPH, 2406.13897): **multi-resolution VAE + latent DiT**. Its shape VAE extends the vector-set representation of **3DShape2VecSet** (Zhang et al. 2023, arXiv:2301.11445), placing it in the VecSet lineage; the DiT diffuses on that latent and **decodes an occupancy field**, from which marching cubes extracts the surface, followed by a PBR texture stage.
- **Direct3D** (2405.14832): D3D-VAE encodes the mesh into an **explicit triplane latent** and D3D-DiT diffuses on the triplane. Triplane lets you reuse the entire 2D conv/attention toolchain; the cost is projection ambiguity across the three planes (thin elongated structures overlap on some plane).
- **Direct3D-S2** (2505.17412): switches to a **sparse voxel latent** with **spatial sparse attention (SSA)**, bringing **1024³** training within reach of **8 GPUs**. The reported 3.9× forward / 9.6× backward figures are **the SSA operator against FlashAttention-2**, not an end-to-end model speedup.
- **Dora** (2412.17808, CVPR 2025): stays on **VecSet** and changes the sampling and attention — **Sharp Edge Sampling** tilts the sampling budget toward geometric sharp edges, paired with **dual cross-attention**. The result: Dora-VAE uses **1,280 latent codes** to match the dense XCube-VAE's reconstruction quality on Dora-bench, where the latter needs >10,000. **An order-of-magnitude drop in latent count directly sets the DiT's sequence length. Open weights.**
- **Step1X-3D** (2505.07747): geometry via a hybrid VAE-DiT with a **VecSet latent decoded to TSDF**, trained on **2M** curated assets (filtered from 5M+); code, weights and the training pipeline for both the geometry and texture stages are fully open (Apache-2.0). **Open weights.**
- **Meta AssetGen 2** (2605.26137): a closed product line pushing single-asset generation to roughly **30 seconds** on its H100 deployment, optimized for production usability rather than paper metrics.

**Rodin** (Microsoft 2023, commercial): an early product-grade text-to-3D-avatar system, diffusion on triplane, aimed at characters / avatars.

### 9.4　Comparison table: latent representation is the axis

| Method | **Latent representation** | Decoding target | Prior | Weights |
| --- | --- | --- | --- | --- |
| **TRELLIS** | **SLAT** (sparse voxel support + per-voxel latent) | 3DGS / field / mesh (FlexiCubes) | Rectified Flow ×2 | Open weights |
| **TRELLIS.2** | SC-VAE compressed latent (native representation **O-Voxel**, geometry + PBR) | mesh (dual-grid) + PBR | Flow ×3 | Weights unverified |
| **Hunyuan3D 2.0 / 2.1** | **VecSet** (ShapeVAE shape latent) | SDF → mesh; texture in a separate stage | Flow matching | Open weights |
| **Hunyuan3D 2.5** | VecSet (LATTICE 10B) | SDF → mesh | Diffusion | Weights unverified |
| **CLAY** | **VecSet lineage** (multi-resolution, extends 3DShape2VecSet) | occupancy → mesh | Diffusion (DiT) | Partial |
| **Direct3D** | **Triplane** | implicit field → mesh | Diffusion (DiT) | — |
| **Direct3D-S2** | **Sparse voxel** + SSA | high-resolution implicit field → mesh | Diffusion | — |
| **Dora** | **VecSet** (1,280 codes, Sharp Edge Sampling) | implicit field → mesh | VAE + DiT | Open weights |
| **Step1X-3D** | **VecSet** | **TSDF** → mesh | Diffusion | Open weights |
| **TripoSR / SF3D / LRM** | **Triplane** (feed-forward regression, no prior) | NeRF / mesh | None | Open weights |
| **Rodin** | Triplane | avatar | Diffusion | ❌ |

The "Weights" column says "Open weights" only where verified, "Weights unverified" where no official release was found, and "—" where this tutorial did not check.

> 💡 **What this axis actually decides** — all three latents **can be decoded by coordinate query**: triplane by projection, bilinear interpolation and element-wise sum (§8.3), VecSet by cross-attending from the query coordinate into the latent set, SLAT by indexing the voxel position. So "how many output formats" is a consequence of **which decoders each group trained**, not a prohibition of the representation — TRELLIS trained three, others trained only the mesh path. What the representation does decide is **where compute and memory go**: SLAT and sparse voxels hang latents on an explicit sparse support, which makes local computation and position-matched decoding most natural; VecSet has no spatial index but the set is tiny (Dora compresses it to 1,280), so full attention is cheap — the cost is losing locality; triplane reuses 2D engineering most cheaply, at the cost of plane memory growing with resolution. **Dense tokens can also run windowed / sparse attention** — Direct3D-S2's gain comes from a sparse support making the skipped computation genuinely nonexistent, not from "only sparse representations deserve sparse attention".

### 9.5　Native mesh generation: why artists reject isosurface-extracted output

The mesh output of the methods above comes mainly by **two different routes**: extracting an isosurface from a field — naive marching cubes (§5.1) or a differentiable variant such as FlexiCubes (§5.2, which is what TRELLIS uses); or, as in TRELLIS.2, **converting O-Voxel directly through a dual grid, never passing through a field at all**. **For both routes the problem is not extraction quality but topology**: the triangle layout follows the extraction or voxel grid rather than the shape's structure. They can be UV-unwrapped, decimated and turned into LODs — none of that is impossible, and methods like FlexiCubes preserve sharp edges to a considerable degree. What hurts is that **the output is not guaranteed to have edge flow suited to editing, rigging and deformation**: there are no edge loops organized along silhouettes and creases, so reshaping and skinning are awkward, and decimation has to fight the existing triangulation instead of following it. A hand-built **artist-created mesh** typically has a few hundred to a few thousand faces, with every edge hugging a geometric feature.

Hence **native mesh generation**: no isosurface, no voxel conversion — **autoregressively emit the face sequence directly**. The main thread of this line is **token compression**: sequence length = face count × tokens per face, and transformers are quadratic. The baseline everyone is measured against is the **naive sequence of 9 coordinate tokens per face** (3 vertices × 3 coordinates) — until Meshtron, which changes tack and carries the long sequence with architecture instead of compressing further.

- **MeshGPT** (2311.15475): the origin. Graph-convolutional encoding plus **residual quantization** compresses each face into **6 codebook tokens**, generated autoregressively by a decoder-only transformer.
- **MeshAnything** (2406.10163): adds **shape conditioning** — obtain a coarse shape by any means (reconstruction, generation), then have the AR model "re-model" it into an artist-style mesh, with hundreds of times fewer faces than isosurface extraction gives.
- **MeshAnything V2** (2408.02555): introduces **Adjacent Mesh Tokenization (AMT)** — since adjacent faces share vertices, most faces need only **one new vertex** encoded instead of three, bringing sequence length to about **half** the naive one and doubling the achievable face limit.
- **BPT** (2411.07025): Blocked and Patchified Tokenization uses **block indexing** (coordinates split into block id + in-block offset) together with **vertex sharing inside a patch**, shortening the **naive sequence** by **~75%** and pushing generatable faces past **8K**.
- **TreeMeshGPT** (2503.11629, CVPR 2025): replaces "order the faces somehow" with autoregression along an **adjacency tree** — decoding pops an edge from a stack of expandable edges and grows a face outward. The compressed sequence is **about 22% of the original length** (≈2 tokens/face) — that is "22% remaining", not "22% fewer". Its sequence construction also constrains face orientation, **markedly reducing normal flips**, though not eliminating post-processing entirely.
- **Meshtron** (2412.09548): does not compress tokens but takes the long sequence head-on — **hourglass architecture + sliding-window attention** reaching **64K faces @ 1024 coordinate resolution**. **Weights unverified.**
- **DeepMesh** (2503.15265): attaches **DPO** to AR mesh generation, writing "topology humans prefer" straight into the objective — among the first post-training work in mesh generation.

> ⚠️ **The 2026 counter-current: Nexus** — Nexus (2607.13563) raises a criticism aimed at autoregression itself: **token-by-token generation lets earlier errors propagate down the sequence**, and since this task inherently needs long sequences, compression only shortens the sequence without removing that path. It switches to **coarse-to-fine vertex octree diffusion** to fix the vertex set first, then recovers edges and faces with **topology embeddings**, so there is no per-token error propagation. The cost belongs in the answer too: **the paper itself notes inference is slow** — so replying "it's slow" to a question about AR mesh generation is not wrong, it just stops short of the propagation argument.

### 9.6　Part-level and sim-ready: from a part-less whole to a usable asset

A generated asset usually **lacks explicit part and joint structure**. The drawer a robot must pull, the wheel it must turn, the button it must press are not separately addressable in the output — the simulator cannot assign them mass, friction, or joints. This is the closest and most blocked link between 3D generation and Embodied AI deployment.

**Step one: split into parts.** Four works with different routes and one goal:

- **PartGen** (2412.18608): multi-view diffusion → part-level segmentation + **completion of the occluded portions** (a segmented part is itself incomplete and must be generatively completed).
- **HoloPart** (2504.07943): formalizes this as **3D part amodal segmentation** — decompose a whole mesh into semantically complete parts, each completed into an independent closed body.
- **PartCrafter** (2506.05573): instead of "whole first, then split", it **denoises multiple parts jointly** — spatial relations between parts are learned in that joint training rather than fixed by post-hoc alignment.
- **PartPacker** (2506.09980, NVIDIA): **dual volume packing** packs multiple parts into two volume fields generated at once, sidestepping the trouble that a variable part count causes for a network's output dimension.

**Step two: geometry → parts → joints → materials → URDF.** Splitting parts only solves geometry; sim-ready is three steps further: what **joint** connects two parts (revolute / prismatic / fixed), where its axis sits, what its limits are; each part's **physical material** (mass, friction, restitution); and finally packaging into a **URDF / MJCF** the simulator can read.

- **EmbodiedGen** (2506.10600): one of the most complete open works walking this whole chain, emitting **URDF assets** with physical attributes that plug into IsaacSim / MuJoCo.
- **Artiverse** (2605.24403): **5.4K articulated objects across 88 categories**. For reference, PartNet-Mobility holds **2,346** — articulated assets remain at the thousands scale, and while Artiverse more than doubles that, it is still three orders of magnitude below the geometry layer.

> 💡 **How to answer this section** — do not recite paper names. The structure here is "**one problem cut into five stages whose data supply differs wildly**": geometry has Objaverse-XL-scale data and works best; part level just got PartGen / HoloPart; articulated assets run from PartNet-Mobility's 2,346 to Artiverse's 5.4K, all still thousands-scale; physical materials are mostly specified by hand. **Naming which stage is the bottleneck is worth more than listing ten papers.**

## §10 Complexity / resource comparison

| Method | Training | Inference time | Runtime VRAM (stage noted) | Model / representation size |
| --- | --- | --- | --- | --- |
| NeRF vanilla | 1-2 days | several seconds | 8 GB | <10 MB MLP |
| Instant-NGP | 5 seconds - 5 min | 30 fps+ | 4-12 GB | 100-500 MB hash |
| 3DGS | 10-30 min | 100 fps+ | 6-24 GB | 100 MB - 1 GB Gaussians |
| 2DGS | Close to 3DGS | Close to 3DGS | Similar | Similar |
| DreamFusion (NeRF+SDS) | 2 hr / object | — | 12 GB | NeRF itself |
| DreamGaussian (3DGS+SDS) | 2 min / object | — | 8-16 GB | — |
| ProlificDreamer (VSD) | 3-6 hr / object | — | 24 GB | — |
| TripoSR feedforward | 50 GPU-days training | 0.5 s (A100) | 6 GB inference | 1.5 GB |
| GS-LRM / Long-LRM (feed-forward 3DGS) | Large-scale multi-view training | 0.23 s / ~1 s (32 views at 960×540) | — | — |
| VGGT (feed-forward reconstruction) | Large-scale multi-view training | seconds / N images in one pass | — | ~1B params |
| TRELLIS | 100+ GPU-days training | several seconds | 16 GB inference | a few GB |
| Hunyuan3D 2.0 / 2.1 | Training on large cluster | tens of seconds | 24+ GB inference | combination of multiple models |

## §11 Comparison with related methods & Embodied AI applications

### 11.1　Key differences between 3D and 2D generation

| Dimension | 2D generation (Stable Diffusion) | 3D generation |
| --- | --- | --- |
| **Data scale** | LAION-5B 5B images | Objaverse-XL (2307.05663) 10M objects (500× smaller) |
| **Data format** | Image (uniform RGB) | mesh / SDF / point cloud / NeRF / 3DGS (**fragmented**) |
| **Training prior** | Train diffusion directly | Distill from 2D diffusion (SDS / Zero-1-to-3), **or** 3D-native diffusion (TRELLIS / CLAY), **or** pure feed-forward regression (LRM / VGGT) |
| **Evaluation** | FID, CLIP score | Chamfer / IoU / PSNR (recon) + perceptual + user study |
| **Downstream** | Output image directly | Output asset → rendering / simulation / editing |

### 11.2　Embodied AI / AR / VR practical routes

| Task | Recommended representation | Key toolchain / constraints |
| --- | --- | --- |
| **Sim2Real assets** | mesh (PBR) | TRELLIS / Hunyuan3D 2.1 → IsaacSim / MuJoCo; movable assets still need parts / joints / physical materials (§9.6) |
| **Large indoor scenes** | 3DGS | COLMAP → 3DGS (chunk-wise with VastGS / CityGS); poses can also come from the VGGT family (§12.1) |
| **NeRF/3DGS as simulator** | NeRF / 3DGS + physics | DreamGaussian-Sim / Splatting Physics |
| **3D affordance / manipulation** | point cloud / 3DGS feature | OpenScene / LERF / RVT / 3D Diffuser Actor |
| **AR object scanning** | 3DGS (realistic lighting + real-time) | mobile compute (PostShot / Luma), pruning / quantization |
| **VR large scenes** | 3DGS (large-scale) | 60 fps stereo + 6DoF |
| **Avatar** | mesh + LBS or 3DGS avatar | real-time expressions / hair |
| **Object insertion** | mesh + PBR | consistent environment lighting (IBL) |

> ⚠️ **Embodied AI interview follow-up example** — "Biggest challenge in making NeRF a physics simulator?" Key points: NeRF is radiance, no mass / friction → physics priors must be added manually; mesh extraction has floaters → collision detection is hard; differentiable but slow backward; **industry mostly uses 3DGS / mesh rather than vanilla NeRF**.

## §12 Engineering practice & common footguns

### 12.1　COLMAP or feed-forward?

COLMAP: input multi-view → output intrinsics $K$ + extrinsics $\{R_i, t_i\}$ + sparse point cloud; standard pipeline SIFT → matching → incremental SfM → bundle adjustment. **Common pitfalls**: SfM fails on texture-less / specular objects; dynamic objects pollute extrinsics.

After §7 this step is no longer mandatory, but it is nowhere near deletable. The division of labour is clear:

- **Accuracy depends on conditions, not on the brand name.** COLMAP is strong on static scenes with enough texture and overlap; but **monocular SfM is itself up-to-scale** — absolute scale has to come from a calibration target, a known baseline or another sensor, and COLMAP does not supply it.
- **Benchmarks do not share one GT convention.** Some reconstruction benchmarks do derive their poses from COLMAP; others use sensor ground truth or synthetic data. Match whichever convention the baseline you want to compare against used, rather than asserting that "the benchmark is COLMAP".
- **Feed-forward models deliver seconds instead of hours**, and still produce results where COLMAP struggles: low overlap, few views, texture-less. Note that **unknown intrinsics are not a COLMAP failure condition** — it can self-calibrate, it just drifts more easily.
- **Connecting the two costs engineering.** **VGGT-X** (2509.25191) existing at all is the evidence: feeding VGGT's outputs into large-scale 3DGS training needed extra work on memory, point-cloud noise and pose accuracy before it ran — this is not merely "swap the pose source".

> 💡 **How to actually choose** — on capture conditions, time budget and **measured** accuracy: static, well-textured, offline, must match some existing benchmark's convention → COLMAP; online, few views or poor overlap → feed-forward; want both → feed-forward for initialization, then BA / COLMAP refinement. Treat scale as a separate question: neither DUSt3R-style training nor VGGT guarantees metres, and monocular COLMAP does not either; real scale needs a calibration target, a known baseline, or a metric model (§7.3) — and **a learned metric scale is not measurement-grade accuracy**.

### 12.2　Numerical stability (general NeRF/3DGS)

| Issue | Symptom | Fix |
| --- | --- | --- |
| Sigma blowup | Floaters fill the space | Use softplus or truncated $\sigma$; occupancy grid skip |
| Alpha saturation | 1-α underflow → T all 0 | `(1-α).clamp(min=1e-10)` or log-space cumprod |
| Gaussian degeneration | Extremely small scale / extreme anisotropy | Clamp scale lower bound; regularize anisotropy |
| Densify explosion | Gaussian count skyrockets to memory limit | Add max gaussian count; periodic prune; reset opacity |
| SDS Janus | Faces / heads in multiple views | Add view-conditioning ("front view" / "back view"); MVDream |
| SDS over-saturation | Saturated colors | Lower CFG; switch to VSD; or negative prompt |

### 12.3　Multi-machine distributed & evaluation metrics

**Distributed**: NeRF / Instant-NGP / 3DGS are single-GPU standard; large 3DGS scenes use chunk-wise (VastGaussian, CityGaussian); SDS/VSD runs 2 SD forwards per iter, 8×A100 gives significant speedup; TRELLIS / Hunyuan3D training is large-scale multi-node DDP.

| Evaluation metric | Use | Algorithm |
| --- | --- | --- |
| **PSNR / SSIM / LPIPS** | View synthesis (reconstruction) | Compare with real views |
| **Chamfer Distance** | Mesh geometry | Average nearest-neighbor distance between two point clouds |
| **F-Score (3D)** | Mesh / point | Precision + recall under threshold |
| **CLIP Score / CLIP-R-Prec** | Text-to-3D alignment | Render → CLIP similarity / distinguish distractor prompts |
| **ULIP / ULIP-2 alignment score** | 3D ↔ image ↔ text semantic alignment | Similarity in a tri-modal shared space (ULIP 2212.05171 / ULIP-2 2305.08275; ULIP-2 auto-generates language descriptions with a large model, removing manual annotation) |
| **User study** | Final quality | MTurk / lab-internal |

**Commonly used evaluation sets**: **GSO** (Google Scanned Objects, 2204.11918), 1000+ real scanned objects, the most common evaluation set for single-image-to-3D; **Toys4K** (2101.07296), ~4K objects across 105 categories, good category diversity, common for few-shot and part-level experiments; **Objaverse-XL** (2307.05663), 10M+ objects, treated by most work as a training source. **A dataset does not carry a "train" or "test" identity of its own** — the same assets are training data in one paper and evaluation data in another, so when you see a score, ask how the split was drawn; the generalization claim holds only on that split.

## §13 25 frequently-asked interview questions

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

<summary>Q15. Feed-forward 3DGS vs per-scene 3DGS optimization: when is 800× not worth it?</summary>

- **What each line wins on**: feed-forward (GS-LRM, 2-4 posed views, 0.23 s / Long-LRM, 32 views at 960×540, ~1 s) wins on latency and robustness under sparse views; per-scene optimization (§4, 10-30 min) wins by squeezing every observation of this one scene. **Which is higher quality is conditional** — Long-LRM reports parity or better than per-scene optimization in its room-scale setting, while optimization usually stays ahead when views are dense

- **Not worth it, case one: dense views, offline** — tens to hundreds of images and no deadline: optimization keeps converging, while a feed-forward model's prior invents detail wherever observations are thin

- **Not worth it, case two: outside the training distribution** — these models are trained mostly on object- and room-scale data, so capture conditions they never saw (scale, materials, lighting) become guesswork, whereas optimization only trusts the images in front of it. Note that "outdoor" is not automatically out-of-distribution; it depends on what the specific model was trained on

- **Cases where it is worth it**: interactive / online use; very few views (2-8), where optimization is itself under-constrained and the prior is a net gain; batch-processing thousands of scenes, where total throughput beats per-scene fidelity

- **What industry actually does**: feed-forward initialization plus optimization refinement — the Gaussians a feed-forward model emits are ordinary Gaussians and **can keep training**, so this removes optimization's slow cold start

- **It replaces the optimization, not the representation**: the output is still Gaussians, still rasterized per §4.3

Treating 800× as a universal factor — it is Long-LRM's speedup over per-scene optimization in its own setting (32 views at 960×540, A100); change the view count, resolution or hardware and the number changes.

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

<summary>Q19. You just captured a new dataset — COLMAP or VGGT?</summary>

- **Ask four things first**: capture conditions (texture, overlap, moving objects), how fast you need results, whether downstream needs absolute scale, and whether you must match some existing benchmark's pose convention

- **COLMAP**: mature and controllable, accurate on static scenes with enough texture and overlap; failure modes cluster around texture-less / strongly specular / dynamic / low-overlap input. **Two caveats**: unknown intrinsics do not automatically break it (it can self-calibrate, it just drifts more easily); and it **does not always fail visibly** — every image can register with a decent reprojection error and the poses can still be wrong, so the consistency checks are needed either way

- **The VGGT family** (§7): seconds, no dependence on pre-calibrated intrinsics, far more tolerant of thin overlap. Its failure mode is **always producing output with no accuracy guarantee**, which also has to be verified yourself

- **Scale**: DUSt3R-style training (two-sided prediction + GT normalization) is up-to-scale by default and VGGT normalizes only the GT — neither guarantees metres. **Monocular SfM is up-to-scale too**: COLMAP does not supply absolute scale on its own. Real scale comes from a calibration target, a known-baseline stereo/multi-sensor rig, or a model like MapAnything (2509.13414) that predicts a global scale — but **a learned metric scale is not the same thing as measurement-grade accuracy**

- **Benchmark conventions**: some reconstruction benchmarks do derive their poses from COLMAP; others use sensor ground truth or synthetic data. Match whichever convention the baseline you want to compare against used

- **The combined answer**: feed-forward for initialization → BA / COLMAP for refinement; or run VGGT first to see whether structure emerges at all, then decide whether COLMAP's hours are worth waiting for

Answering "VGGT has fully replaced COLMAP", or the reverse, "COLMAP is always more accurate". Choose on capture conditions, speed budget and **measured** accuracy. VGGT-X (2509.25191) is the concrete reminder: feeding VGGT outputs into large-scale 3DGS took extra engineering — the handoff is not free.

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

<summary>Q21. Derive DUSt3R's confidence-weighted loss and explain the $-\alpha\log C$ term.</summary>

- **Output convention**: both heads output pointmaps $X^{1,1}, X^{2,1}$, **both in the camera frame of $I^1$**. Pose comes from solving PnP on 2D-3D correspondences between $X^{2,1}$ and $I^2$'s pixels — **not from "registering the two pointmaps"**, which are already in the same frame

- **Regression term**: for valid pixels $i\in\mathcal{D}^v$, $\ell_{\text{regr}}(v,i)=\lVert \tfrac{1}{z}X^{v,1}_i-\tfrac{1}{\bar z}\bar X^{v,1}_i\rVert$ — a Euclidean norm, **not a squared error**

- **Scale normalization**: $z=\text{norm}(X^1,X^2)=\frac{1}{\lvert\mathcal{D}^1\rvert+\lvert\mathcal{D}^2\rvert}\sum_v\sum_{i}\lVert X^v_i\rVert$, the mean distance of all valid points to the origin, computed over **both images jointly**. Dividing by it makes the loss invariant to global scaling, so DUSt3R-style training is up-to-scale. Absolute scale is obtained differently, and MapAnything is one way: it keeps this scale-invariant supervision and **separately predicts a global scale**. Normalisation is a separate matter — VGGT normalizes only the GT and still learns the normalised scale

- **Confidence form**: $\mathcal{L}_{\text{conf}}=\sum_v\sum_i\big[\,C^{v,1}_i\,\ell_{\text{regr}}(v,i)-\alpha\log C^{v,1}_i\,\big]$ with $C=1+\exp(\check C)\gt 1$. The paper calls $-\alpha\log C$ a **regularization term**

- **Without the $\log$ term**: $f(C)=C\ell$ is non-decreasing in $C$, so the optimum pushes $C$ to the lower end of the domain, $C\to 1$, and **the per-pixel loss falls back to the plain regression loss $\ell$**. Geometry is still learned; what degenerates is the confidence head, which becomes a constant carrying no information

- **With the $\log$ term, take the infimum over $C$** (an optimization elimination, not a probabilistic marginalization): $f'(C)=\ell-\alpha/C$, so for $\ell\lt \alpha$ the stationary point $C^\star=\alpha/\ell\gt 1$ lies inside the domain, while for $\ell\ge\alpha$ we have $f'\gt 0$ and the infimum is still at $C\to1^+$:

  $$\inf_{C\gt 1}f(C)=\begin{cases}\alpha+\alpha\log(\ell/\alpha), & 0\lt \ell\lt \alpha\\[2pt] \ell, & \ell\ge\alpha\end{cases}$$

- **State the conclusion precisely**: only the $\ell\lt \alpha$ branch is reshaped into a logarithm; **large residuals with $\ell\ge\alpha$ are still counted linearly** — this is not a robust loss that suppresses outliers. $\alpha$ is the boundary: raising $C$ only pays once the residual drops below $\alpha$ (buying a negative $\alpha\log(\ell/\alpha)$), so $C$ learns "can I get this pixel right"; on sky, specularities and moving objects, where it cannot, $C$ sits near 1 and earns no reward

Answering "marginalizing out confidence gives a logarithmic robust loss": first, this is an infimum-based optimization elimination, not an integral marginalization; second, only the $\ell\lt \alpha$ half is logarithmic, the large-residual half stays linear. The other way to lose points is claiming "without the log term geometry stops being learned" — without it $C\to 1$ and the loss is exactly the plain regression loss.

</details>

<details>

<summary>Q22. VGGT predicts four quantities from one backbone — why does joint training beat specialists, and what does DA3's "one depth-ray target is enough" say about that?</summary>

- **Describe the four heads precisely**: camera (extrinsics + intrinsics), depth (dense depth per image), point map (3D points in the common frame), track (**given query pixels, the 2D position of the same physical point in every image** — 2D correspondence, not "cross-view 3D trajectories")

- **The case for joint training**: all four targets share one geometry — depth plus camera yields the point cloud, and the points' 2D projections across images are the tracks. They constrain each other, forcing the backbone into a self-consistent 3D representation; a single-task model has no other head to expose its inconsistencies

- **VGGT's own observation**: the paper reports that **at inference, combining the depth head with the camera head gives a more accurate point cloud than the point map head's direct output**. This is an empirical observation and **does not by itself establish that the multi-task benefit lives in the shared representation rather than in that head**; its safe reading is that which supervision targets you train on and which inference path you take are separable decisions

- **DA3's subtraction (2511.10647)**: no four heads, no bespoke architecture — **a plain transformer with a single depth-ray prediction target**, reporting average relative improvements over VGGT on the authors' own benchmark: 44.3% on pose, 25.1% on geometry

- **How to read that comparison**: the two papers differ in data, training scale and evaluation set, so **a cross-paper score gap cannot prove which supervision mechanism caused the gain**. What holds up is weaker and still useful: there exists at least one configuration where a single prediction target suffices for both pose and geometry, so head count is not a necessary condition — causal claims need same-paper ablations

- **One extra point worth making**: depth + ray already span pose and point cloud informationally, so extra heads mostly supervise the same information in pieces, at the cost of more architecture and loss-weight tuning surface

Stopping at "multi-task = more supervision = better"; or, in the other direction, citing DA3's scores as proof that multi-task is useless — that is a cross-paper comparison and cannot carry a causal conclusion.

</details>

<details>

<summary>Q23. The token budget of AR mesh generation: what do AMT / BPT / tree sequencing each compress, and why is maximal compression still not enough?</summary>

- **The baseline**: the naive sequence is 3 vertices × 3 coordinates = **9 coordinate tokens per face**. Sequence length = face count × tokens per face, transformers are quadratic, so token count converts directly into "how many faces can be generated at all"

- **MeshGPT** (2311.15475): graph-convolutional encoding plus **residual quantization** compresses each face into **6 codebook tokens** (not 9 — 9 is the naive baseline it set out to beat)

- **AMT** (MeshAnything V2, 2408.02555) compresses **vertex repetition**: adjacent faces share vertices, so most faces encode **one new vertex** instead of three → about **half** the naive sequence length

- **BPT** (2411.07025) compresses **coordinate representation**: **block indexing** (coordinate split into block id + in-block offset) plus **vertex sharing within a patch**, shortening the **naive sequence** by **~75%** and pushing faces past **8K**

- **TreeMeshGPT** (2503.11629) compresses **face ordering freedom**: autoregression along an adjacency tree, popping an edge from a stack of expandable edges and growing a face outward. The compressed length is **about 22% of the original** (≈2 tokens/face) — "22% remaining", not "22% fewer". Its construction also constrains orientation, **markedly reducing normal flips**, without fully eliminating post-processing

- **Do the arithmetic for 64K faces (where L3 follow-ups land)**: naive at 9/face → ~**576K** tokens; AMT at about half → ~**288K**; tree sequencing at ≈2/face → ~**128K**. **Even the best is still six figures**, beyond ordinary context lengths, and quadratic attention cannot carry it

- **Which is why Meshtron (2412.09548) takes a different route**: rather than expecting compression to make the sequence "fit", it makes a six-figure sequence trainable — an **hourglass architecture** downsamples tokens in the middle and **sliding-window attention** truncates the quadratic cost, reaching **64K faces @ 1024 coordinate resolution**

- **Nexus's critique (2607.13563) targets autoregression itself**: token-by-token generation lets earlier errors propagate down the sequence, and compression only shortens the sequence without removing that path. It switches to **coarse-to-fine vertex octree diffusion** to fix the vertex set, then **topology embeddings** to recover edges and faces, so there is no per-token propagation; the cost, which the paper states, is **slow inference**

Reading "22%" as "22% fewer" (it means 22% remaining), or describing MeshGPT as 9 tokens per face (that is its baseline). The other way to lose points: treating compression as the endpoint — applied individually, the best of these still lands at six-figure token counts (the compression rates do not multiply), and Meshtron's answer is architecture, not a fiercer tokenizer.

</details>

<details>

<summary>Q24. SLAT / VecSet / triplane: what is each one's actual architectural trade-off?</summary>

- **First, dismantle a common wrong answer**: all three latents **can be decoded by coordinate query** — triplane by projecting onto three planes, bilinear interpolation and element-wise sum (§8.3); VecSet by cross-attending from the query coordinate into the latent set; SLAT by indexing the voxel position. So "how many formats it can emit" follows from **which decoders each group trained**, not from a prohibition in the representation

- **SLAT** (TRELLIS, 2412.01506): $\boldsymbol z=\{(\boldsymbol z_i,\boldsymbol p_i)\}$, voxel resolution $N=64$, $L\approx 20\text{K}$ active voxels (~8% occupancy). $\boldsymbol p_i$ supplies only the **sparse support**; fine geometry and appearance both live in $\boldsymbol z_i$. An explicit support makes local computation and position-matched decoding most natural, which is why TRELLIS trained 3DGS / field / mesh decoders; the price is an extra "generate the structure" stage

- **VecSet** (3DShape2VecSet / Hunyuan3D ShapeVAE / Dora / Step1X-3D): **an unordered latent set with no spatial index**. The upside is that the set can be tiny — Dora (2412.17808) uses Sharp Edge Sampling plus dual cross-attention to reach **1,280 codes**, matching the dense XCube-VAE on Dora-bench (which needs >10,000) — and a short sequence makes the downstream DiT cheap. The price is losing locality: the latent carries no "which part of space is this" structure to exploit

- **Triplane** (EG3D / Direct3D 2405.14832 / LRM): three **dense** 2D planes, whose biggest advantage is reusing the whole 2D conv and attention toolchain; the price is plane memory growing with resolution, plus axis-aligned projection ambiguity (thin elongated structures overlap on some plane)

- **The decoding target is its own axis**: CLAY emits an occupancy field, Hunyuan3D-DiT and Dora emit implicit fields, Step1X-3D emits a **TSDF**, and TRELLIS goes through FlexiCubes to a mesh. Same VecSet family, completely different decoding targets

- **The correct statement about sparse attention**: **dense tokens can run windowed / sparse attention too**. Direct3D-S2's (2505.17412) gain comes from a sparse support making the skipped computation genuinely nonexistent; its reported 3.9× / 9.6× figures are **the SSA operator against FlashAttention-2**, not an end-to-end model speedup

Saying "VecSet can only produce meshes" or "sparse attention is meaningless for triplane" — neither holds; the first is a decoder choice, and dense tokens can be attended sparsely. The other way to lose points is calling SLAT "just a voxel grid": voxels are only $\boldsymbol p_i$, while $\boldsymbol z_i$ comes from projecting active voxels into multi-view DINOv2 feature maps, sampling, and compressing.

</details>

<details>

<summary>Q25. What failure of reference-view anchoring does π³'s permutation equivariance fix, and how is NoPoSplat's canonical frame different?</summary>

- **What reference-view anchoring is**: DUSt3R places all pointmaps in the camera frame of $I^1$; the VGGT family likewise nominates one image as the common coordinate origin

- **What is wrong with it**: (1) the result depends on which image was picked — reorder the same images and the output changes, which is variance that should not exist; (2) when the reference is degenerate, it is still treated as the global datum. **Do not invent an "error accumulates with distance to the reference" story**: models like VGGT predict all views jointly, so there is no chained hop-by-hop path

- **What π³'s (2507.13347) permutation equivariance actually is**: $f(PX)=P f(X)$ — a constraint on **permutation behaviour**. What it deterministically removes is **dependence on reference choice and input ordering**; it is **not the same as** being insensitive to a bad view. The robustness gains π³ reports come from its experiments, not from equivariance by derivation

- **When comparing two reconstructions in different reference frames**: align the coordinate transform first, then compare geometry — otherwise you are measuring a change of frame, not a change in quality

- **NoPoSplat (2410.24207) deliberately keeps the anchor**: it **defines the first input image's camera frame as canonical space**, emits all Gaussians there, and injects intrinsics as tokens. Its "pose-free" claim is scoped to the Gaussian network itself

- **Why this is not a contradiction — they cure different diseases**: π³ wants **independence from any particular view** (geometric reconstruction, where views have no natural hierarchy); NoPoSplat wants **independence from the pose-estimation step** (under sparse views, estimating pose then reconstructing propagates error, so a fixed convention replaces it)

- **L3 follow-up: canonical frame and scale are two different things, and three distinctions settle it**: (1) **choosing a reference frame fixes only the origin and axes** and says nothing about scale; (2) **known intrinsics do not remove the global scale freedom** — with an unknown baseline, scaling the scene and the baseline together produces identical images, so scale is unobservable; (3) NoPoSplat's novel-view evaluation is **not "align the scale first"** — it **fixes the Gaussians and optimizes the target camera poses**. Relatedly, AnySplat's (2505.23716) joint intrinsics/extrinsics prediction solves calibration and registration — it **does not yield absolute scale either**

Treating "pose-free" and "reference-free" as synonyms: NoPoSplat is pose-free but **not** reference-free — its canonical frame is precisely a privileged reference view.

</details>

## §A Appendix: full code skeleton + references

### A.1　Complete from-scratch code includes

`volume_render()` (NeRF α-compositing with numerical stability) · `positional_encoding()` (γ(p) Fourier features) · `gaussian_splat_forward()` (3DGS pedagogical forward + projection Jacobian) · `densify_and_prune()` (3DGS densification heuristics) · `sds_loss()` (SDS gradient surrogate) · `dust3r_conf_loss()` (DUSt3R confidence-weighted pointmap loss, with scale normalization and a metric switch) · `marching_cubes_sketch()` (mesh extraction interface using scikit-image).

### A.2　Key papers reading list

- **NeRF family**: Mildenhall 2020 ECCV (HM); Müller **Instant-NGP** SIGGRAPH 2022 Best; Barron **Mip-NeRF** / **360** ICCV 2021 / CVPR 2022; Wang **NeuS** + Yariv **VolSDF** NeurIPS 2021; Fridovich-Keil **Plenoxels** CVPR 2022; Chen **TensoRF** ECCV 2022.
- **3DGS family**: Kerbl **3D Gaussian Splatting** SIGGRAPH 2023 Best; Huang **2D Gaussian Splatting** SIGGRAPH 2024; Luiten **Dynamic 3DGS** 3DV 2024; Wu **4DGS** CVPR 2024; Guédon **SuGaR** CVPR 2024.
- **Mesh / SDF**: Shen **DMTet** NeurIPS 2021 / **FlexiCubes** SIGGRAPH 2023.
- **SDS family**: Poole **DreamFusion** arXiv 2022.09 → ICLR 2023 Outstanding; Wang **ProlificDreamer (VSD)** NeurIPS 2023 Spotlight; Lin **Magic3D** CVPR 2023; Chen **Fantasia3D** ICCV 2023; Tang **DreamGaussian** ICLR 2024; Yi **GaussianDreamer** CVPR 2024.
- **Single-image 3D**: Liu **Zero-1-to-3** ICCV 2023 / **One-2-3-45** NeurIPS 2023 / **SyncDreamer** ICLR 2024; Shi **Zero-1-to-3++** arXiv 2023 / **MVDream** ICLR 2024; Hong **LRM** arXiv 2023.11 → ICLR 2024; Tochilkin **TripoSR** arXiv 2024; Xu **InstantMesh** arXiv 2024; Boss **Stable Fast 3D** arXiv 2024.
- **3D Foundation Models**: Xiang **TRELLIS** 2412.01506 (Microsoft) / **TRELLIS.2** 2512.14692; Tencent **Hunyuan3D 2.0** 2501.12202 / **2.1** 2506.15442 / **2.5** 2506.16504 / **Omni** 2509.21245 / **Studio** 2509.12815 / **Buffalo 1.0** 2608.02711; Zhang **CLAY** SIGGRAPH 2024 (2406.13897); **Direct3D** 2405.14832 / **Direct3D-S2** 2505.17412; **Dora** 2412.17808 (CVPR 2025); **Step1X-3D** 2505.07747; Meta **AssetGen 2** 2605.26137.
- **Feed-forward reconstruction**: **DUSt3R** 2312.14132 (CVPR 2024) / **MASt3R** 2406.09756; **VGGT** 2503.11651 (CVPR 2025 Best Paper); **Fast3R** 2501.13928 (CVPR 2025); **CUT3R** 2501.12387; **π³** 2507.13347; **MapAnything** 2509.13414 (3DV 2026); **Depth Anything 3** 2511.10647; **VGGT-Ω** 2605.15195 (CVPR 2026 Oral); **MV-DUSt3R+** 2412.06974; **VGGT-X** 2509.25191; survey 2507.08448.
- **Feed-forward 3DGS**: **LGM** 2402.05054; **GS-LRM** 2404.19702; **Long-LRM** 2410.12781 / **Long-LRM++** 2512.10267; **DepthSplat** 2410.13862 (CVPR 2025); **NoPoSplat** 2410.24207; **AnySplat** 2505.23716.
- **Native mesh generation**: **MeshGPT** 2311.15475; **MeshAnything** 2406.10163 / **V2** 2408.02555; **BPT** 2411.07025; **TreeMeshGPT** 2503.11629 (CVPR 2025); **Meshtron** 2412.09548; **DeepMesh** 2503.15265; **Nexus** 2607.13563.
- **Part-level / sim-ready**: **PartGen** 2412.18608; **HoloPart** 2504.07943; **PartCrafter** 2506.05573; **PartPacker** 2506.09980 (NVIDIA); **EmbodiedGen** 2506.10600; **Artiverse** 2605.24403.
- **Datasets / evaluation**: **Objaverse-XL** 2307.05663 (10M+); **GSO** 2204.11918 (real scanned objects); **Toys4K** 2101.07296 (~4K objects / 105 categories); **ULIP** 2212.05171 / **ULIP-2** 2305.08275 (3D-image-text alignment).

### A.3　Common Embodied AI / AR / VR follow-ups

3DGS connected to a physics engine → first extract mesh via 2DGS / SuGaR → IsaacSim / MuJoCo; dynamic NeRF → 4DGS / D-NeRF / K-Planes; real-time AR 3DGS → mobile-friendly (PostShot, Luma) + pruning / quantization; insufficient 3D data → Objaverse-XL (TRELLIS), 2D distillation (DreamFusion family), or multi-view heuristics (MVDream); a fresh capture to reconstruct → run VGGT first for structure, then decide whether COLMAP is worth waiting for (§12.1); a generated asset heading into a simulator → mesh alone is not enough, parts, joints and physical materials are still missing (§9.6).

---

**3D Generation Quick Reference** · Main references: Mildenhall 2020 (NeRF), Müller 2022 (Instant-NGP), Kerbl 2023 (3DGS), Poole 2022/ICLR 2023 (DreamFusion), Wang 2023 (VSD), Wang 2024 (DUSt3R), Wang 2025 (VGGT), Xiang 2024 (TRELLIS), Tencent 2025-26 (Hunyuan3D 2.x / Omni / Buffalo). Covers: NeRF volume-rendering derivation, Instant-NGP hash grid, 3DGS projection Jacobian, SDS / VSD gradient derivation, DUSt3R's confidence-weighted loss, feed-forward reconstruction and feed-forward 3DGS, single-image 3D, the latent-representation axis across 3D foundation models, native mesh generation, and part-level / sim-ready assets. Essential for Embodied AI / AR / VR.
