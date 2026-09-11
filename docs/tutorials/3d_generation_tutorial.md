## §0 TL;DR Cheat Sheet

> 💡 **11 句话搞定 3D Generation** — Embodied AI / AR / VR 面试核心要点（详见后文 §1–§12 推导）。

1. **三大表示**：**NeRF**（隐式神经场 + 体渲染）、**3DGS**（显式 Gaussian 点云 + 光栅化）、**Mesh / SDF**（显式表面 / 隐式距离场）。重建质量与速度的 sweet spot：3DGS（Kerbl 2023 SIGGRAPH Best Paper）。

2. **NeRF 核心公式**：$C(\mathbf{r}) = \int_{t_n}^{t_f} T(t)\sigma(\mathbf{r}(t))\mathbf{c}(\mathbf{r}(t),\mathbf{d})\,dt$，其中 $T(t) = \exp\!\left(-\int_{t_n}^{t}\sigma(\mathbf{r}(s))\,ds\right)$ 是 transmittance。离散化得到 $\alpha$-compositing：$C \approx \sum_i T_i (1-e^{-\sigma_i\delta_i})\mathbf{c}_i$。

3. **Instant-NGP** (Müller 2022 SIGGRAPH)：**多分辨率 hash 网格** + tiny MLP，约 4+ 个数量级加速；hash collision 由 MLP 在含碰撞表项上自动学习消歧（被 loss + 多尺度冗余共同压制）。

4. **3DGS 核心**：场景表示为一组 3D Gaussian $\{\mu_i, \Sigma_i, \alpha_i, c_i(\mathbf{d})\}$，**可微光栅化**通过将 3D 协方差用 Jacobian $J$ 投影到 2D：$\Sigma' = J W \Sigma W^\top J^\top$，按深度排序后做 front-to-back alpha-blending。

5. **DreamFusion SDS** (Poole et al. 2022 arXiv → ICLR 2023 Outstanding)：用 pretrained 2D diffusion 监督 3D 表示：$\nabla_\theta \mathcal{L}_\text{SDS} = \mathbb{E}_{t,\epsilon}[w(t)(\epsilon_\phi(x_t;y,t)-\epsilon)\,\partial x/\partial \theta]$，**故意去掉 U-Net 对 $x_t$ 求导的 Jacobian 项**，使得训练 simulation-free。代价：mode-seeking → over-saturation / Janus。

6. **VSD** (Wang 2023 NeurIPS, ProlificDreamer)：把 3D 参数 $\theta$ 视为 random variable $\mu(\theta)$，**最小化的是渲染加噪图像分布**之间的 KL：$\mathbb{E}_t\big[D_\text{KL}\big(q_\mu^t(x_t|y)\,\|\,p_\phi^t(x_t|y)\big)\big]$。梯度形式为 **relative score** $\nabla_\theta \approx (\epsilon_\phi(x_t;y,t) - \epsilon_\psi(x_t;y,t,\pi))\,\partial x/\partial\theta$，其中 $\epsilon_\psi$ 是 LoRA 微调的辅助 score。CFG 可从 100 降至 7.5。

7. **Single-image / Few-view 3D**：Zero-1-to-3 (Liu 2023 ICCV) 用 viewpoint conditioned diffusion；SyncDreamer / MVDream 学多视图联合一致性；TripoSR / InstantMesh / Stable Fast 3D 把 image-to-mesh 推到秒级（TripoSR ~0.5 秒、InstantMesh ~10 秒）。

8. **Feed-forward 重建**：**DUSt3R** (arXiv:2312.14132) 把 SfM 换成一次前向——直接回归 pointmap，两张图的 3D 点都落在第一张图的相机系里，**位姿成了输出而不是输入**；损失除以有效点到原点的平均距离，所以默认 up-to-scale。**VGGT** (2503.11651, CVPR 2025 Best Paper) 一个 backbone 引出四个 head（相机 / depth / point map / track）。COLMAP 由此从"必经"降级为"精度基准"（§7、§12.1）。

9. **3D Foundation Models (2024-26 开源)**：**TRELLIS** 的 **SLAT** = 稀疏体素坐标 + per-voxel latent（$N=64$，$L\approx 20\text{K}$ active voxel），两阶段 rectified flow，一份 latent 解码成 3DGS / 场 / mesh；**Hunyuan3D** 沿 2.0 → 2.1 → 2.5 → Omni / Studio / Buffalo 迭代（**没有 3.0**），shape→texture 两阶段；**CLAY** (arXiv:2406.13897) 多分辨率 VAE + latent DiT。

10. **原生 mesh 与 latent 表示这条轴**：无论是从场里提等值面（marching cubes、FlexiCubes）还是像 TRELLIS.2 那样从 O-Voxel 直接转 mesh，**这些输出都不保证具有适合编辑、绑骨和形变的边流**——这才是美术不收货的原因。**MeshGPT → MeshAnything V2 (AMT) → BPT → TreeMeshGPT** 这条线在压 token（基线是朴素序列的 9 token / 面），**Meshtron 则换成用架构扛长序列**（hourglass + 滑动窗口）。latent 表示（SLAT / 稀疏体素 / VecSet / triplane）都能按坐标查询解码，它决定的不是"能出几种格式"，而是**计算与显存摆在哪里**。

11. **Embodied AI 关键应用**：Sim2Real 资产生成、NeRF/3DGS 作为可微 simulator、language-conditioned 3D affordance。**面试常见交叉**：NeRF SLAM、Gaussian-Splat scene editing、3D 物理一致性。

## §1 三大表示的直觉对比

3D 生成的第一选择题是 **representation**——选错了下游全废。

|  | NeRF (Implicit Field) | 3DGS (Explicit Point) | Mesh / SDF |
| --- | --- | --- | --- |
| **存储** | MLP 权重 $f_\theta(\mathbf{x},\mathbf{d}) \to (\sigma, \mathbf{c})$ | 一堆 3D Gaussian $\{\mu_i, \Sigma_i, \alpha_i, c_i\}$ | 三角网格 / signed distance |
| **渲染** | Ray marching + 体积分（GPU 数百 ms/帧） | Differentiable rasterization（GPU 数 ms/帧）| Rasterization（实时） |
| **训练** | 数百视图，数小时 (vanilla) | 数十视图，10-30 分钟 | 需要 mesh + texture optim |
| **质量** | 视图合成 SOTA | 与 NeRF 持平甚至更好（PSNR 高） | 受多边形分辨率制约 |
| **编辑** | 困难（神经场不可解释） | 容易（点可移动、删除、合并） | 容易（标准 DCC 流程） |
| **导出 mesh** | 难（需 NeuS / Poisson）| 中等（2DGS / GSDF / SuGaR） | 自身就是 mesh |
| **下游适配** | 物理仿真不友好 | 需先转 mesh（无碰撞体/材质/物理参数）；标准 robot/物理仿真管线仍以 mesh 为主 | 标准 robot / AR/VR pipeline |

> 💡 **面试直觉** — Embodied AI 倾向 mesh / 3DGS（仿真器友好）；AR/VR 看场景规模（前景小物 mesh，大场景 3DGS）；视觉重建 SOTA 用 3DGS。NeRF 现在偏 research baseline，工业落地以 3DGS 为主。

## §2 NeRF：体渲染原理推导（必考）

### 2.1　连续体渲染公式

NeRF (Mildenhall 2020 ECCV **Best Paper Honorable Mention**) 把场景表示为 **5D 神经场** $f_\theta : (\mathbf{x}, \mathbf{d}) \to (\sigma, \mathbf{c})$：

- 输入：3D 位置 $\mathbf{x} \in \mathbb{R}^3$ + 视角方向 $\mathbf{d} \in \mathbb{S}^2$
- 输出：体密度 $\sigma \ge 0$（与方向无关）+ 颜色 $\mathbf{c} \in \mathbb{R}^3$（依赖方向，捕捉镜面反射）

对相机射线 $\mathbf{r}(t) = \mathbf{o} + t\mathbf{d}$，沿 $t \in [t_n, t_f]$ 体积分得到像素颜色：

$$\boxed{\;C(\mathbf{r}) = \int_{t_n}^{t_f} T(t)\,\sigma(\mathbf{r}(t))\,\mathbf{c}(\mathbf{r}(t),\mathbf{d})\,dt\;}$$

其中 **transmittance**（光线从 $t_n$ 到 $t$ 没被遮挡的概率）：

$$\boxed{\;T(t) = \exp\!\left(-\int_{t_n}^{t}\sigma(\mathbf{r}(s))\,ds\right)\;}$$

### 2.2　为什么是这个形式？— 从物理推导

考虑光线穿过参与介质（participating medium）。在 $[t, t+dt]$ 段内：

- 被吸收 / 散射出射线的概率：$\sigma(\mathbf{r}(t))\,dt$
- 介质在该点发射的颜色贡献：$\mathbf{c}(\mathbf{r}(t),\mathbf{d})$

令 $T(t)$ 为光线从 $t_n$ 到 $t$ 仍存活的概率。从 $t \to t + dt$，存活概率变化：

$$T(t+dt) = T(t)\,(1 - \sigma\,dt) \;\Rightarrow\; \frac{dT}{dt} = -\sigma(t)\,T(t)$$

这是一个一阶 ODE，初值 $T(t_n) = 1$，解出：

$$T(t) = \exp\!\left(-\int_{t_n}^{t}\sigma(\mathbf{r}(s))\,ds\right)$$

每个深度 $t$ 处贡献到像素的颜色 = **存活概率 × 该处吸收概率 × 该处颜色**：

$$dC = T(t)\,\sigma(t)\,\mathbf{c}(t)\,dt$$

积分即得 $C(\mathbf{r})$。

### 2.3　离散化：$\alpha$-compositing（**必考推导**）

实际无法做连续积分，把 $[t_n, t_f]$ 切成 $N$ 段，每段间距 $\delta_i = t_{i+1} - t_i$，假设段内 $\sigma, \mathbf{c}$ 为常数 $\sigma_i, \mathbf{c}_i$。

**段内 transmittance 衰减**：在 $[t_i, t_{i+1}]$ 上 $T$ 满足 $dT/dt = -\sigma_i T$，所以

$$\frac{T(t_{i+1})}{T(t_i)} = e^{-\sigma_i \delta_i}$$

由此**段间 transmittance**：

$$T_i := T(t_i) = \prod_{j=1}^{i-1} e^{-\sigma_j \delta_j} = \exp\!\Big(\!-\!\sum_{j=1}^{i-1}\sigma_j\delta_j\Big)$$

**段内颜色贡献**（积分而非简单矩形）：

$$\int_{t_i}^{t_{i+1}} T(t)\sigma_i\mathbf{c}_i\,dt = T_i\,\mathbf{c}_i \int_0^{\delta_i} \sigma_i e^{-\sigma_i s}\,ds = T_i\,\mathbf{c}_i\,(1 - e^{-\sigma_i\delta_i})$$

记 $\alpha_i := 1 - e^{-\sigma_i \delta_i}$（**段不透明度**），合并得 NeRF 离散公式：

$$\boxed{\;C(\mathbf{r}) \approx \sum_{i=1}^{N} T_i\,\alpha_i\,\mathbf{c}_i,\quad T_i = \prod_{j<i}(1-\alpha_j),\quad \alpha_i = 1 - e^{-\sigma_i\delta_i}\;}$$

这正是图形学 **front-to-back alpha-compositing** 公式。**关键**：$\alpha_i = 1 - e^{-\sigma_i\delta_i}$ 而非 $\sigma_i\delta_i$；当 $\sigma_i\delta_i$ 小时近似相等（一阶 Taylor），但大时差异显著（饱和到 1 vs 线性发散）。

> ✅ **物理一致性** — $\sigma \ge 0$ 与 $\alpha = 1 - e^{-\sigma\delta} \in [0, 1)$ 保证颜色合成 **永远在 [0, 1] 内**；且无论 ray 怎么穿，$\sum T_i\alpha_i \le 1$（剩余 $T_{N+1}$ 给背景）。

### 2.4　位置编码 $\gamma(p)$：表示高频细节

MLP 默认是低频偏置（NTK 分析），直接拟合 $f(\mathbf{x})$ 会糊。NeRF 用 **positional encoding** 提频：

$$\gamma(p) = \big(\sin(2^0\pi p),\cos(2^0\pi p),\sin(2^1\pi p),\cos(2^1\pi p),\dots,\sin(2^{L-1}\pi p),\cos(2^{L-1}\pi p)\big)$$

对 $\mathbf{x}$ 用 $L=10$（60 维），对 $\mathbf{d}$ 用 $L=4$（24 维）。后续 Tancik et al. 2020 "Fourier Features" 给了 NTK 解释：高频 $\sin/\cos$ 让 kernel 衰减更慢，使 MLP 可学高频。

### 2.5　Hierarchical Sampling：粗 → 细

- **Coarse 网络**：均匀采样 $N_c = 64$ 点，渲染得到 weights $w_i = T_i \alpha_i$
- **Fine 网络**：归一化 $w$ 为 PDF，按重要性采 $N_f = 128$ 新点（**重要性采样**：表面附近权重大，应密集采样）
- 用 coarse + fine 总共 $N_c + N_f$ 点合成最终颜色
- Loss：$\mathcal{L} = \|C_c - C_\text{gt}\|^2 + \|C_f - C_\text{gt}\|^2$（两个网络都监督，coarse 提供 sampler 的 well-defined 梯度）

### 2.6　NeRF 训练代码（核心 30 行）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def positional_encoding(x: torch.Tensor, L: int) -> torch.Tensor:
    """ x: [..., D];  返回 [..., D*2*L]，NeRF γ(p) 不含原始 x """
    freqs = 2.0 ** torch.arange(L, device=x.device, dtype=x.dtype) * torch.pi
    args = x.unsqueeze(-1) * freqs                # [..., D, L]
    pe = torch.stack([torch.sin(args), torch.cos(args)], dim=-1)  # [..., D, L, 2]
    return pe.flatten(-3)                          # [..., D*2*L]

def volume_render(sigma: torch.Tensor, color: torch.Tensor, t_vals: torch.Tensor,
                  ray_d: torch.Tensor):
    """
    NeRF 离散 α-compositing
        sigma:   [B, N]        体密度 (>= 0; 通常已经过 softplus / ReLU)
        color:   [B, N, 3]     颜色
        t_vals:  [B, N]        ray 上采样点的 t 值（单调递增）
        ray_d:   [B, 3]        ray 方向（用于把 Δt → 真实距离）
    返回 C: [B, 3], weights: [B, N], depth: [B]
    """
    # δ_i = t_{i+1} - t_i ；最后一段补 1e10（吸收掉远端到无穷的剩余）
    deltas = t_vals[..., 1:] - t_vals[..., :-1]
    delta_far = torch.full_like(deltas[..., :1], 1e10)
    deltas = torch.cat([deltas, delta_far], dim=-1)            # [B, N]
    deltas = deltas * torch.norm(ray_d[:, None, :], dim=-1)    # 把 t-间距换成真实欧氏距离

    alpha = 1.0 - torch.exp(-sigma * deltas)                   # [B, N]
    # T_i = ∏_{j<i} (1 - α_j) —— 用 cumprod，shift 一位让 T_1 = 1
    T = torch.cumprod(torch.cat([torch.ones_like(alpha[..., :1]),
                                 1.0 - alpha + 1e-10], dim=-1), dim=-1)[..., :-1]
    weights = T * alpha                                        # [B, N]
    C = (weights[..., None] * color).sum(dim=-2)               # [B, 3]
    depth = (weights * t_vals).sum(dim=-1)                     # [B]
    return C, weights, depth
```

> ⚠️ **数值陷阱** — `1 - alpha` 在 alpha 接近 1 时会下溢到 0，连乘后 T 整体被 zeroed；加 `+ 1e-10` 防止 backward 时 `log(0)`。最后一段 `δ → 1e10` 强行把背景的 transmittance 推到 0，否则不可见区域的 ray 颜色会受未采样段污染。

### 2.7　Mip-NeRF / Mip-NeRF 360（抗锯齿）

vanilla NeRF 在低分辨率 / 缩放下走样严重（同一像素对应不同尺度的 cone，但 NeRF 当作 ray 处理）。

- **Mip-NeRF** (Barron 2021 ICCV)：把 ray 视为 cone（视锥），用 **Integrated Positional Encoding (IPE)**——对 cone 段内的 PE 做闭式 Gaussian 期望 $\mathbb{E}_{\mathbf{x}\sim\mathcal{N}(\mu,\Sigma)}[\gamma(\mathbf{x})]$。对频率向量 $\boldsymbol{\omega} = 2^k\pi\,\mathbf{e}$ 而言，$\mathbb{E}_{\mathbf{x}\sim\mathcal{N}(\boldsymbol{\mu},\Sigma)}[\sin(\boldsymbol{\omega}^\top\mathbf{x})] = \sin(\boldsymbol{\omega}^\top\boldsymbol{\mu})\,e^{-\frac{1}{2}\boldsymbol{\omega}^\top\Sigma\,\boldsymbol{\omega}}$（标量情形即 $e^{-\frac{1}{2}\omega^2\sigma^2}$）；**高频系数被 cone 协方差 $\Sigma$ 通过 $e^{-\frac{1}{2}\boldsymbol{\omega}^\top\Sigma\boldsymbol{\omega}}$ 自动衰减**，自然实现 multi-scale。
- **Mip-NeRF 360** (Barron 2022 CVPR)：unbounded scene 用 contraction $f(x) = (2 - 1/\|x\|)\,x/\|x\|$ for $\|x\| > 1$，把无穷远压缩到 ball；加 distortion / proposal MLP 损失。

### 2.8　NeuS / VolSDF：体渲染 + SDF（导出 mesh 的关键）

NeRF 是 density-based，提取 mesh 需选 $\sigma$ 阈值（不稳）。**NeuS** (Wang 2021 NeurIPS) 用 **SDF $d(\mathbf{x})$** 替换 density：

$$\sigma(t) = \max\!\left(\frac{-\frac{d}{dt}\Phi_s(d(\mathbf{r}(t)))}{\Phi_s(d(\mathbf{r}(t)))},\; 0\right),\quad \Phi_s(d) = (1 + e^{-sd})^{-1}$$

其中 $\Phi_s$ 是 sigmoid，$s$ 是可学习"sharpness"。性质：表面处（$d=0$）权重峰值；可直接 Marching Cubes 提 mesh（mesh 是 $\{d = 0\}$）。

**VolSDF** (Yariv 2021 NeurIPS) 用 Laplace CDF $\sigma = \alpha\,\Phi(-d/\beta)$，思想类似。

## §3 Instant-NGP：约 4+ 个数量级加速（必考）

NeRF (vanilla) 训练一个场景要 1-2 天。**Instant-NGP** (Müller 2022 SIGGRAPH **Best Paper**) 5 秒就能拟合一个简单场景。

### 3.1　核心 idea：多分辨率 hash 网格

把"密集网格 vs 大 MLP"换成"**稀疏 hash 网格 + tiny MLP**"。

- $L$ 层分辨率（如 $L = 16$），第 $\ell$ 层格点数 $N_\ell = \lfloor N_\min \cdot b^\ell \rfloor$，几何级数（$b \approx 1.38$；论文典型取值 $N_\min = 16$，$N_\max \in [512, 2048]$ 视场景大小而定）
- 每层用 **hash function** 把格点坐标映到固定大小的特征表（$T = 2^{14}$–$2^{24}$，**典型 $T = 2^{19} = 524288$**）
- 查询点 $\mathbf{x}$：对每层做 8 角点三线性插值 → 拼成 $L \times F$ 维特征（$F = 2$）
- 喂给 **tiny MLP** ($2$ 层, hidden 64) 输出 $\sigma, \mathbf{c}$

### 3.2　Hash function

$$\text{hash}(\mathbf{x}) = \bigg(\bigoplus_{i=1}^{d} x_i \cdot \pi_i\bigg) \bmod T$$

$\pi_i$ 是互质的大常数（论文取 $\pi_1 = 1, \pi_2 = 2654435761, \pi_3 = 805459861$——注意 $\pi_1 = 1$ 并非质数，是论文有意设的，使第一维直接映射不打散）。 $\oplus$ 是 XOR。这是一种 **spatial hash**：常用于物理仿真的 BVH。

### 3.3　Hash collision 怎么消歧？（**L3 高频追问**）

当 $N_\ell^d > T$（fine level 必然发生），多个格点映到同一表项 → 冲突。Why does it still work?

1. **Multi-resolution 冗余**：粗 level 的特征 unique（$N_\ell^d \le T$），细 level 提供补充。MLP 可从粗特征恢复结构，细 level 只负责 detail。
2. **稀疏性优先**：大部分空间是空（NeRF 场景多数 voxel 是 background），有意义的 query 集中在表面附近，冲突的"有效格点对"很少。
3. **梯度自动消歧**：训练早期空白区采样点若密度非零，仍会通过体渲染 photometric loss 获得梯度（梯度幅度 $\propto \delta_i e^{-\sigma_i\delta_i}$，方向是把 $\sigma$ 压向 0）；随着 $\sigma\to 0$ 该梯度幅度自然衰减，且 occupancy grid 更新后会跳过该处后续采样，才真正不再产生梯度信号——不是"从一开始就得不到梯度"。
4. **MLP 后处理**：tiny MLP 在 $L \times F$ 拼接特征上学一个分类/回归，遇到冲突的表面点可用 **其他 level 不冲突的特征** disambiguate。

> 💡 **面试高分回答** — "Hash collision 看似破坏 unique 性，但**实际生效区域是稀疏的**（场景的 thin surface 仅占 voxel 总数极小比例），冲突区域大概率是无监督信号的 background；即便表面也有冲突，多分辨率层的非冲突特征 + tiny MLP 也能学到一致输出。这是个 **'lazy collision resolution'**：与其代价昂贵地搞 perfect hash，不如用冗余 + 数据驱动消歧。"

### 3.4　Instant-NGP 训练公式

参数：hash table $\theta_\text{hash} \in \mathbb{R}^{L \times T \times F}$ + MLP weights $\theta_\text{MLP}$。Loss 仍是 photometric MSE，但训练 5 秒 vs NeRF 1 天的差别来自：

- **Tiny MLP**：参数少 100×，前向快 ~50×
- **Hash 表**：稀疏激活，cache friendly
- **CUDA kernel fuse**：tiny-cuda-nn 把 forward + backward 融合
- **Occupancy grid**：粗占用网格 skip 空白区采样，避免无效 query

### 3.5　Plenoxels / TensoRF（同期的 explicit 派）

**Plenoxels** (Fridovich-Keil 2022 CVPR)：纯 voxel 网格 + 球谐 SH 系数 + density，**完全没 MLP**，直接梯度下降到 voxel；速度类似 Instant-NGP 但显存大。**TensoRF** (Chen 2022 ECCV)：把 3D grid field 用 **VM / CP 分解** 压缩，参数量从 $O(N^3)$ 降到 CP 的 $O(N)$ 或 VM 的 $O(N^2)$。

## §4 3D Gaussian Splatting：显式可微光栅化（**当前主力**）

**3DGS** (Kerbl 2023 SIGGRAPH **Best Paper**) 解决了 NeRF 的两大痛：渲染慢、editing 难。

### 4.1　场景表示

场景 = 一组 3D Gaussian $\{G_i\}$，每个：

- **均值** $\mu_i \in \mathbb{R}^3$（位置）
- **协方差** $\Sigma_i \in \mathbb{R}^{3\times 3}$（形状），分解为 $\Sigma = R S S^\top R^\top$（rotation $R$ + diag scaling $S$）
- **不透明度** $\alpha_i \in [0, 1]$
- **颜色** $c_i(\mathbf{d})$ 用 SH 系数（$\ell = 3$，每色 16 系数，共 48 参数）

为什么用 $R S S R^\top$ 而不直接学 $\Sigma$？— 要保证 $\Sigma$ 正定。直接学 $\Sigma$ 矩阵在梯度下会跑出半正定锥；分解后只需保证 $R$ 正交（用四元数 $q$ 参数化）+ $S$ 正（用 $\exp(s)$ 参数化），自然满足。

### 4.2　3D → 2D 投影 Jacobian（**L3 必考推导**）

把 3D Gaussian splat 到屏幕上做光栅化，需要把 3D 协方差 $\Sigma$ 投影到 2D 协方差 $\Sigma'$。

**Step 1**：World → Camera：刚体变换 $W \in SE(3)$。$\Sigma_\text{cam} = W \Sigma W^\top$（这里 $W$ 取旋转部分；平移不影响协方差）。

**Step 2**：Camera → Screen：透视投影**非线性**：

$$\pi(\mathbf{x}) = \begin{pmatrix} f_x\,x/z \\ f_y\,y/z \end{pmatrix}$$

非线性映射的协方差近似用一阶 Taylor。在均值 $\mu_\text{cam} = (x, y, z)$ 处求 Jacobian：

$$J = \frac{\partial \pi}{\partial \mathbf{x}}\bigg|_{\mu_\text{cam}} = \begin{pmatrix}\dfrac{f_x}{z} & 0 & -\dfrac{f_x\,x}{z^2}\\[2pt] 0 & \dfrac{f_y}{z} & -\dfrac{f_y\,y}{z^2}\end{pmatrix} \in \mathbb{R}^{2\times 3}$$

**Step 3**：2D 协方差（**核心公式**）：

$$\boxed{\;\Sigma' = J\,W\,\Sigma\,W^\top\,J^\top \in \mathbb{R}^{2\times 2}\;}$$

推导：若 $\mathbf{x} \sim \mathcal{N}(\mu, \Sigma)$，则一阶近似 $\pi(\mathbf{x}) \approx \pi(\mu) + J(\mathbf{x} - \mu)$，所以 $\text{Cov}[\pi(\mathbf{x})] \approx J\,\Sigma_\text{cam}\,J^\top = J W \Sigma W^\top J^\top$。这就是 EWA splatting (Zwicker 2001) 的经典推论。

### 4.3　可微光栅化：tile-based front-to-back alpha-blending

像素 $\mathbf{p}$ 的颜色：

$$C(\mathbf{p}) = \sum_{i \in \mathcal{N}(\mathbf{p})}\,c_i\,\alpha_i\,G_i'(\mathbf{p}) \prod_{j < i}\big(1 - \alpha_j\,G_j'(\mathbf{p})\big)$$

其中 $G_i'(\mathbf{p}) = \exp\!\big(-\tfrac{1}{2}(\mathbf{p} - \mu_i')^\top \Sigma_i'^{-1} (\mathbf{p} - \mu_i')\big)$ 是 2D Gaussian 在像素的值，$\mathcal{N}(\mathbf{p})$ 是覆盖 $\mathbf{p}$ 的 Gaussian 按深度排序。

**关键工程**：

1. **Tile 划分**：屏幕分成 $16\times 16$ tile，每 tile 内 Gaussian 按深度排序，并行渲染
2. **GPU sort**：用 radix sort，按 `(tile_id, depth)` 复合键
3. **Front-to-back early stop**：当累计 $\prod(1 - \alpha G') < 10^{-4}$ 时退出
4. **CUDA kernel**：作者公开 `diff-gaussian-rasterization`，前向 + 反向都是 manual derivative

### 4.4　3DGS 前向（PyTorch reference 实现）

```python
def quat_to_rot(q: torch.Tensor) -> torch.Tensor:
    """ q: [N, 4] (w, x, y, z) already normalized;  返回 R: [N, 3, 3] """
    w, x, y, z = q.unbind(-1)
    R = torch.stack([
        1 - 2*(y*y + z*z),   2*(x*y - w*z),     2*(x*z + w*y),
        2*(x*y + w*z),       1 - 2*(x*x + z*z), 2*(y*z - w*x),
        2*(x*z - w*y),       2*(y*z + w*x),     1 - 2*(x*x + y*y),
    ], dim=-1).reshape(-1, 3, 3)
    return R

def gaussian_splat_forward(
    means3D: torch.Tensor,        # [N, 3]  Gaussian 中心 (world)
    scales: torch.Tensor,          # [N, 3]  log-scale (取 exp 得真实 scale)
    quats: torch.Tensor,           # [N, 4]  四元数 (会归一化)
    opacities: torch.Tensor,       # [N, 1]  σ(logit) → α
    colors: torch.Tensor,          # [N, 3]  (这里简化为 RGB，不展 SH)
    viewmat: torch.Tensor,         # [4, 4]  world→camera
    K: torch.Tensor,               # [3, 3]  内参 (fx, fy, cx, cy)
    H: int, W: int,
):
    """ 教学版前向：不做 tile sort / CUDA，仅展示数学。
        实际生产用 gsplat / diff-gaussian-rasterization。 """
    N = means3D.shape[0]
    device = means3D.device

    # --- 1. World → Camera ---
    homo = torch.cat([means3D, torch.ones(N, 1, device=device)], dim=-1)
    mu_cam = (homo @ viewmat.T)[:, :3]                          # [N, 3]
    z = mu_cam[:, 2].clamp(min=1e-4)                            # 防除零

    # --- 2. 协方差 (3D) ---
    q = quats / quats.norm(dim=-1, keepdim=True)
    R = quat_to_rot(q)                                          # [N, 3, 3]
    S = torch.diag_embed(torch.exp(scales))                     # [N, 3, 3]
    cov3D = R @ S @ S.transpose(-1, -2) @ R.transpose(-1, -2)   # [N, 3, 3]

    # World→Cam 旋转部分 W_rot (3x3) 应用到协方差
    W_rot = viewmat[:3, :3]
    cov_cam = W_rot @ cov3D @ W_rot.T                           # [N, 3, 3]

    # --- 3. 投影 Jacobian J (2x3) ---
    fx, fy = K[0, 0], K[1, 1]
    x_c, y_c, z_c = mu_cam[:, 0], mu_cam[:, 1], z
    J = torch.zeros(N, 2, 3, device=device)
    J[:, 0, 0] = fx / z_c
    J[:, 0, 2] = -fx * x_c / z_c**2
    J[:, 1, 1] = fy / z_c
    J[:, 1, 2] = -fy * y_c / z_c**2

    # --- 4. 2D 协方差 Σ' = J W Σ W^T J^T ---
    cov2D = J @ cov_cam @ J.transpose(-1, -2)                   # [N, 2, 2]
    cov2D = cov2D + 0.3 * torch.eye(2, device=device)           # low-pass filter (anti-aliasing)

    # --- 5. 2D 中心 (像素坐标) ---
    cx, cy = K[0, 2], K[1, 2]
    mu2D = torch.stack([fx * x_c / z_c + cx, fy * y_c / z_c + cy], dim=-1)  # [N, 2]

    # --- 6. 深度排序 (front to back) ---
    depth = z_c
    order = depth.argsort()                                     # ascending z
    mu2D, cov2D = mu2D[order], cov2D[order]
    colors_o = colors[order]
    alphas = torch.sigmoid(opacities[order]).squeeze(-1)        # [N]

    # --- 7. 像素遍历 (教学版用全图 loop；真实实现 tile + CUDA) ---
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
        contrib = contrib.clamp(max=0.99)                       # 数值
        img = img + (T_acc * contrib).unsqueeze(-1) * colors_o[i]
        T_acc = T_acc * (1 - contrib)
        if (T_acc < 1e-4).all():                                # early stop
            break

    return img
```

> ⚠️ **教学版 vs 生产版差距** — 上面是 $O(N \cdot HW)$，30k Gaussian + 800×800 图就要好几秒。真实 `gsplat` 是 (a) tile-based: 每 tile 只处理"接触此 tile"的 Gaussian；(b) GPU radix sort 复合键；(c) 整个 forward / backward 全 manual CUDA，1080p ≤ 10ms。

### 4.5　自适应密度控制（**面试常问**）

3DGS 初始用 SfM (COLMAP) 稀疏点云，但训练过程要"撒"出更多 Gaussian。

| 触发条件 | 操作 | 直觉 |
| --- | --- | --- |
| **梯度大 + scale 小** | **clone**（原地复制一份：new_xyz = 原 xyz，无显式沿梯度偏移；两个重合 Gaussian 靠后续独立梯度下降自然分开） | "under-reconstruction"——这块区域缺细节 |
| **梯度大 + scale 大** | **split**（真正采样新位置的一步：以原 Gaussian 的 scale 当标准差做 torch.normal 采样 + 旋转，再把 scale ÷ 1.6，拆成 2 个小 Gaussian） | "over-reconstruction"——一个大 Gaussian 覆盖了不该覆盖的区域 |
| **opacity 接近 0** | **prune**（删除） | 该 Gaussian 没贡献，浪费显存 |
| **每 3k iter** | reset opacities to 0.01（0.005 是下面 prune 用的 min_opacity 阈值，两者不是同一个数） | 让 model 重新学透明度，防止 floater |

启发式条件：`gradient norm > τ_pos`（如 $2 \times 10^{-4}$），`scale > τ_scale`（场景尺度 1%）。

```python
def densify_and_prune(gaussians, grad_thresh=2e-4, scale_thresh=0.01,
                      max_screen_size=None):
    """ 教学版 densify 决策（简化；真实 gsplat 还有 screen-size 触发）。
        假设 gaussians 暴露以下 1D 形状的字段（N = 当前 Gaussian 数量）:
          xyz_grad_accum: [N]  ‖累积 xyz 梯度范数‖
          denom:          [N]  累积次数（防 /0）
          scales:         [N, 3]  log-scale
          opacities:      [N]  sigmoid 后 ∈ (0, 1)
          screen_size:    [N]  最近一次渲染的屏幕投影大小（可选）
    """
    grad_norm  = gaussians.xyz_grad_accum / gaussians.denom.clamp(min=1)      # [N]
    mean_scale = gaussians.scales.exp().max(dim=-1).values                    # [N]

    # ⚠️ 先在原始 N 个 Gaussian 上把 clone / split 两个 mask 都算出来，
    #    再做任何 append/删除——否则 clone_at append 之后数组变长，原 N 长度的
    #    split_mask 会与新数组错位（mask-length mismatch / 作用到错误对象）。
    clone_mask = (grad_norm > grad_thresh) & (mean_scale <= scale_thresh)     # [N]
    split_mask = (grad_norm > grad_thresh) & (mean_scale >  scale_thresh)     # [N]

    # CLONE：高梯度 + 小 scale —— 原地复制一份（new_xyz = 原 xyz，不做位置偏移）；
    # 原始保留（append 到末尾），两个重合的 Gaussian 靠后续独立梯度下降自然分开
    gaussians.clone_at(clone_mask)

    # SPLIT：高梯度 + 大 scale —— 才是真正采样新位置的一步：以原 Gaussian 的 scale 为
    # 标准差做 torch.normal 采样 + 旋转得到新位置，再把 scale ÷ 1.6，拆成 2 个子高斯，
    # 并在末尾删除原始
    # split_at 显式只作用于原始 N 个：传 original_n，不依赖 append 后的数组长度隐式对齐
    gaussians.split_at(split_mask, n=2, scale_div=1.6, original_n=split_mask.shape[0])

    # PRUNE：低 opacity / 屏幕过大 / 已被 split 标记
    # ⚠️ 注意：clone 增加的新 Gaussian 已 append 到末尾，长度变了；这里的 mask 仅作用于原 N 个
    prune_mask = (gaussians.opacities[:split_mask.shape[0]] < 0.005) | split_mask
    if max_screen_size is not None:
        prune_mask = prune_mask | (gaussians.screen_size[:split_mask.shape[0]] > max_screen_size)
    gaussians.remove_original(prune_mask)   # 只删除原始 N 个里被标记的

    gaussians.reset_grad_accum()
    return gaussians
```

> 💡 **典型超参数** — Kerbl 2023 paper: densify every 100 iters；最大 Gaussian 数 5e6；总训练 30k iter；约 30 分钟到几小时。

### 4.6　2DGS / Surfels（surface-aligned）

3DGS 的 ellipsoid 不是 surface-aware；提 mesh 需 SuGaR / GSDF 后处理。**2DGS** (Huang 2024 SIGGRAPH) 把 3D ellipsoid **退化成 2D disk**（一个 axis = 0），直接对齐表面，更适合 normal / depth 监督，提 mesh 时质量明显更好。

### 4.7　动态 4DGS

**Dynamic 3DGS** (Luiten 2024 3DV, arXiv:2308.09713) 用一组**持续存在的 Gaussian**：颜色 / 不透明度 / 大小跨帧保持不变，只让位置 $\mu(t)$ 与旋转随时间运动，并用 **local-rigidity** 正则约束邻域刚性；**4DGS** (Wu 2024 CVPR / Yang 2024 ICLR) 把 $\mu(t), \Sigma(t)$ 写成时间函数（MLP 或 spline）；**SC-GS** (Huang 2024) 用 sparse control points 驱动密集 Gaussian（类似 LBS）。

## §5 Mesh 提取：Marching Cubes / DMTet

NeRF / 3DGS 重建后，下游（仿真器、AR、3D 打印）经常需要 mesh。

### 5.1　Marching Cubes（**经典必考**）

输入：3D 标量场 $f(\mathbf{x})$（density / SDF）+ 阈值 $\tau$。输出：水平集 $\{f = \tau\}$ 的三角网格。

**算法骨架**：

1. 把空间 voxel 化（每 voxel 8 角点）
2. 对每个 voxel，**8 角点二值化**（$f > \tau$ 为 1, 否则 0）→ 256 种可能配置
3. 查 lookup table：每个配置预定义了几条等值面三角片 + edge 上的顶点位置
4. **线性插值** 找精确顶点：在 edge 两端 $\mathbf{a}, \mathbf{b}$ 之间，插值 $t = (\tau - f(\mathbf{a})) / (f(\mathbf{b}) - f(\mathbf{a}))$，顶点 $= \mathbf{a} + t(\mathbf{b} - \mathbf{a})$
5. 合并所有 voxel 三角片 → 完整 mesh

```python
def marching_cubes_sketch(density: torch.Tensor, threshold: float):
    """ 真实实现用 mcubes / scikit-image / pytorch3d；这里写思路 """
    from skimage.measure import marching_cubes
    # density: [Nx, Ny, Nz]；detach() 切断 autograd 图，cpu().numpy() 转 host
    verts, faces, normals, _ = marching_cubes(
        density.detach().cpu().numpy(),
        level=threshold,
        spacing=(1.0, 1.0, 1.0),
        gradient_direction='descent',  # 法线方向；descent = surface 朝低密度
    )
    return verts, faces, normals
```

> ⚠️ **NeRF 提 mesh 坑** — vanilla NeRF 没有"表面"概念，提 mesh 时阈值 $\tau$ 难选，且 floater 会被一起提出来。**用 NeuS / VolSDF 提 mesh 才稳**（SDF 0 等值面定义良好）。

### 5.2　Differentiable: DMTet / FlexiCubes（端到端学 mesh）

Marching Cubes 不可微（lookup table 离散）。

- **DMTet** (Shen 2021 NeurIPS / Munkberg 2022 CVPR)：用 **deformable tetrahedral grid**，每四面体 4 顶点 SDF + 位置 offset 可微。**Marching Tetrahedra** 替代 MC，topology 由 SDF sign 决定，几何由顶点位置决定：在拓扑不变的区域内，几何（顶点位置）对 SDF 值和偏移量**可微**；但顶点 SDF 跨越零点、引发该四面体 look-up case 切换的瞬间，拓扑本身**不可微**（测度零集合，类似 ReLU 在 0 点的 kink），不应无条件称"全程可微"。
- **FlexiCubes** (Shen 2023 SIGGRAPH)：泛化 dual marching cubes，引入额外可学参数（dual vertex offset / interpolation weight）解决 quality artifacts。

**典型用法**：DreamFusion 之后的 Magic3D、Fantasia3D 用 DMTet 在 SDS 监督下学 mesh + texture。

## §6 SDS Loss：用 2D Diffusion 监督 3D（DreamFusion 系列）

### 6.1　问题设置

我们想生成 3D 资产但 **没有 3D 训练数据**——3D 数据稀缺（ShapeNet ~5万件，Objaverse-XL 1000万件但质量参差）。**Pretrained 2D diffusion**（Stable Diffusion, Imagen）海量。能否用 2D diffusion 当老师 supervise 3D？

**DreamFusion** (Poole et al. 2022 arXiv → **ICLR 2023 Outstanding Paper**) 提出 **Score Distillation Sampling (SDS)**。

### 6.2　Setup

- 3D 表示 $\theta$（NeRF 参数 / DMTet vertices / 3DGS 点云）
- 可微渲染器 $g(\theta, \pi) \to x$，$x$ 是图像（$\pi$ 是相机视角）
- Pretrained 2D diffusion $\epsilon_\phi(x_t; y, t)$（$y$ 是文本 prompt）

**目标**：让 $g(\theta, \pi)$ 看起来像"$y$ 的 photo"，即 $g(\theta, \pi)$ 落在 diffusion 学到的数据流形上。

### 6.3　SDS gradient 推导（**L3 必考**）

直觉：用 diffusion training loss 反传到 $\theta$。**Naive 想法**：把渲染图 $x = g(\theta, \pi)$ 当训练样本，最小化

$$\mathcal{L}_\text{diff}(\theta) = \mathbb{E}_{t, \epsilon}\Big[w(t)\big\|\epsilon_\phi(x_t; y, t) - \epsilon\big\|^2\Big],\quad x_t = \alpha_t x + \sigma_t \epsilon$$

对 $\theta$ 求梯度（链式法则）：

$$\nabla_\theta \mathcal{L}_\text{diff} = \mathbb{E}\Big[w(t)\,2\big(\epsilon_\phi(x_t; y, t) - \epsilon\big)\,\underbrace{\frac{\partial \epsilon_\phi(x_t;y,t)}{\partial x_t}}_{\text{U-Net Jacobian}}\,\alpha_t\,\underbrace{\frac{\partial x}{\partial \theta}}_{\text{renderer Jacobian}}\Big]$$

**问题**：U-Net Jacobian $\partial \epsilon_\phi / \partial x_t$ 计算昂贵且数值差（diffusion 模型大且未训练 second-order 稳定）。

**SDS trick：直接扔掉 U-Net Jacobian**，得到：

$$\boxed{\;\nabla_\theta \mathcal{L}_\text{SDS} \;=\; \mathbb{E}_{t, \epsilon}\Big[w(t)\,\big(\epsilon_\phi(x_t; y, t) - \epsilon\big)\,\frac{\partial x}{\partial \theta}\Big]\;}$$

(原 DreamFusion 论文写成 $\partial L/\partial \theta$ 形式；$\alpha_t$ 与常数 2 被吸收到 $w(t)$ 里。)

### 6.4　为什么扔掉 Jacobian 反而 work？

**第一种解释（DreamFusion 原版，score 视角）**：$\epsilon_\phi(x_t; y, t)/\sigma_t \approx -\nabla_{x_t}\log p_\phi(x_t|y)$（score）。 SDS gradient = `(predicted score - noise) × renderer Jacobian`，相当于把渲染图朝高概率方向推。

**第二种解释（mode-seeking）**：SDS 等价最小化 $\mathbb{E}_t[D_\text{KL}(q(x_t|\theta) \,\|\, p_\phi(x_t|y))]$ 的某种 mode-seeking 形式：往 $p_\phi(\cdot|y)$ 的高概率区跑。

### 6.5　SDS 副作用：over-saturation / mode collapse / Janus

- **Over-saturation**：颜色饱和，对比度过高（"plastic-y look"）
- **Over-smoothing**：细节糊
- **Mode collapse**：物体趋于"canonical" 单一形式
- **Janus problem**：3D 物体在不同视角都长出一张"前脸"（人脸出现在头后；动物两边都是头）

**根因**：SDS 等价 mode-seeking KL，加上**大 CFG 系数（DreamFusion 默认 100）才能逃出 mean-mode**。CFG=100 把分布锐化到极端 → over-saturation。

> ⚠️ **面试高分点** — SDS 公式形式上"丢掉 Jacobian"省了计算，但**代价**是隐式变成 mode-seeking KL，需要超大 CFG 来缓解 mean-seeking blur；超大 CFG 又导致 over-saturation。这是个**信息论 trade-off**：simulation-free + computationally cheap = mode-seeking artifact。

### 6.6　SDS 代码（核心 30 行）

```python
def sds_loss(
    renderer,                 # θ → x (B, 3, H, W)
    theta,                    # 3D 参数 (NeRF / 3DGS / DMTet)
    prompt_emb,               # 文本 embedding (cond) [B, L, D]
    uncond_emb,               # 文本 embedding (uncond / null) [B, L, D]
    unet,                     # frozen 2D diffusion U-Net (e.g. SD); 返回 noise pred Tensor
    alpha_cumprod: torch.Tensor,  # [T_max] precomputed bar-alpha schedule
    cfg_scale: float = 100.0,
    t_range: tuple = (0.02, 0.98),
):
    """ Score Distillation Sampling loss (DreamFusion).
        约定: unet(x_t, t, encoder_hidden_states=emb) -> [B, 3, H, W] noise pred.
        如果用 diffusers 的 UNet2DConditionModel, 包一层取 .sample 即可。
        返回的是 grad surrogate, 直接 backward 即可。 """
    x = renderer(theta)                                  # [B, 3, H, W]
    B = x.shape[0]
    device, dtype = x.device, x.dtype
    T_max = alpha_cumprod.shape[0]                       # 通常 1000

    # 1. 采样 t 和 noise，forward 加噪
    t = torch.randint(int(t_range[0] * T_max), int(t_range[1] * T_max),
                      (B,), device=device)
    noise = torch.randn_like(x)
    abar = alpha_cumprod.to(device=device, dtype=dtype)[t].view(B, 1, 1, 1)
    x_t = abar.sqrt() * x + (1 - abar).sqrt() * noise

    # 2. U-Net 预测 noise (cond / uncond)，CFG 组合；关键：不对 diffusion 求导
    with torch.no_grad():
        eps_uncond = unet(x_t, t, encoder_hidden_states=uncond_emb)
        eps_cond   = unet(x_t, t, encoder_hidden_states=prompt_emb)
        eps_pred = eps_uncond + cfg_scale * (eps_cond - eps_uncond)

    # 3. SDS gradient: w(t)(ε_pred - ε) · ∂x/∂θ; w(t) = σ_t² 是常见选择
    grad = ((1 - abar) * (eps_pred - noise)).detach()
    # backward of (grad · x) 给出 grad · ∂x/∂θ
    return (grad * x).sum() / B
```

> 💡 **训练 loop** — 每 iter 随机采视角 $\pi$，渲染 $x$，算 SDS loss，反传到 $\theta$。NeRF 表示要训 10k-100k 步（GPU 几小时）；3DGS 表示（GaussianDreamer / DreamGaussian）几分钟到一小时。

### 6.7　VSD：变分 SDS（**ProlificDreamer**, NeurIPS 2023 Spotlight）

**VSD** (Wang 2023 NeurIPS) 把 SDS 视为"对 single $\theta$ 做点估计"的特殊情况，泛化为**对 $\theta$ 分布 $\mu(\theta)$ 做变分推断**。

#### Setup
- 把 3D 参数 $\theta$ 当 latent random variable，$\mu(\theta)$ 是其分布
- 目标：让 rendered image distribution 与 diffusion 学到的 prior **分布**对齐（不是 mode 对齐）

#### Objective and gradient

ProlificDreamer 把目标写成 KL：

$$\min_{\mu}\; D_\text{KL}\!\Big(q_\mu^t(x_t|y)\;\Big\|\;p_\phi^t(x_t|y)\Big),\quad t\sim\mathcal{U}[0,1]$$

其中 $q_\mu^t$ 是 "从 $\theta\sim\mu$ 渲染 + 加噪到 $t$" 诱导的分布。**变分梯度**（Wang et al. 2023, Theorem 2 略写）给出关于 $\theta$ 的更新方向为 **relative score**：

$$\boxed{\;\nabla_\theta \mathcal{L}_\text{VSD} \;=\; \mathbb{E}_{t,\epsilon}\Big[\,w(t)\,\big(\epsilon_\phi(x_t;y,t) \;-\; \epsilon_\psi(x_t;y,t,\pi)\big)\,\frac{\partial x}{\partial \theta}\,\Big]\;}$$

对比 SDS：把 raw noise $\epsilon$ 替换成**辅助 score** $\epsilon_\psi$。$\epsilon_\psi$ 是 **LoRA 微调** 的 score network，在线最小化 score-matching loss 来跟踪当前 $q_\mu^t$ 的 score；它是 KL 目标 $D_\text{KL}(q_\mu^t\|p_\phi^t)$ 对 $\theta$ 求导后、$q$ 分布自身 score（entropy 项）对应的必要成分，**不是**零均值、只减方差的 variance-reduction control variate——丢弃它会把梯度目标从"分布匹配"退化为 SDS 的 mode-seeking（改变期望梯度方向，而非只增加噪声），因此不能等同于 RL actor-critic 中只减方差、不改变期望梯度的 value baseline。**注意**：上面是 gradient form，不是平方-loss form；论文没有"先写一个 $\|\epsilon_\phi-\epsilon_\psi\|^2$ 标量 loss 再求导"的可实现形式——$\epsilon_\psi$ 依赖 $\mu$，那样会丢掉 KL 的关键项。

#### 直觉为什么缓解 over-saturation
- SDS：把 rendered $x$ 推向 prior $p_\phi(\cdot|y)$ 的 mode（mean-mode → 需 CFG=100 → over-saturation）
- VSD：用辅助 score 学当前 rendered 分布的"自己的 mode"，更新方向 = 从"我现在在哪"指向"prior 在哪"（**relative gradient**），不需大 CFG 拉满。CFG 可降到 7.5（常规 diffusion 默认值），avoid extreme sharpening
- 实验上：VSD 颜色更自然，几何更复杂，可同时维护多个 mode（ProlificDreamer 给出 50k 步训练得 photorealistic Buddha 等）

> ✅ **VSD vs SDS 的关键认知** — SDS 是 "**single-point + mode-seeking**"；VSD 是 "**particle / variational + relative score**"。后者的 $\epsilon_\psi$ 是 KL 目标对 $\theta$ 求导后 $q$ 分布自身 score 对应的必要项，丢掉它会让梯度目标退化回 SDS 的 mode-seeking——不是只减方差、不改变期望梯度的 RL actor-critic value baseline。

### 6.8　SDS 衍生家族：mesh / 3DGS + SDS

| 方法 | 表示 / Stages | 关键点 |
| --- | --- | --- |
| **DreamFusion** | NeRF + SDS @ low-res | 原版，提 mesh 难 |
| **Magic3D** (Lin 2023 CVPR) | Instant-NGP @ 64px → DMTet + SDS @ 512px | **两阶段**：粗结构 → 高分辨率端到端 mesh |
| **Fantasia3D** (Chen 2023 ICCV) | DMTet 几何 + PBR material | normal-as-input + 物理材质 BRDF |
| **DreamGaussian** (Tang 2024 ICLR) | 3DGS + SDS, ~2 分钟 / 物体 | GPU 速度优势；mesh export + UV-Net texturing |
| **GaussianDreamer** (Yi 2024 CVPR) | Point-E / Shap-E init → 3DGS + SDS | 缓解 from-scratch 几何混乱 |

## §7 Feed-forward 重建：从 COLMAP 到 pointmap regression

2024 年之前，"多视图 → 3D" 的第一步几乎必然是 SfM：COLMAP 做 SIFT → matching → incremental SfM → bundle adjustment，几十分钟到几小时，而且 texture-less、低重叠、动态物体这几种情况经常失败。**DUSt3R** (Wang 2024 CVPR, arXiv:2312.14132, Naver) 把这整条管线换成一次网络前向：不做显式匹配、不三角化、不做 bundle adjustment，**直接回归 pointmap**。到 2025 年 VGGT 拿下 CVPR Best Paper，这条线已经是 3D 视觉岗位的默认提问方向。

### 7.1　DUSt3R：pointmap 与置信度加权损失（**必考推导**）

**Pointmap** $X \in \mathbb{R}^{H\times W\times 3}$ 是"每个像素一个 3D 点"的稠密图，而且这些点的坐标**写在某一个指定的相机系里**。它同时约束了几何、内参和相机位姿，但取出来各有各的做法：

- **内参**：把 $X^{1,1}$ 的 3D 点与它们自己的像素坐标配对，拟合投影模型。
- **位姿**：把 $X^{2,1}$ 的 3D 点与 $I^2$ 的像素坐标配对，解 **PnP**——靠的是像素与 3D 点的对应关系。
- **深度**：$X^{2,1}$ 的 $z$ 分量**不是** $I^2$ 的深度，因为这些点写在 $I^1$ 系下；要拿 $I^2$ 的深度，得先把点变换回 $I^2$ 的相机系再取 $z$。

给两张图 $I^1, I^2$，DUSt3R 用共享权重的 ViT encoder 编码，两个 decoder 之间用 cross-attention 互看，输出两张 pointmap $X^{1,1}$ 和 $X^{2,1}$。**上标是关键：两张 pointmap 都表达在 $I^1$ 的相机系下。** 两张图的像素被放进了同一个坐标系，于是 **位姿不再是输入，而是可以从输出解出来的量**。这一句是整个 feed-forward 重建家族的原点。

> ⚠️ **不要说"把两张 pointmap 刚体对齐得到位姿"** — 它们本来就在同一个系里，配准无从谈起。位姿来自 $X^{2,1}$ 与 $I^2$ 像素之间的 2D-3D 对应（PnP），这是两回事。

**回归损失**。对视图 $v \in \{1, 2\}$ 的有效像素 $i \in \mathcal{D}^v$（GT 有定义的像素），逐点欧氏误差：

$$\ell_{\text{regr}}(v,i)=\Big\lVert \tfrac{1}{z}X^{v,1}_i-\tfrac{1}{\bar z}\bar X^{v,1}_i\Big\rVert$$

$\bar X$ 是 GT。两边各除以一个 scale factor，$z$ 用预测、$\bar z$ 用 GT，二者定义相同：

$$z=\text{norm}(X^1,X^2)=\frac{1}{\lvert\mathcal{D}^1\rvert+\lvert\mathcal{D}^2\rvert}\sum_{v}\sum_{i\in\mathcal{D}^v}\lVert X^v_i\rVert$$

即**所有有效点到原点的平均距离**。除以它以后，损失对整体缩放不变——网络只被要求把形状学对，尺度自由。**这正是"DUSt3R 式训练默认 up-to-scale"的技术来源。**

想要绝对尺度，实际做法不止一种，**MapAnything** 是其中一种：保留这种尺度不变的几何监督，**另外单独预测一个全局尺度并给它自己的损失**。归一化本身也有不同做法——**VGGT** 只归一化 GT、不做 DUSt3R 这种预测侧的归一化，但它学的仍是归一化后的尺度，不是 metric 路线。下面代码里的 `use_metric`（令 $z=\bar z$）是最直接的教学变体，把绝对尺度直接压进回归项——它便于理解，但不是上面两篇的做法。

**置信度加权**。真实数据里有天空、反光、半透明、动态物体这些"GT 本身不可信"的像素。DUSt3R 让网络额外吐一张置信度图 $C^{v,1}$，损失变成：

$$\boxed{\;\mathcal{L}_{\text{conf}}=\sum_{v}\sum_{i\in\mathcal{D}^v}\Big[\,C^{v,1}_i\,\ell_{\text{regr}}(v,i)\;-\;\alpha\log C^{v,1}_i\,\Big]\;},\qquad C=1+\exp(\check C)\gt 1$$

其中 $\check C$ 是 confidence head 的原始输出，$1+\exp(\cdot)$ 的参数化保证 $C\gt 1$（每个像素至少被计入一次）。论文把 $-\alpha\log C$ 称为**正则项**。

**为什么必须有 $-\alpha\log C$**（面试就问这个）。把单个像素的损失看成 $C$ 的函数，定义域 $C\gt 1$：$f(C)=C\ell-\alpha\log C$。

- **去掉 $\log$ 项**：$f(C)=C\ell$ 对 $C$ 单调不减（$\ell\ge 0$），最优把 $C$ 推到定义域下界 $C\to 1$，逐像素损失退回普通回归损失 $\ell$。**几何照样在学**——退化的只是置信度 head：它变成常数，不再携带任何信息。正则项存在的意义就是挡住这个退化。
- **加上 $\log$ 项**：$f'(C)=\ell-\alpha/C$。$\ell\lt \alpha$ 时驻点 $C^\star=\alpha/\ell\gt 1$ 落在定义域内；$\ell\ge\alpha$ 时 $f'(C)\gt 0$ 恒成立，下确界仍在 $C\to 1^+$。
- **把 $C$ 消掉**（对 $C$ 取下确界的**优化消元**，不是概率意义上的边缘化）：

$$\inf_{C\gt 1} f(C)=\begin{cases}\alpha+\alpha\log(\ell/\alpha), & 0\lt \ell\lt \alpha\\[2pt] \ell, & \ell\ge\alpha\end{cases}$$

- **所以它到底做了什么**：$\ell\lt \alpha$ 的那一段被换成对数形状，$\ell\ge\alpha$ 的那一段**仍然是线性的 $\ell$**。它**不是**把大残差压成对数的 outlier 抑制器——大残差照常线性计入。$\alpha$ 是这条分界线的位置：只有当一个像素的残差小于 $\alpha$ 时，网络抬高 $C$ 才有收益（换来 $\alpha\log(\ell/\alpha)\lt 0$ 的负贡献）。于是 $C$ 学到的是"这个像素我算得准不准"，而天空、反光、动态物体这些算不准的地方，$C$ 就停在 1 附近拿不到奖励。

> 💡 **$\alpha$ 是唯一的旋钮** — 它同时是"多小的残差才值得表态"的阈值和置信度奖励的强度。$\alpha$ 调大，更多像素进入对数段、置信度图更"敢说"；固定残差时，降低 $\alpha$ 会让更多像素落进 $C\to 1$ 的那一支，机制退回普通回归。答这题时能说出分界点在 $\ell=\alpha$，比背公式高一档。

```python
import torch

def dust3r_conf_loss(pred_pts, gt_pts, conf_raw, valid, alpha=0.2, use_metric=False):
    """ DUSt3R 置信度加权 pointmap 损失（教学版）。
        pred_pts: [B, V, H, W, 3]  两个 head 的 pointmap，都在 I^1 相机系下
        gt_pts:   [B, V, H, W, 3]  同一坐标系的 GT（V = 2），无效像素可以是 NaN
        conf_raw: [B, V, H, W]     confidence head 原始输出 Ĉ（未激活）
        valid:    [B, V, H, W]     bool，GT 有定义的像素集合 D^v
        use_metric: 令 z = z̄ 的教学变体，把绝对尺度压进回归项（不是论文做法）
        注意：论文的 L_conf 是在有效像素上求和；这里返回的是"逐样本有效像素均值
        再对 batch 取均值"，便于不同 batch 之间比较。
    """
    eps = 1e-8
    vm = valid.unsqueeze(-1)                                       # [B, V, H, W, 1]
    zero = torch.zeros((), dtype=pred_pts.dtype, device=pred_pts.device)
    # 先选后算：GT 的无效深度常用 NaN 编码，NaN * 0 仍是 NaN，会污染整个样本
    pred_v = torch.where(vm, pred_pts, zero)
    gt_v = torch.where(vm, gt_pts, zero)

    n = valid.flatten(1).sum(1).to(pred_pts.dtype).clamp(min=1.0)  # [B]  |D^1| + |D^2|
    # norm(·) = 有效点到原点的平均距离；pred / gt 各算一个
    z_bar = gt_v.norm(dim=-1).flatten(1).sum(1) / n                # [B]  z̄
    z = z_bar if use_metric else pred_v.norm(dim=-1).flatten(1).sum(1) / n

    bshape = (-1,) + (1,) * (pred_pts.dim() - 1)                   # [B, 1, 1, 1, 1]
    l_regr = (pred_v / z.view(bshape).clamp(min=eps)
              - gt_v / z_bar.view(bshape).clamp(min=eps)).norm(dim=-1)   # [B, V, H, W]

    C = 1.0 + conf_raw.exp()                                       # C > 1，保证每像素都计入
    per_pix = C * l_regr - alpha * C.log()                         # 去掉 log 项 → C 全塌到 1
    per_pix = torch.where(valid, per_pix, torch.zeros((), dtype=per_pix.dtype,
                                                      device=per_pix.device))
    return per_pix.flatten(1).sum(1).div(n).mean()
```

> ⚠️ **两处容易写错** — 一是 scale 归一化必须在**两张图一起**算（$\lvert\mathcal{D}^1\rvert+\lvert\mathcal{D}^2\rvert$ 是分母），分开归一化会把两张图之间的相对尺度关系抹掉；二是 $C$ 不能 detach，梯度必须流回 confidence head，否则"残差小于 $\alpha$ 才抬 $C$"这条机制根本形成不了。

**MASt3R** (2406.09756) 在 DUSt3R 上加了一个稠密局部特征 head，把"回归 pointmap"和"做像素级匹配"合成一个模型，匹配精度大幅提升，后续 MASt3R-SfM 用它替换 COLMAP 的 matching 前端。

### 7.2　共通配方与 VGGT 的四个 head

这一族的配方大体一致，但**每一条都有例外，答题时要把例外说出来**：

1. **视图之间靠 attention 交换信息**，而不是先做手工特征匹配再解几何。但"一次前向吃掉所有视图"不是全族通性：DUSt3R 本身是**成对**模型，多视图仍要接一个全局对齐阶段（而且不必跑满所有图像对）；Fast3R 才是真的一次前向出全部 pointmap，不过它的**相机参数是事后从 pointmap 恢复的**。
2. **输出落在同一个公共坐标系**，通常是某一张参考图的相机系。**π³ 是例外**：它的原始输出是每个视图**各自相机系下的局部 pointmap** 加上相应位姿，公共坐标系要由这些位姿拼出来。
3. **位姿通常是输出而不是必需输入**——不需要先跑 SfM 初始化。
4. **尺度看训练方式**：DUSt3R 式的预测 + GT 双边归一化 → up-to-scale；VGGT 只归一化 GT。metric 输出不止一条路——MapAnything 通过单独预测全局尺度获得 metric 输出，也可以直接采用 metric 几何监督（CUT3R 即如此，MASt3R 同样在 metric 数据上训）。

**VGGT** (Wang 2025 CVPR **Best Paper**, arXiv:2503.11651, Oxford + Meta) 是这条线目前的参考实现。它用交替的 frame-wise / global attention 处理 N 张图，从**同一个 backbone** 引出四个 head：

| Head | 输出 | 下游用途 |
| --- | --- | --- |
| **Camera** | 每张图的外参 + 内参（四元数 + 平移 + FOV） | 替代 SfM 的位姿估计 |
| **Depth** | 每张图的稠密深度（DPT 式） | 深度补全 / 融合 |
| **Point map** | 公共坐标系下的稠密 3D 点 | 直接出点云 |
| **Track** | 给定查询像素，预测**同一物理点在每张图上的 2D 位置** | 对应关系 / 跟踪与匹配 |

> 💡 **VGGT 最反直觉的一条观察** — 论文报告：**推理时把 depth head 和 camera head 的结果组合出点云，比直接用 point map head 的输出更准**。注意这是一条实验观察，不等于证明了"多任务训练的收益不在 head 上"；但它至少说明，**训练时用哪些监督目标**和**推理时走哪条路径**是两个可以分开的决策。

**为什么面试里 VGGT 取代了 "先 COLMAP" 的标准答案**：秒级而非小时级；不依赖已标定内参，对重叠的要求也宽松得多；四种量一次输出，下游（3DGS 初始化、SLAM、机器人建图）拿来即用。但这不是"COLMAP 已死"——精度与基准 GT 的讨论见 §12.1。

### 7.3　分支结构：这一族在往哪几个方向长

- **Fast3R** (2501.13928, Meta, CVPR 2025)：DUSt3R 是成对模型，多视图要"跑若干图像对 + 全局对齐"两个阶段。Fast3R 把 **N 张图一次前向**全部送进 transformer 直接出 pointmap，论文示范到 1000+ 张，省掉全局对齐；相机参数则由 pointmap 在后处理里恢复。
- **CUT3R** (2501.12387, Berkeley)：改成**循环式**——维护一个持久状态，每读入一帧就更新状态并输出该帧的 pointmap。这让它变成 **online / 流式**方法，可以边采边建，也能读单张图；输出的 pointmap 与位姿是 **metric** 的。
- **π³** (2507.13347)：前面的方法都要指定一张**参考视图**当坐标原点，于是结果依赖"选了哪张"，参考图退化时整体跟着受累。π³ 做成**置换等变**（permutation-equivariant）：输出随输入顺序等变，**不存在被特权化的那一张**。
- **MapAnything** (2509.13414, Meta + CMU, 3DV 2026)：用**因子化的 metric 表示**（ray 方向 + 深度 + 位姿 + 尺度）统一一堆任务，并且**输入是可选的**——有内参就用内参，有位姿就用位姿，什么都没有也能跑。尺度由单独预测的全局因子给出。
- **Depth Anything 3** (2511.10647, ByteDance)：反向做减法——不要特制架构、不要四个 head，**一个 plain transformer + 单一的 depth-ray 预测目标**就够了。论文在作者自建的基准上报告相对 VGGT 的**平均相对提升**：相机位姿 44.3%、几何 25.1%。**开源权重。**
- **VGGT-Ω** (2605.15195, CVPR 2026 Oral)：把 VGGT 的**训练显存**降到 30%，并把训练用的**有监督数据**扩到 15×（这是两件事，不是"同一预算下的等价交换"），能力扩到**动态场景**（VGGT 系原本假设静态）。它预测 depth 与相机。**权重未确认。**

| 模型 | 输入视图 | 主要输出 | 需要位姿输入？ | metric？ | 权重 |
| --- | --- | --- | --- | --- | --- |
| **DUSt3R** (2312.14132) | 2（多视图靠全局对齐拼） | pointmap + 置信度 | 否 | 否（up-to-scale） | 开源权重 |
| **MASt3R** (2406.09756) | 2 | pointmap + 稠密局部特征 | 否 | 是（metric 数据训练） | 开源权重 |
| **VGGT** (2503.11651) | N，一次前向 | 相机 / depth / pointmap / track | 否 | 否 | 开源权重（非商用许可） |
| **Fast3R** (2501.13928) | N，示范 1000+ | pointmap（相机后处理恢复） | 否 | 否 | 开源权重 |
| **CUT3R** (2501.12387) | 流式逐帧（含单图） | pointmap + 位姿 + 持久状态 | 否 | 是 | 开源权重 |
| **π³** (2507.13347) | N，置换等变 | pointmap + 相机 | 否 | 否 | 开源权重 |
| **MapAnything** (2509.13414) | N，其余输入可选 | 因子化 metric 表示 | 可选（有就用） | 是 | 开源权重 |
| **Depth Anything 3** (2511.10647) | N | depth-ray → 位姿 / 几何 | 否 | — | 开源权重 |
| **VGGT-Ω** (2605.15195) | N，含动态场景 | depth + 相机 | 否 | 否 | 权重未确认 |

**MV-DUSt3R+** (2412.06974) 是 DUSt3R → 多视图的另一条过渡路线（multi-view decoder block + 跨参考视图融合）；这一族的系统梳理见综述 arXiv:2507.08448。

### 7.4　Feed-forward 3DGS

上面这条线在回归几何，另一条线在回归**可渲染的 3DGS**——把 §4 里那 10-30 分钟的逐场景优化也换成一次前向。**注意这条线对输入位姿的要求并不统一**：

- **LGM** (2402.05054, Tang 2024 ECCV)：非对称 U-Net 吃 4 张多视图图像，**每个像素直接吐一个 3D Gaussian**，几秒出资产。这 4 张图来自**预设的固定环绕相机位姿**，由上游多视图生成模型按这套位姿出图。
- **GS-LRM** (2404.19702)：把 LRM 的 transformer 换成输出 Gaussian 参数——**per-pixel Gaussians**，输入 patch token 进、Gaussian 出，**2-4 张已标定位姿的视图**，A100 上 **0.23 秒**。这是"LRM 出 triplane"到"LRM 出 3DGS"的转折点（LRM 见 §8.3）。
- **Long-LRM** (2410.12781)：把输入推到 **32 张 960×540 的已标定视图**，在 A100 上约 **1 秒**内重建整个房间级场景——相对逐场景优化约 **800×** 加速，质量与优化方法相当甚至更好。后续 Long-LRM++ (2512.10267)。
- **DepthSplat** (2410.13862, CVPR 2025)：把预训练单目深度特征接进多视图 Gaussian 预测，让 depth 和 splatting 互相促进——深度先验补上了稀疏视图下最缺的那部分约束。输入同样是**已标定**的多视图。
- **NoPoSplat** (2410.24207)：**Gaussian 网络本身不需要位姿输入**——它把**第一张输入图的相机系直接定义为 canonical space**，所有 Gaussian 都吐在那里，内参以 token 注入。注意"pose-free"限定在这个网络上：要渲染目标视角、要评测，仍然有一个单独的位姿估计流程。
- **AnySplat** (2505.23716)：未标定图集进去，**Gaussians + 内参 + 外参**一起出来——几何回归那条线和 3DGS 回归这条线在这里合流。

> ⚠️ **别把 feed-forward 3DGS 当成 3DGS 的替代品** — 它换掉的是"逐场景优化"这一步，不是 3DGS 表示本身；输出仍然是 Gaussian，仍然按 §4.3 光栅化，**也仍然可以接着优化**。谁的质量更高要看设定：Long-LRM 在它的房间级设定下报告与逐场景优化相当或更好，视图很密时优化通常仍占优。取舍见 Q15。

## §8 Single-Image / Few-View 3D 生成

更实用的设定：**给一张图，生成 3D**。

### 8.1　Zero-1-to-3 范式（novel view via diffusion）

**Zero-1-to-3** (Liu 2023 ICCV)：用 Objaverse 上 finetune Stable Diffusion，让它接收 (input view, target camera) → output novel view。

- Input：单图 $x$ + 相对相机位姿 $\Delta R, \Delta T$
- Diffusion conditioning：image embedding (CLIP) + camera embedding (sinusoidal)
- Output：在 $\Delta R, \Delta T$ 视角下的图

**用法**：给一个 input view，sample 16-32 个 novel view，再用 NeRF / 3DGS 重建。

**衍生**：
- **Zero-1-to-3++** (Shi 2023)：固定生成 6 个 anchor view（方位角在 30°/90°/150°/210°/270°/330° 每 60° 均匀环绕一周，仰角在 -10° 与 20° 之间交替排列，即 interleaved elevation + uniform azimuth，而非"一个北极视角 + 四个平视角 + 一个俯视角"），减少 randomness
- **SyncDreamer** (Liu 2024 ICLR)：在 latent 上**联合**预测多视图（cross-attention 让 views 看到彼此），保证 3D 一致
- **MVDream** (Shi 2024 ICLR)：text-to-multi-view，4 视图同时生成；后接 SDS 精化

### 8.2　One-2-3-45 / InstantMesh / TripoSR / Stable Fast 3D

| 方法 | 输入 | 输出 | 速度 | 关键 |
| --- | --- | --- | --- | --- |
| **One-2-3-45** (Liu 2023 NeurIPS) | 单图 | mesh | 45 秒 | Zero-1-to-3 → SparseNeuS |
| **One-2-3-45++** (Liu 2024) | 单图 | mesh | 60 秒 | 多视图 + SDF |
| **TripoSR** (Tochilkin 2024, Stability+Tripo) | 单图 | NeRF/mesh | 0.5-2 秒 | LRM (Large Reconstruction Model) 风格 transformer |
| **InstantMesh** (Xu 2024, arXiv:2404.07191) | 单图 | mesh | ~10 秒 | Zero-1-to-3++ 多视图 → sparse-view recon transformer |
| **Stable Fast 3D** (SF3D, Stability 2024) | 单图 | textured mesh | ~0.5 秒 | TripoSR 后继；加 illumination disentangle + UV unwrap |

**LRM (Hong et al. 2023 arXiv → ICLR 2024) 设定**：把图当 token + Plucker ray embedding，transformer 输出 NeRF triplane。这是 TripoSR / InstantMesh 的母模型。

### 8.3　LRM Triplane 表示（**面试高频**）

- **Triplane** (Chan 2022 EG3D)：3 个轴对齐 2D 平面（XY, YZ, XZ），共 $3 \times C \times N \times N$ 维
- 查询 3D 点 $(x, y, z)$：在每个平面双线性插值 → **三个平面特征逐元素相加**（EG3D 是 sum，不是 concat）→ 小 MLP → $(\sigma, \mathbf{c})$
- 优点：比 voxel grid 显存少（$O(N^2)$ vs $O(N^3)$），比 hash grid 更 dense 适合 transformer 输出
- LRM / TripoSR / InstantMesh 都让 transformer 直接 regress triplane tokens

## §9 3D Foundation Models（2024–26 开源浪潮）

### 9.1　TRELLIS：SLAT 与两阶段 rectified flow

**TRELLIS** (Xiang 2024, arXiv:2412.01506, Microsoft) 是"3D 的 Stable Diffusion"这条路线上最完整的开源尝试，也是面试里最该讲清楚的那一个——因为它的 latent 设计直接决定了它的工程形态。

**SLAT (Structured LATents)** 的定义：

$$\boldsymbol z=\{(\boldsymbol z_i,\boldsymbol p_i)\}_{i=1}^{L},\qquad \boldsymbol p_i\in\{0,1,\dots,N-1\}^3,\quad N=64$$

即"**latent 向量 + 它所在的体素坐标**"的集合。$N=64$ 是体素分辨率（$64^3\approx 2.6\times10^5$ 个格子），而 $L\approx 20\text{K}$ 是**被占用的 active voxel 数**，约 8% 的占用率。

**分工要说准**：$\boldsymbol p_i$ 只确定**稀疏支撑**——资产的物质分布在哪些格子里，粒度就是 $64^3$；真正的**细几何和外观都编码在 $\boldsymbol z_i$ 里**，否则 64 分辨率的体素根本表达不出锐边和曲面。$\boldsymbol z_i$ 的来源是：把每个 active voxel 投影到多视图渲染图上，在 **DINOv2 特征图**里采样并聚合，再由一个 sparse VAE 压缩。所以 SLAT 从一开始就是**几何与外观联合编码**的。

**两阶段 rectified flow**（顺序不能反）：

1. **Structure 阶段**：先生成"哪些体素是 active"，即稀疏结构 $\{\boldsymbol p_i\}$ 本身。
2. **Latent 阶段**：在已确定的 active voxel 上生成 $\{\boldsymbol z_i\}$。

拆成两段的直接收益是第二阶段只在约 20K 个 token 上跑 flow，而不是 $64^3\approx262\text{K}$ 个格子上——**先定骨架再填内容，省掉的是被跳过的那 92% 空格子的注意力与前向计算**。

**多 decoder**：同一份 SLAT 训了三个 decoder，分别出 **3DGS / Radiance Field / Mesh**（mesh 走的是 FlexiCubes，不是朴素 marching cubes，见 §5.2）。

模型规模 **342M / 1.1B / 2B**，训练数据是从四个公开数据集（Objaverse(-XL)、ABO、3D-FUTURE、HSSD）筛出的**约 50 万件**资产。**开源权重。**

**TRELLIS.2** (2512.14692) 是它的后继。它的 Sparse Compression VAE 以 **O-Voxel** 为**原生表示**——稀疏体素同时携带几何与 PBR 材质；VAE 把 O-Voxel 压成更紧凑的 latent，**三个 flow model** 在压缩 latent 上分工生成，mesh 走自己的 dual-grid 转换。**权重未确认。**

### 9.2　Hunyuan3D 家族：2.0 → 2.1 → 2.5 → Omni / Studio / Buffalo

Hunyuan3D 走 **shape-then-texture** 两阶段路线，是国内开源生态里迭代最快的一条线。

- **Hunyuan3D-1** (Yang 2024)：stage 1 文/图 → 多视图（Zero-1-to-3 系），stage 2 多视图 → mesh（LRM-like）。
- **Hunyuan3D 2.0** (2501.12202)：**Hunyuan3D-DiT**——ShapeVAE 把点云编成 **vecset 形状 latent**，在这个 latent 上做 **flow matching**，解码出 SDF；**Hunyuan3D-Paint** 做多视图贴图 diffusion + UV 空间 refine，输出的是 RGB 贴图。**开源权重。**
- **Hunyuan3D 2.1** (2506.15442)：**PBR 材质生成是这一版才进主线的**，同时公开**训练代码**——后一条对想复现的人比版本号重要得多。
- **Hunyuan3D 2.5** (2506.16504)：几何模型换成 **LATTICE，10B 参数**，细节和锐边明显改善。**权重未确认。**
- **Hunyuan3D-Omni** (2509.21245)：可控生成——除了图像/文本，还接受 **位姿 / bounding box / 体素** 作为控制信号。**权重未确认。**
- **Hunyuan3D Studio** (2509.12815)：面向生产管线的资产化工具链（重拓扑、UV、贴图整合）。**权重未确认。**
- **Hunyuan3D Buffalo 1.0** (2608.02711)：把**生成、理解、编辑**统一进一个模型。**权重未确认。**

> ⚠️ **没有 "Hunyuan3D 3.0"** — 这条线是 2.0 → 2.1 → 2.5 之后直接分叉成 Omni / Studio / Buffalo 的命名体系。面试里被问"最新版本"时说成 3.0，会当场暴露是从二手摘要里背的。

### 9.3　latent 表示这条轴：CLAY / Direct3D / Dora / Step1X-3D

把这几个工作按发布时间排没有意义，按 **latent 表示与解码目标**排才是面试要的结构。

- **CLAY** (Zhang 2024 SIGGRAPH, 2406.13897)：**多分辨率 VAE + latent DiT**。它的 shape VAE 在 **3DShape2VecSet** (Zhang et al. 2023, arXiv:2301.11445) 的 vector-set 表示上扩展而来（属于 VecSet 谱系），DiT 在这个 latent 上做 diffusion，**解码出 occupancy field**，再 marching cubes 取表面，后面接 PBR 贴图阶段。
- **Direct3D** (2405.14832)：D3D-VAE 把 mesh 编到**显式 triplane latent**，D3D-DiT 在 triplane 上做 diffusion。triplane 的好处是 2D 卷积/注意力的整套工程直接复用，坏处是三个平面的投影歧义（细长结构在某个平面上会重叠）。
- **Direct3D-S2** (2505.17412)：换成**稀疏体素 latent** + **spatial sparse attention (SSA)**，把 **1024³** 分辨率的训练压到 **8 张 GPU** 上跑得动。论文报告的 3.9× 前向 / 9.6× 反向是 **SSA 算子相对 FlashAttention-2 的加速**，不是整个模型的端到端倍数。
- **Dora** (2412.17808, CVPR 2025)：走 **VecSet** 路线，改的是采样与注意力——**Sharp Edge Sampling** 把采样预算倾斜到几何锐边上，再配 **dual cross-attention**。结果是 Dora-VAE 只用 **1,280 个 latent code**，就在 Dora-bench 上达到与稠密的 XCube-VAE 相当的重建质量（后者需要 >10,000）。**latent 数量下降一个数量级，直接决定了后面 DiT 的序列长度。开源权重。**
- **Step1X-3D** (2505.07747)：几何走 **VecSet latent，解码为 TSDF** 的 hybrid VAE-DiT，配 **200 万**件清洗后的资产（从 500 万+筛出）；几何与贴图两阶段的代码、权重与训练流程全部开源（Apache-2.0）。**开源权重。**
- **Meta AssetGen 2** (2605.26137)：闭源产品线，在其 H100 部署上把单资产生成压到约 **30 秒**，主打生产可用性而非论文指标。

**Rodin** (Microsoft 2023, 商业)：早期 text-to-3D-avatar 产品级系统，diffusion on triplane，主打 character / avatar。

### 9.4　对比表：latent 表示才是那条轴

| 方法 | **latent 表示** | 解码目标 | Prior | 权重 |
| --- | --- | --- | --- | --- |
| **TRELLIS** | **SLAT**（稀疏体素支撑 + per-voxel latent） | 3DGS / 场 / mesh（FlexiCubes） | Rectified Flow ×2 | 开源权重 |
| **TRELLIS.2** | SC-VAE 压缩 latent（原生表示为 **O-Voxel**，几何 + PBR） | mesh（dual-grid）+ PBR | Flow ×3 | 权重未确认 |
| **Hunyuan3D 2.0 / 2.1** | **VecSet**（ShapeVAE 形状 latent） | SDF → mesh；贴图另起一阶段 | Flow matching | 开源权重 |
| **Hunyuan3D 2.5** | VecSet（LATTICE 10B） | SDF → mesh | Diffusion | 权重未确认 |
| **CLAY** | **VecSet 谱系**（多分辨率，扩展自 3DShape2VecSet） | occupancy → mesh | Diffusion (DiT) | 部分 |
| **Direct3D** | **Triplane** | 隐式场 → mesh | Diffusion (DiT) | — |
| **Direct3D-S2** | **稀疏体素** + SSA | 高分辨率隐式场 → mesh | Diffusion | — |
| **Dora** | **VecSet**（1,280 codes，Sharp Edge Sampling） | 隐式场 → mesh | VAE + DiT | 开源权重 |
| **Step1X-3D** | **VecSet** | **TSDF** → mesh | Diffusion | 开源权重 |
| **TripoSR / SF3D / LRM** | **Triplane**（前馈 regress，无 prior） | NeRF / mesh | 无 | 开源权重 |
| **Rodin** | Triplane | avatar | Diffusion | ❌ |

「权重」列只在已核实处写"开源权重"，"权重未确认"表示没找到官方发布，"—"表示本文未核实。

> 💡 **这条轴真正决定什么** — 三种 latent **都能按坐标查询解码**：triplane 用投影 + 双线性插值 + 逐元素相加（§8.3），VecSet 用查询坐标对 latent 集合做 cross-attention，SLAT 按体素位置取。所以"能出几种格式"是**各家训了哪些 decoder** 的结果，不是表示本身的禁令——TRELLIS 一口气训了三个，别家只训了 mesh 那条路。表示真正决定的是**计算和显存摆在哪里**：SLAT / 稀疏体素把 latent 挂在显式稀疏支撑上，局部计算和按位置匹配的解码最顺手；VecSet 没有空间索引，但集合本身很小（Dora 压到 1,280），全连接注意力也不贵，代价是丢掉了局部性；triplane 复用 2D 工程最省事，代价是分辨率一上去平面显存就涨。**稠密 token 一样可以上窗口化 / 稀疏注意力**——Direct3D-S2 的收益来自稀疏支撑让被跳过的那部分计算真的不存在，而不是"只有稀疏表示配得上稀疏注意力"。

### 9.5　原生 mesh 生成：为什么美术不要等值面提取的输出

上述方法的 mesh 输出主要来自**两条不同的路**：从场里提等值面——朴素的 marching cubes（§5.1）或 FlexiCubes 这类可微变体（§5.2，TRELLIS 用的就是它）；或者像 TRELLIS.2 那样从 **O-Voxel 直接做 dual-grid 转换，根本不经过场**。**两条路的问题都不在提取质量而在拓扑**：三角形的排布跟着提取网格或体素网格走，不跟着形状的结构走。UV 能展、能简化、能做 LOD——这些都不是做不到，FlexiCubes 这类方法也能相当程度地保住锐边。难受的是**输出不保证具有适合编辑、绑骨和形变的边流**：没有沿轮廓和折痕组织的 edge loop，手工改形和蒙皮都别扭，减面也得跟原有三角化对抗而不是顺着它走。美术手工建的 mesh（**artist-created mesh**）通常只有几百到几千面，每条边都贴着几何特征。

于是有了**原生 mesh 生成**：不提等值面、不做体素转换，**直接自回归地吐出面片序列**。这条线的主线是 **token 压缩**——序列长度 = 面数 × 每面 token 数，而 transformer 是平方复杂度。基线是**朴素序列的 9 个坐标 token / 面**（3 顶点 × 3 坐标），下面几篇都在跟这个 9 比；到 Meshtron 则换了思路，用架构去扛长序列而不是继续压 token。

- **MeshGPT** (2311.15475)：开山之作。用图卷积编码 + **残差量化**把每个面压成 **6 个码本 token**，再用 decoder-only transformer 自回归生成。
- **MeshAnything** (2406.10163)：加**形状条件**——先用任意方法（重建、生成）得到一个粗形状，再让 AR 模型按这个形状"重新建模"成 artist-style mesh。面数相对等值面提取的输出少**几百倍**。
- **MeshAnything V2** (2408.02555)：提出 **Adjacent Mesh Tokenization (AMT)**——利用相邻面共享顶点，一个面在多数情况下只需编码**一个新顶点**而非三个，token 数**约为朴素序列的一半**，可生成面数上限翻倍。
- **BPT** (2411.07025)：Blocked and Patchified Tokenization，**块索引**（坐标拆成块号 + 块内偏移）加上**patch 内的顶点共享**，相对**朴素序列**缩短 **约 75%**，把可生成面数推过 **8K**。
- **TreeMeshGPT** (2503.11629, CVPR 2025)：把面片序列换成沿**邻接树**的自回归——解码时从一个"待扩展边"的栈里取边并向外长面。压缩后的序列长度**约为原始的 22%**（≈ 2 token / 面，而不是"少 22%"）。它还在序列构造里约束面的定向，**明显减少法线翻转**（不等于完全省掉后处理）。
- **Meshtron** (2412.09548)：不压 token，硬扛长序列——**hourglass 架构 + 滑动窗口注意力**，做到 **64K 面 @ 1024 坐标分辨率**。**权重未确认。**
- **DeepMesh** (2503.15265)：在 AR mesh 生成上接 **DPO**，用偏好数据把"人类觉得好的拓扑"直接写进目标——mesh 生成里第一批 post-training 工作。

> ⚠️ **2026 年的反向潮流：Nexus** — Nexus (2607.13563) 的批评针对自回归本身：**逐 token 生成会让前面的错误顺着序列往后传**，而这个任务天生要长序列，压缩只是把序列变短，没有消除这条传播路径。它改成**由粗到细的顶点八叉树 diffusion**先定出顶点集合，再用 **topology embedding** 恢复边与面，因此不存在逐 token 的误差传播。代价也要说：**论文自己指出推理偏慢**——所以被问"AR 做 mesh 的问题"时答"慢"并不算错，只是没答到误差传播这一层。

### 9.6　部件级与 sim-ready：从"缺部件的整体"到能用的资产

生成出来的资产通常**缺少显式的部件和关节结构**。机器人要抓的抽屉、要转的轮子、要按的按钮，在输出里不是可以单独寻址的东西——仿真器没法给它们分配质量、摩擦、关节。这是 3D 生成离 Embodied AI 落地最近、也最卡的一环。

**第一步：拆成部件。** 这一簇的四个工作路线不同但目标一致：

- **PartGen** (2412.18608)：多视图 diffusion → 部件级分割 + **补全被遮挡的部分**（分割出来的部件本身是不完整的，必须生成式补全）。
- **HoloPart** (2504.07943)：把它形式化成 **3D part amodal segmentation**——把整体 mesh 分解成语义完整的部件，每个部件补成独立闭合体。
- **PartCrafter** (2506.05573)：不做"先整体再拆"，而是在生成时就**联合去噪多个部件**——部件之间的空间关系在联合训练里学到，而不是靠后处理对齐。
- **PartPacker** (2506.09980, NVIDIA)：**dual volume packing**，把多个部件打包进两个体积场里一次生成，避开了"部件数不定"给网络输出维度带来的麻烦。

**第二步：几何 → 部件 → 关节 → 材质 → URDF。** 拆完部件只解决了几何，离 sim-ready 还差三步：部件之间是什么**关节**（revolute / prismatic / fixed）、轴在哪、限位多少；每个部件的**物理材质**（质量、摩擦、恢复系数）；最后打包成仿真器读得懂的 **URDF / MJCF**。

- **EmbodiedGen** (2506.10600)：目前把这条链路走得最完整的开源工作之一，直接产出带物理属性的 **URDF 资产**，接 IsaacSim / MuJoCo。
- **Artiverse** (2605.24403)：**5.4K 件铰接物体 / 88 类**。对照系是 PartNet-Mobility 的 **2,346** 件——铰接资产的规模仍在千级，Artiverse 把它推高了一倍多，但离几何那一层的百万级差着三个数量级。

> 💡 **面试怎么答这一节** — 别背论文名。这一节的结构是"**一个问题被切成五段，每段的数据供给差得很远**"：几何有 Objaverse-XL 级的规模所以做得最好，部件级刚有 PartGen / HoloPart 这批方法，关节级的资产从 PartNet-Mobility 的 2,346 件到 Artiverse 的 5.4K 件都还是千级，物理材质基本还靠手工指定。**说得出哪一段是瓶颈，比列出十篇论文有用。**

## §10 复杂度 / 资源对比

| 方法 | 训练 | 推理耗时 | 运行显存（注明阶段） | 模型 / 表示规模 |
| --- | --- | --- | --- | --- |
| NeRF vanilla | 1-2 天 | 数秒 | 8 GB | <10 MB MLP |
| Instant-NGP | 5 秒 - 5 分钟 | 30 fps+ | 4-12 GB | 100-500 MB hash |
| 3DGS | 10-30 分钟 | 100 fps+ | 6-24 GB | 100 MB - 1 GB Gaussian |
| 2DGS | 与 3DGS 接近 | 与 3DGS 接近 | 类似 | 类似 |
| DreamFusion (NeRF+SDS) | 2 hr / 物体 | — | 12 GB | NeRF 本身 |
| DreamGaussian (3DGS+SDS) | 2 分钟 / 物体 | — | 8-16 GB | — |
| ProlificDreamer (VSD) | 3-6 hr / 物体 | — | 24 GB | — |
| TripoSR feedforward | 训练 50 GPU 天 | 0.5 秒 (A100) | inference 6 GB | 1.5 GB |
| GS-LRM / Long-LRM（前馈 3DGS） | 大规模多视图训练 | 0.23 秒 / 约 1 秒（32 视图 960×540） | — | — |
| VGGT（前馈重建） | 大规模多视图训练 | 秒级 / N 张图一次前向 | — | ~1B 参数 |
| TRELLIS | 训练 100+ GPU 天 | 数秒 | inference 16 GB | 数 GB |
| Hunyuan3D 2.0 / 2.1 | 训练大集群 | 数十秒 | inference 24+ GB | 多模型组合 |

## §11 与相关方法对比 & Embodied AI 应用

### 11.1　3D-vs-2D 生成关键区别

| 维度 | 2D 生成 (Stable Diffusion) | 3D 生成 |
| --- | --- | --- |
| **数据量** | LAION-5B 50亿图 | Objaverse-XL (2307.05663) 1000万件（小 500×） |
| **数据格式** | 图像（统一 RGB） | mesh / SDF / point cloud / NeRF / 3DGS（**碎片化**） |
| **训练 prior** | 直接 train diffusion | 用 2D diffusion 蒸馏 (SDS / Zero-1-to-3) **或** 用 3D-native diffusion (TRELLIS / CLAY) **或** 纯前馈 regression (LRM / VGGT) |
| **评测** | FID, CLIP score | Chamfer / IoU / PSNR (recon) + perceptual + user study |
| **下游** | 直接出图 | 出资产 → 渲染 / 仿真 / 编辑 |

### 11.2　Embodied AI / AR / VR 实战路线

| 任务 | 推荐表示 | 关键工具链 / 约束 |
| --- | --- | --- |
| **Sim2Real 资产** | mesh (PBR) | TRELLIS / Hunyuan3D 2.1 → IsaacSim / MuJoCo；要能动的资产还差部件 / 关节 / 物理材质（§9.6） |
| **室内大场景** | 3DGS | COLMAP → 3DGS（chunk-wise 用 VastGS / CityGS）；位姿也可由 VGGT 系给（§12.1） |
| **NeRF/3DGS as simulator** | NeRF / 3DGS + physics | DreamGaussian-Sim / Splatting Physics |
| **3D affordance / manipulation** | point cloud / 3DGS feature | OpenScene / LERF / RVT / 3D Diffuser Actor |
| **AR 物体扫描** | 3DGS（光照真实 + 实时）| mobile 算力（PostShot / Luma），剪枝 / 量化 |
| **VR 大场景** | 3DGS (large-scale) | 60 fps stereo + 6DoF |
| **Avatar** | mesh + LBS 或 3DGS avatar | 实时表情 / 头发 |
| **Object insertion** | mesh + PBR | 环境光照一致（IBL）|

> ⚠️ **Embodied AI 面试追问示例** — "做 NeRF 物理仿真器最大挑战？" 要点：NeRF 是 radiance，没 mass / friction → 需手动叠物理 prior；mesh 提取有 floater → 碰撞检测难；可微但 backward 慢；**业界更多用 3DGS / mesh 而非 vanilla NeRF**。

## §12 工程实战 & 易踩坑

### 12.1　COLMAP 还是 feed-forward？

COLMAP：输入多视图 → 输出内参 $K$ + 外参 $\{R_i, t_i\}$ + 稀疏点云；标准流程 SIFT → matching → incremental SfM → bundle adjustment。**常见坑**：texture-less / 镜面物体 SfM 失败；动态物体污染外参。

§7 之后这一步不再是"重建必经"，但也远没到可以删掉。要点是几条**条件性**的判断，不是谁替代谁：

- **精度看条件，不看牌子。** COLMAP 在纹理和重叠都够的静态场景上很强；但**单目 SfM 本身是 up-to-scale 的**——绝对尺度要靠标定物、已知基线或其他传感器，COLMAP 不自带。
- **基准的真值口径不统一。** 有些重建基准的位姿确实由 COLMAP 生成，另一些用传感器真值或合成数据。想跟哪条基线比，就对齐它用的那套口径，别一概说成"基准就是 COLMAP"。
- **Feed-forward 模型给的是秒级而不是小时级**，而且在 COLMAP 吃力的地方仍然出结果：低重叠、视图很少、texture-less。注意**内参未知不是 COLMAP 的失败条件**——它可以自标定，只是更容易漂。
- **两者对接是有工程量的。** **VGGT-X** (2509.25191) 这篇存在本身就是证据：把 VGGT 的输出直接喂给大规模 3DGS 训练，需要额外处理显存、点云噪声与位姿精度才跑得动，不是"换个位姿来源"这么轻。

> 💡 **实际怎么选** — 按拍摄条件、时间预算和**实测精度**选：静态、纹理够、离线、要对齐某个已有基准的口径 → COLMAP；在线、视图少或重叠差 → feed-forward；两个都想要 → feed-forward 出初值，再用 BA / COLMAP 精修。尺度要单独想：DUSt3R 式训练与 VGGT 都不保证是米，单目 COLMAP 同样不保证；要真实尺度就得引入标定物、已知基线或 metric 模型（§7.3），而**学出来的 metric 尺度不等于测量级精度**。

### 12.2　数值稳定（NeRF/3DGS 通用）

| 问题 | 症状 | 修复 |
| --- | --- | --- |
| Sigma 爆炸 | floater 充斥空间 | $\sigma$ 用 softplus 或 truncated；occupancy grid skip |
| Alpha 饱和 | 1-α 下溢 → T 全 0 | `(1-α).clamp(min=1e-10)` 或 log-space cumprod |
| Gaussian 退化 | 极小 scale / 极大 anisotropy | clamp scale lower bound；regularize anisotropy |
| Densify 爆炸 | Gaussian 数量飙到内存上限 | 加 max gaussian 数；周期 prune；reset opacity |
| SDS Janus | 多视角脸 / 头 | 加 view-conditioning（"front view" / "back view"）；MVDream |
| SDS over-sat | 颜色饱和 | CFG 降低；改用 VSD；或 negative prompt |

### 12.3　多机分布式 & 评测指标

**分布式**：NeRF / Instant-NGP / 3DGS 单 GPU 标准；大场景 3DGS 用 chunk-wise (VastGaussian, CityGaussian)；SDS/VSD 每 iter 跑 2 次 SD forward，8×A100 可显著提速；TRELLIS / Hunyuan3D 训练是大规模 multi-node DDP。

| 评测指标 | 用途 | 算法 |
| --- | --- | --- |
| **PSNR / SSIM / LPIPS** | 视图合成（重建）| 与真实视图对比 |
| **Chamfer Distance** | mesh 几何 | 两点云最近邻距离平均 |
| **F-Score (3D)** | mesh / point | precision + recall under threshold |
| **CLIP Score / CLIP-R-Prec** | text-to-3D 对齐 | render → CLIP 相似度 / 区分干扰 prompt |
| **ULIP / ULIP-2 对齐分数** | 3D ↔ 图像 ↔ 文本 语义对齐 | 在三模态共享空间里算相似度（ULIP 2212.05171 / ULIP-2 2305.08275；ULIP-2 用大模型自动生成语言描述，免掉人工标注） |
| **User study** | 最终质量 | MTurk / lab-internal |

**常用评测集**：**GSO** (Google Scanned Objects, 2204.11918) 1000+ 件真实扫描物体，是 single-image-to-3D 最常用的评测集；**Toys4K** (2101.07296) 约 4K 件 / 105 类，类别多样性好，常用于 few-shot 与部件级实验；**Objaverse-XL** (2307.05663) 1000 万+ 件，绝大多数工作把它当训练来源。**数据集本身不自带"训练集/测试集"身份**——同一批资产在这篇是训练数据、在那篇是评测数据都很常见，所以看到一个分数先问它的 split 是怎么切的，泛化结论只在那个 split 上成立。

## §13 25 高频面试题

按难度分 3 档（L1 必会 / L2 进阶 / L3 顶级 lab）。每题点开看答案要点 + 易踩坑。

### L1 必会题（任何 3D / vision 岗都会问）

<details>

<summary>Q1.NeRF 体渲染公式？</summary>

- $C(\mathbf{r}) = \int T(t)\sigma(\mathbf{r}(t))\mathbf{c}(\mathbf{r}(t),\mathbf{d})dt$

- $T(t) = \exp(-\int_{t_n}^t\sigma\,ds)$ 是透射率

- 离散化 → $\alpha$-compositing：$C \approx \sum T_i\alpha_i \mathbf{c}_i$，$\alpha_i = 1 - e^{-\sigma_i\delta_i}$

只写 $\sum \alpha_i \mathbf{c}_i$ 漏 $T_i$；或把 $\alpha_i$ 写成 $\sigma_i\delta_i$（一阶近似但严格错）。

</details>

<details>

<summary>Q2.为什么 NeRF 要 positional encoding？</summary>

- MLP 默认低频偏置（NTK 分析）

- $\gamma(p) = (\sin 2^k\pi p, \cos 2^k\pi p)_{k=0}^{L-1}$ 提供高频 basis

- 直接学 $(x,y,z) \to (\sigma,\mathbf{c})$ 出来的图糊；加 PE 后高频细节恢复

误以为 PE 是给 MLP 加位置（其实是给空间频率谱），或弄反 $\mathbf{x}$ vs $\mathbf{d}$ 的频率级数（$L=10$ vs $L=4$）。

</details>

<details>

<summary>Q3.NeRF 的 hierarchical sampling 是什么？</summary>

- 两个网络：coarse + fine

- coarse 均匀采 64 点，渲染得 weights $w_i = T_i\alpha_i$

- 把 $w$ 归一化为 PDF，按重要性采 128 个 fine 点（密集采在表面）

- Loss 同时监督两网络

说"只采一次更密集"——错过了 importance sampling 的核心。

</details>

<details>

<summary>Q4.Instant-NGP 为什么比 NeRF 快约 4+ 个数量级？</summary>

- **Hash 网格替代密集 grid**：固定 $T$ 大小哈希表，cache-friendly

- **Tiny MLP** (2 层 hidden 64) 替代大 MLP（NeRF 8 层 256）

- **多分辨率级联** + **occupancy grid** skip 空白区采样

- **CUDA fused kernel**（tiny-cuda-nn）

只说"用了哈希"——漏了 multi-resolution + tiny-MLP + occupancy skip 的组合贡献。

</details>

<details>

<summary>Q5.3DGS 的"高斯"是怎么定义的？</summary>

- 每个 Gaussian $G_i = (\mu_i, \Sigma_i, \alpha_i, c_i(\mathbf{d}))$

- $\mu \in \mathbb{R}^3$ 位置，$\Sigma \in \mathbb{R}^{3\times 3}$ 协方差

- $\Sigma = R S S^\top R^\top$ 分解（$R$ 用四元数，$S$ 用对角 + $\exp$），保证半正定

- $c(\mathbf{d})$ 用球谐 SH 系数（$\ell = 3$，48 参数）

只说"高斯分布"——漏了协方差参数化技巧 + SH color。

</details>

<details>

<summary>Q6.3DGS 渲染怎么做？</summary>

- 把 3D Gaussian 投影到 2D（$\Sigma' = JW\Sigma W^\top J^\top$）

- 按深度排序

- Front-to-back alpha-blending（与 NeRF $\alpha$-compositing 同源）

- 实际是 tile-based + CUDA radix sort

只说"光栅化"，不提投影 Jacobian / 排序 / alpha-blend。

</details>

<details>

<summary>Q7.3DGS 的 densification 怎么做？</summary>

- 高梯度 + 小 scale → **clone**（under-reconstruction）

- 高梯度 + 大 scale → **split**（over-reconstruction）

- 低 opacity 或过大 screen-size → **prune**

- 周期性 reset opacity 防 floater

把 clone 和 split 弄反；忘了 reset 这步。

</details>

<details>

<summary>Q8.NeRF vs 3DGS 对比？</summary>

- **NeRF**：隐式 (MLP)，渲染慢（ray march），editing 难

- **3DGS**：显式（点云），渲染快（rasterize），editing 易

- **质量**：3DGS PSNR 通常 ≥ NeRF；NeRF 在体积效应（烟雾 / 半透明）更好

- **业界趋势**：3DGS 主流，NeRF research-only

把两者当不可比较的不同事物——其实都是 volumetric scene rep，3DGS 是 explicit version of NeRF。

</details>

<details>

<summary>Q9.Marching Cubes 是什么？</summary>

- 输入 3D 标量场 + 阈值，输出三角网格

- 每 voxel 8 角点二值化（高/低于阈值）→ 256 种 lookup table

- Edge 上线性插值定顶点位置

- 不可微（lookup 离散）

说"找等高线"——MC 是 3D，等高线是 2D Marching Squares 的事。

</details>

<details>

<summary>Q10.SDS 大致是什么？</summary>

- 用 pretrained 2D diffusion (Stable Diffusion) 监督 3D 表示

- 渲染 $x = g(\theta, \pi)$，加噪 $x_t$，问 diffusion "这是 $y$ 的图吗"

- gradient $\propto (\epsilon_\phi(x_t; y) - \epsilon)\cdot \partial x/\partial \theta$

- DreamFusion (Poole et al. 2022 arXiv → ICLR 2023 Outstanding Paper) 提出

只说"用 SD 训 NeRF"，漏了 SDS gradient 的特殊形式（去掉 U-Net Jacobian）。

</details>

### L2 进阶题（research-oriented 岗位）

<details>

<summary>Q11.推导 NeRF 连续积分 → 离散 $\alpha$-compositing。</summary>

- $T$ 满足 $dT/dt = -\sigma T$，段内 $\sigma$ 常数 → $T(t_{i+1})/T(t_i) = e^{-\sigma_i\delta_i}$

- 段内颜色贡献 $\int_0^{\delta_i} T_i e^{-\sigma_i s}\sigma_i \mathbf{c}_i\,ds = T_i\mathbf{c}_i(1 - e^{-\sigma_i\delta_i})$

- 记 $\alpha_i = 1 - e^{-\sigma_i\delta_i}$，则 $C \approx \sum T_i\alpha_i \mathbf{c}_i$，$T_i = \prod_{j<i}(1 - \alpha_j)$

把 $\alpha_i$ 写成 $\sigma_i\delta_i$ 而非 $1 - e^{-\sigma_i\delta_i}$；或省了 ODE 求解过程。

</details>

<details>

<summary>Q12.推导 3DGS 的 3D→2D 投影 Jacobian。</summary>

- 透视投影 $\pi(\mathbf{x}) = (f_x x/z, f_y y/z)$ 非线性

- 一阶 Taylor：$\pi(\mathbf{x}) \approx \pi(\mu) + J(\mathbf{x}-\mu)$

- $J = \partial\pi/\partial\mathbf{x}|_\mu = \begin{pmatrix} f_x/z & 0 & -f_x x/z^2 \\ 0 & f_y/z & -f_y y/z^2 \end{pmatrix}$

- $\Sigma' = JW\Sigma W^\top J^\top$（$W$ 是 world→cam 旋转）

直接套 "covariance projection" 公式不推；或忘了 $W$ 这步（World→Cam 旋转）。

</details>

<details>

<summary>Q13.Instant-NGP 的 hash collision 如何消歧？</summary>

- **Multi-resolution 冗余**：粗 level $N_\ell^d \le T$ 不冲突，细 level 才冲突；MLP 可从粗-fine 共同推

- **稀疏激活**：有效 supervision 集中在 surface 附近；空白区采样点密度非零时仍有梯度（把 $\sigma$ 压向 0），$\sigma\to 0$ 后梯度幅度自然衰减，occupancy grid 更新后才真正跳过该处采样

- **MLP 后处理**：在 $L\times F$ 拼接特征上学非线性融合，可 disambiguate

- 没有 explicit collision resolution；靠"lazy resolution by sparsity + redundancy"

以为有 hash chaining 之类的传统消歧——实际是数据驱动 implicit 消歧。

</details>

<details>

<summary>Q14.SDS gradient 漏了哪项 Jacobian？为什么？</summary>

- Naive diffusion training grad：$(\epsilon_\phi - \epsilon)\cdot \partial \epsilon_\phi/\partial x_t \cdot \alpha_t \cdot \partial x/\partial \theta$

- SDS 把 $\partial \epsilon_\phi/\partial x_t$ **U-Net Jacobian** 扔掉

- 直觉：(1) 计算昂贵；(2) U-Net 没训练 second-order 稳定 → Jacobian 噪声大

- 代价：SDS 变成 mode-seeking KL，需要大 CFG (100) 才能逃 mean-mode → over-saturation

只说"为了简化"不说后果。或不知道 mode-seeking 是 KL 方向决定的。

</details>

<details>

<summary>Q15.Feed-forward 3DGS vs 逐场景 3DGS 优化：800× 什么时候不值得？</summary>

- **两条线各赢在哪**：feed-forward（GS-LRM 2-4 张已标定视图 0.23 秒 / Long-LRM 32 视图 960×540 约 1 秒）赢在延迟和稀疏视图下的鲁棒性；逐场景优化（§4，10-30 分钟）赢在能把这一个场景的每一张观测都榨干。**谁质量更高是条件性的**——Long-LRM 在它的房间级设定下报告与逐场景优化相当甚至更好，视图很密时优化通常仍占优

- **不值得的情况之一：视图密且离线**——几十到上百张图、不赶时间，优化能持续收敛，而 feed-forward 的先验在观测不足处是"编"出来的

- **不值得的情况之二：落在训练分布之外**——这些模型多在物体级 / 房间级数据上训，遇到训练里没覆盖的拍摄条件（尺度、材质、光照）就要靠猜，优化则只信眼前这批图。注意"室外"本身不等于分布外，要看具体模型训过什么

- **值得的情况**：交互 / 在线；视图极少（2-8 张）时优化本身欠约束，先验是净收益；批量跑上万个场景时总吞吐比单场景画质重要

- **工业界的实际做法**：feed-forward 出初值 + 优化精修——前馈输出的 Gaussian 就是一组普通 Gaussian，**可以接着训**，等于拿掉了优化最慢的冷启动那一段

- **换掉的是优化不是表示**：输出仍然是 Gaussian，仍按 §4.3 光栅化

把 800× 当成通用倍数——那是 Long-LRM 在它自己的设定（32 视图 960×540、A100）下相对逐场景优化的加速，换个视图数、分辨率或硬件就不是这个数。

</details>

<details>

<summary>Q16.Zero-1-to-3 / SyncDreamer / MVDream 区别？</summary>

- **Zero-1-to-3** (Liu 2023 ICCV)：input view + 相机 $\Delta R, \Delta T$ → single novel view；每次独立 sample

- **Zero-1-to-3++** (Shi 2023)：固定 6 个 anchor view，一次出多张（减 randomness）

- **SyncDreamer** (Liu 2024 ICLR)：在 latent 上**联合**预测多视图，cross-attention 让 views 互看 → 一致性更好

- **MVDream** (Shi 2024 ICLR)：text-to-multi-view（不需要 input image），4 视图同生成 + SDS 精化

只说"都是 novel view"——漏了独立 vs 联合 vs text-only 这条主线。

</details>

<details>

<summary>Q17.Mip-NeRF 怎么抗锯齿？</summary>

- Vanilla NeRF 把像素当 ray；不同分辨率下同像素对应不同尺度 → aliasing

- **Mip-NeRF** 把像素当 cone（视锥），cone 段近似 anisotropic Gaussian

- **IPE (Integrated Positional Encoding)**：$\mathbb{E}_{\mathbf{x}\sim\mathcal{N}(\mu,\Sigma)}[\gamma(\mathbf{x})]$ 有闭式解

- 高频系数被 $\Sigma$ 衰减 → multi-scale 自动平滑

只说"用 cone"，不讲 IPE 的高频衰减作用。

</details>

<details>

<summary>Q18.NeuS vs vanilla NeRF 提 mesh 的差别？</summary>

- vanilla NeRF：density 没明确 surface，提 mesh 要选 $\sigma$ 阈值（不稳）

- **NeuS** (Wang 2021 NeurIPS)：用 **SDF $d(\mathbf{x})$** 替换 density，定义 $\sigma$ via sigmoid 导数

- 表面 = $\{d = 0\}$，**良好定义**

- Marching Cubes 直接对 SDF 跑，质量明显更好

直接说"用 SDF"，但不讲 NeuS 怎么把 SDF 接到 NeRF 体渲染里。

</details>

<details>

<summary>Q19.新采集一批数据，用 COLMAP 还是 VGGT？</summary>

- **先问四件事**：拍摄条件（纹理、重叠、有没有动态物体）、要多快出结果、下游要不要绝对尺度、要不要和某个已有基准的位姿口径对齐

- **COLMAP**：成熟、可控，在纹理和重叠都够的静态场景上精度高，失败模式集中在 texture-less / 强镜面 / 动态 / 低重叠。**注意两点**：内参未知不等于它就跑不动（它可以自标定，只是更容易漂）；而且它**不是总能明显地失败**——图全都注册上了、重投影误差也不大，位姿照样可能是错的，该做的一致性检查一样得做

- **VGGT 系**（§7）：秒级、不依赖已标定内参、对重叠宽容得多。失败模式是**一直有输出但精度不保证**，同样需要自己验

- **尺度**：DUSt3R 式训练（预测与 GT 双边归一化）默认 up-to-scale，VGGT 只归一化 GT——两者输出都不保证是米。**单目 SfM 同样是 up-to-scale**，COLMAP 本身也不自带绝对尺度。要真实尺度：外部标定物、已知基线的双目/多传感器，或 MapAnything (2509.13414) 这类单独预测全局尺度的模型——但**学出来的 metric 尺度和测量级精度不是一回事**

- **基准口径**：有些重建基准的位姿确实来自 COLMAP，另一些用传感器真值或合成数据。想和哪条基线比，就对齐它用的那套口径

- **组合解**：feed-forward 出初值 → BA / COLMAP 精修；或先跑 VGGT 看能不能出结构，再决定值不值得等 COLMAP

答"VGGT 已经全面取代 COLMAP"，或反过来"COLMAP 精度总是更高"。该按拍摄条件、速度预算和**实测精度**来选。VGGT-X (2509.25191) 是个具体提醒：把 VGGT 输出接到大规模 3DGS 需要额外工程，衔接不是免费的。

</details>

<details>

<summary>Q20.3DGS 怎么提 mesh？</summary>

- vanilla 3DGS 不友好（ellipsoid 不是 surface）

- **SuGaR** (Guédon 2024 CVPR)：surface alignment loss + Poisson reconstruction

- **2DGS** (Huang 2024 SIGGRAPH)：把 ellipsoid 退化为 2D disk，对齐表面 → MC 提 mesh 更稳

- **GSDF** (Yu 2024)：joint train SDF head 与 3DGS

说"直接 MC"——3DGS 没有 density 场，直接 MC 不 work；必须先 surface-align。

</details>

### L3 顶级 lab 题（顶会 / industry 研究岗）

<details>

<summary>Q21.推导 DUSt3R 的置信度加权损失，并解释 $-\alpha\log C$ 这一项。</summary>

- **输出约定**：两个 head 都输出 pointmap $X^{1,1}, X^{2,1}$，**都在 $I^1$ 的相机系下**。位姿由 $X^{2,1}$ 与 $I^2$ 像素之间的 2D-3D 对应解 PnP 得到，**不是"把两张 pointmap 配准"**——它们本来就在同一个系里

- **回归项**：对有效像素 $i\in\mathcal{D}^v$，$\ell_{\text{regr}}(v,i)=\lVert \tfrac{1}{z}X^{v,1}_i-\tfrac{1}{\bar z}\bar X^{v,1}_i\rVert$——是欧氏范数，**不是平方误差**

- **尺度归一化**：$z=\text{norm}(X^1,X^2)=\frac{1}{\lvert\mathcal{D}^1\rvert+\lvert\mathcal{D}^2\rvert}\sum_v\sum_{i}\lVert X^v_i\rVert$，所有有效点到原点的平均距离，两张图**一起**算。除以它 → 损失对整体缩放不变 → DUSt3R 式训练默认 up-to-scale。要绝对尺度有别的做法，MapAnything 是一种：保留这种尺度不变的几何监督、**另外单独预测一个全局尺度**。归一化方式另论——VGGT 只归一化 GT，学的仍是归一化后的尺度

- **置信度形式**：$\mathcal{L}_{\text{conf}}=\sum_v\sum_i\big[\,C^{v,1}_i\,\ell_{\text{regr}}(v,i)-\alpha\log C^{v,1}_i\,\big]$，$C=1+\exp(\check C)\gt 1$。论文把 $-\alpha\log C$ 称为**正则项**

- **去掉 $\log$ 项**：$f(C)=C\ell$ 对 $C$ 单调不减，最优把 $C$ 推到定义域下界 $C\to 1$，**逐像素损失退回普通回归损失 $\ell$**。几何照样在学，退化的只是置信度 head——它变成常数、不携带信息

- **保留 $\log$ 项、对 $C$ 取下确界**（优化消元，不是概率意义的边缘化）：$f'(C)=\ell-\alpha/C$，$\ell\lt \alpha$ 时驻点 $C^\star=\alpha/\ell\gt 1$ 落在定义域内，$\ell\ge\alpha$ 时 $f'\gt 0$、下确界仍在 $C\to1^+$：

  $$\inf_{C\gt 1}f(C)=\begin{cases}\alpha+\alpha\log(\ell/\alpha), & 0\lt \ell\lt \alpha\\[2pt] \ell, & \ell\ge\alpha\end{cases}$$

- **结论要说准**：只有 $\ell\lt \alpha$ 那一段被换成对数形状，**$\ell\ge\alpha$ 的大残差仍然线性计入**——它不是把 outlier 压下去的鲁棒损失。$\alpha$ 就是这条分界线：残差小于 $\alpha$ 时抬高 $C$ 才划算（换来负的 $\alpha\log(\ell/\alpha)$），所以 $C$ 学到的是"这个像素我算得准不准"；天空、反光、动态物体这些算不准的地方，$C$ 停在 1 附近拿不到奖励

答成"边缘化掉置信度就得到对数鲁棒损失"：一是这是取下确界的优化消元不是积分边缘化，二是只有 $\ell\lt \alpha$ 半段是对数，大残差那半段是线性的。另一个失分点是说"去掉 log 项几何就学不动了"——去掉之后 $C\to 1$，损失正好退回普通回归损失。

</details>

<details>

<summary>Q22.VGGT 用一个 backbone 预测四种量——联合训练为什么赢过专才？DA3 的"一个 depth-ray 目标就够"又说明了什么？</summary>

- **四个 head 要说准**：camera（外参 + 内参）、depth（每张图的稠密深度）、point map（公共系下的 3D 点）、track（**给定查询像素，预测同一物理点在每张图上的 2D 位置**——是 2D 对应，不是"跨视图 3D 轨迹"）

- **联合训练的理由**：四个目标共享同一套几何——depth 加 camera 就能得到点云，点的跨图 2D 投影就是 track。它们互相约束，backbone 被逼着学一个自洽的 3D 表示；单任务模型没有别的 head 来暴露它的不自洽

- **VGGT 自己的观察**：论文报告**推理时用 depth head + camera head 组合出的点云，比 point map head 直接输出更准**。这是一条实验观察，**不能直接推出"多任务的收益一定落在共享表示而不在那个 head"**；它稳妥的含义是：训练时用哪些监督目标、推理时走哪条路径，是两个可以分开做的决定

- **DA3 (2511.10647) 的减法**：不要四个 head、不要特制架构，**plain transformer + 单一 depth-ray 预测目标**，在作者自建的基准上报告相对 VGGT 的平均相对提升——位姿 44.3%、几何 25.1%

- **怎么读这个对比**：两篇的数据、训练规模、评测集都不同，**跨论文的分数差不能证明是哪一种监督机制带来的收益**。能站得住的读法是：至少存在一种配置，单一预测目标就足以覆盖位姿与几何，说明 head 数量本身不是必要条件；要证因果得看同一篇里的消融

- **能补一句加分**：depth + ray 在信息上已经张成了位姿与点云，所以多 head 更像是把同一份信息拆开监督，代价是多出架构与损失权重的调参面

只说"多任务 = 更多监督 = 更好"；或反过来拿 DA3 的分数当"多任务无用"的证明——那是跨论文比较，撑不起因果结论。

</details>

<details>

<summary>Q23.AR mesh 生成的 token 预算：AMT / BPT / 树序列化各压掉了什么？为什么压到极限也还不够？</summary>

- **基线**：朴素序列 = 每个三角形 3 顶点 × 3 坐标 = **9 个坐标 token / 面**。序列长度 = 面数 × 每面 token 数，transformer 平方复杂度，所以 token 数直接换算成"最多能生成多少面"

- **MeshGPT** (2311.15475)：图卷积编码 + **残差量化**，每个面压成 **6 个码本 token**（不是 9——9 是它要打败的朴素基线）

- **AMT**（MeshAnything V2, 2408.02555）压的是**顶点重复**：相邻面共享顶点，多数面只需编码**一个新顶点**而非三个 → 序列长度约为朴素的**一半**

- **BPT** (2411.07025) 压的是**坐标表示**：**块索引**（坐标拆成块号 + 块内偏移）加上 **patch 内的顶点共享**，相对**朴素序列**缩短 **约 75%**，面数推过 **8K**

- **TreeMeshGPT** (2503.11629) 压的是**面片排列顺序**：沿邻接树自回归，从"待扩展边"的栈里取边向外长面。压缩后长度**约为原始的 22%**（≈ 2 token / 面）——注意是"剩 22%"不是"少 22%"。它还在序列构造里约束定向，**明显减少法线翻转**，但不等于完全免掉后处理

- **算一下 64K 面的 token 预算（L3 常追到这里）**：朴素 9/面 → 约 **576K** token；AMT 约一半 → 约 **288K**；树序列化 ≈2/面 → 约 **128K**。**压到最好仍是十万量级**，超出常规上下文，平方注意力也扛不住

- **所以 Meshtron (2412.09548) 走的是另一条路**：不指望压缩把序列变短到"塞得下"，而是让十万级序列本身可训——**hourglass 架构**在中段降采样 token，**滑动窗口注意力**把平方复杂度截断，做到 **64K 面 @ 1024 坐标分辨率**

- **Nexus (2607.13563) 的批评是针对自回归本身**：逐 token 生成会让前面的错误顺着序列往后传，压缩只是缩短序列，没有消除这条路径。它改成**由粗到细的顶点八叉树 diffusion** 先定顶点集合，再用 **topology embedding** 恢复边与面，因此不存在逐 token 的误差传播；代价是**论文自己说推理偏慢**

把 "22%" 读成"少了 22%"（实际是剩下 22%），或把 MeshGPT 说成 9 token/面（那是它的基线）。另一个失分点：把"压缩"当成解决方案的终点——分别采用这些方案时，最好的也还在十万级 token（各家的压缩率不相乘），Meshtron 靠的是架构而不是更狠的 tokenizer。

</details>

<details>

<summary>Q24.SLAT / VecSet / triplane 三种 3D latent，各自的架构取舍是什么？</summary>

- **先破一个常见误答**：三种 latent **都能按坐标查询解码**——triplane 是投影到三个平面双线性插值再逐元素相加（§8.3），VecSet 是拿查询坐标对 latent 集合做 cross-attention，SLAT 是按体素位置取。所以"能出几种格式"是**各家训了哪些 decoder** 的结果，不是表示本身的禁令

- **SLAT**（TRELLIS, 2412.01506）：$\boldsymbol z=\{(\boldsymbol z_i,\boldsymbol p_i)\}$，$N=64$ 体素分辨率、$L\approx 20\text{K}$ active voxel（约 8% 占用率）。$\boldsymbol p_i$ 只给**稀疏支撑**，细几何与外观都在 $\boldsymbol z_i$ 里。显式支撑让局部计算和按位置匹配的解码最顺手，TRELLIS 因此训了 3DGS / 场 / mesh 三个 decoder；代价是要多一个"生成结构"的阶段

- **VecSet**（3DShape2VecSet / Hunyuan3D ShapeVAE / Dora / Step1X-3D）：**无序 latent 集合，没有空间索引**。好处是集合可以很小——Dora (2412.17808) 用 Sharp Edge Sampling + dual cross-attention 压到 **1,280 codes**，在 Dora-bench 上与稠密的 XCube-VAE 相当（后者 >10,000）；序列短意味着后面的 DiT 便宜。代价是丢掉了局部性：同一份 latent 里没有"这块在哪"的结构可用

- **Triplane**（EG3D / Direct3D 2405.14832 / LRM）：三个**稠密** 2D 平面，最大优势是 2D 卷积与注意力的整套工程直接复用；代价是分辨率一上去平面本身的显存就涨，而且轴对齐投影有歧义（细长结构在某个平面上重叠）

- **解码目标也是一条轴**：CLAY 出 occupancy field，Hunyuan3D-DiT 与 Dora 出隐式场，Step1X-3D 出 **TSDF**，TRELLIS 走 FlexiCubes 出 mesh。同是 VecSet，解码目标可以完全不同

- **关于稀疏注意力的正确说法**：**稠密 token 一样可以上窗口化 / 稀疏注意力**。Direct3D-S2 (2505.17412) 的收益来自稀疏支撑让被跳过的那部分计算**真的不存在**；它报告的 3.9× / 9.6× 是 **SSA 算子相对 FlashAttention-2** 的加速，不是整模型的端到端倍数

说"VecSet 只能出 mesh""triplane 谈不上稀疏注意力"——这两句都不成立，前者是 decoder 选择问题，后者稠密 token 也能做。另一个失分点是把 SLAT 说成"就是体素网格"：体素只是 $\boldsymbol p_i$，$\boldsymbol z_i$ 是把 active voxel 投进多视图 DINOv2 特征图采样再压缩来的。

</details>

<details>

<summary>Q25.π³ 的置换等变修掉了参考视图锚定的什么毛病？NoPoSplat 的 canonical frame 又为什么不同？</summary>

- **参考视图锚定是什么**：DUSt3R 把所有 pointmap 放进 $I^1$ 的相机系，VGGT 一族同样要指定一张图当公共坐标原点

- **它坏在哪**：(1) 结果依赖"选了哪张"——同一批图换个顺序，输出就变，这是不该有的方差；(2) 参考图退化时，它仍然被当作全局基准。**注意不要编造"误差沿到参考图的距离累积"**：VGGT 这类是所有视图联合预测，不存在逐跳串联的链条

- **π³ (2507.13347) 的置换等变到底是什么**：$f(PX)=P f(X)$——它是对**排列行为**的约束。它确定性地消除的是**参考选择依赖和输入顺序依赖**，**不等于**"对坏视图不敏感"。π³ 报告的鲁棒性提升来自它的实验结果，不是从等变性直接推出来的

- **比较不同参考系下的两个重建时**：先把坐标变换对齐再比几何，否则量到的是"换了个系"而不是"几何变差了"

- **NoPoSplat (2410.24207) 反过来，故意保留锚定**：把**第一张输入图的相机系直接定义成 canonical space**，所有 Gaussian 都吐在那里，内参以 token 注入。它的"pose-free"限定在 Gaussian 网络本身

- **为什么不矛盾——两者治的病不同**：π³ 要的是**不依赖任何特定视图**（几何重建，视图之间本无主次）；NoPoSplat 要的是**不依赖位姿估计这一步**（稀疏视图下先估位姿再重建会传播误差，干脆用一个固定约定替掉）

- **L3 追问：canonical frame 和尺度是两件事，三个区分要说清**：(1) **选参考系只固定原点与坐标轴**，本身不决定尺度；(2) **已知内参也消不掉全局尺度自由度**——基线未知时，把场景与基线同比例缩放得到完全相同的图像，尺度不可观测；(3) NoPoSplat 的新视角评测**不是"先做尺度对齐"**，而是**固定 Gaussian、优化目标相机位姿**。顺带一提，AnySplat (2505.23716) 联合出内外参解决的是标定与配准，**同样不给绝对尺度**

把 "pose-free" 和 "reference-free" 当同义词：NoPoSplat 是 pose-free 但**不是** reference-free，它的 canonical frame 恰恰就是一个被特权化的参考视图。

</details>

## §A 附录：代码完整骨架 + 参考文献

### A.1　完整 from-scratch 代码包含

`volume_render()` (NeRF α-compositing 含数值稳定) · `positional_encoding()` (γ(p) Fourier features) · `gaussian_splat_forward()` (3DGS 教学版前向 + 投影 Jacobian) · `densify_and_prune()` (3DGS densification 启发式) · `sds_loss()` (SDS gradient surrogate) · `dust3r_conf_loss()` (DUSt3R 置信度加权 pointmap 损失，含尺度归一化与 metric 开关) · `marching_cubes_sketch()` (mesh 提取接口，用 scikit-image)。

### A.2　关键论文 reading list

- **NeRF 系**：Mildenhall 2020 ECCV (HM); Müller **Instant-NGP** SIGGRAPH 2022 Best; Barron **Mip-NeRF** / **360** ICCV 2021 / CVPR 2022; Wang **NeuS** + Yariv **VolSDF** NeurIPS 2021; Fridovich-Keil **Plenoxels** CVPR 2022; Chen **TensoRF** ECCV 2022.
- **3DGS 系**：Kerbl **3D Gaussian Splatting** SIGGRAPH 2023 Best; Huang **2D Gaussian Splatting** SIGGRAPH 2024; Luiten **Dynamic 3DGS** 3DV 2024; Wu **4DGS** CVPR 2024; Guédon **SuGaR** CVPR 2024.
- **Mesh / SDF**：Shen **DMTet** NeurIPS 2021 / **FlexiCubes** SIGGRAPH 2023.
- **SDS 系**：Poole **DreamFusion** arXiv 2022.09 → ICLR 2023 Outstanding; Wang **ProlificDreamer (VSD)** NeurIPS 2023 Spotlight; Lin **Magic3D** CVPR 2023; Chen **Fantasia3D** ICCV 2023; Tang **DreamGaussian** ICLR 2024; Yi **GaussianDreamer** CVPR 2024.
- **Single-image 3D**：Liu **Zero-1-to-3** ICCV 2023 / **One-2-3-45** NeurIPS 2023 / **SyncDreamer** ICLR 2024; Shi **Zero-1-to-3++** arXiv 2023 / **MVDream** ICLR 2024; Hong **LRM** arXiv 2023.11 → ICLR 2024; Tochilkin **TripoSR** arXiv 2024; Xu **InstantMesh** arXiv 2024; Boss **Stable Fast 3D** arXiv 2024.
- **3D Foundation Models**：Xiang **TRELLIS** 2412.01506 (Microsoft) / **TRELLIS.2** 2512.14692; Tencent **Hunyuan3D 2.0** 2501.12202 / **2.1** 2506.15442 / **2.5** 2506.16504 / **Omni** 2509.21245 / **Studio** 2509.12815 / **Buffalo 1.0** 2608.02711; Zhang **CLAY** SIGGRAPH 2024 (2406.13897); **Direct3D** 2405.14832 / **Direct3D-S2** 2505.17412; **Dora** 2412.17808 (CVPR 2025); **Step1X-3D** 2505.07747; Meta **AssetGen 2** 2605.26137.
- **Feed-forward 重建**：**DUSt3R** 2312.14132 (CVPR 2024) / **MASt3R** 2406.09756; **VGGT** 2503.11651 (CVPR 2025 Best Paper); **Fast3R** 2501.13928 (CVPR 2025); **CUT3R** 2501.12387; **π³** 2507.13347; **MapAnything** 2509.13414 (3DV 2026); **Depth Anything 3** 2511.10647; **VGGT-Ω** 2605.15195 (CVPR 2026 Oral); **MV-DUSt3R+** 2412.06974; **VGGT-X** 2509.25191; 综述 2507.08448.
- **Feed-forward 3DGS**：**LGM** 2402.05054; **GS-LRM** 2404.19702; **Long-LRM** 2410.12781 / **Long-LRM++** 2512.10267; **DepthSplat** 2410.13862 (CVPR 2025); **NoPoSplat** 2410.24207; **AnySplat** 2505.23716.
- **原生 mesh 生成**：**MeshGPT** 2311.15475; **MeshAnything** 2406.10163 / **V2** 2408.02555; **BPT** 2411.07025; **TreeMeshGPT** 2503.11629 (CVPR 2025); **Meshtron** 2412.09548; **DeepMesh** 2503.15265; **Nexus** 2607.13563.
- **部件级 / sim-ready**：**PartGen** 2412.18608; **HoloPart** 2504.07943; **PartCrafter** 2506.05573; **PartPacker** 2506.09980 (NVIDIA); **EmbodiedGen** 2506.10600; **Artiverse** 2605.24403.
- **数据集 / 评测**：**Objaverse-XL** 2307.05663（1000 万+）; **GSO** 2204.11918（真实扫描物体）; **Toys4K** 2101.07296（约 4K 件 / 105 类）; **ULIP** 2212.05171 / **ULIP-2** 2305.08275（3D-图像-文本对齐）.

### A.3　Embodied AI / AR / VR 常见追问

3DGS 接物理引擎 → 先 2DGS / SuGaR 提 mesh → IsaacSim / MuJoCo；NeRF 动态化 → 4DGS / D-NeRF / K-Planes；AR 实时 3DGS → mobile-friendly (PostShot, Luma) + 剪枝 / 量化；3D 数据不足 → Objaverse-XL (TRELLIS) 或 2D 蒸馏 (DreamFusion 系) 或 multi-view 启发式 (MVDream)；扫一批图重建 → 先 VGGT 出结构再决定要不要等 COLMAP（§12.1）；生成资产要进仿真器 → 光有 mesh 不够，还缺部件、关节、物理材质（§9.6）。

---

**3D Generation Quick Reference** · 主要参考：Mildenhall 2020 (NeRF), Müller 2022 (Instant-NGP), Kerbl 2023 (3DGS), Poole 2022/ICLR 2023 (DreamFusion), Wang 2023 (VSD), Wang 2024 (DUSt3R), Wang 2025 (VGGT), Xiang 2024 (TRELLIS), Tencent 2025-26 (Hunyuan3D 2.x / Omni / Buffalo). 涵盖：NeRF 体渲染推导、Instant-NGP hash 网格、3DGS 投影 Jacobian、SDS / VSD 梯度推导、DUSt3R 置信度加权损失、feed-forward 重建与 feed-forward 3DGS、single-image 3D、3D foundation models 的 latent 表示轴、原生 mesh 生成、部件级与 sim-ready 资产。Embodied AI / AR / VR 必备。
