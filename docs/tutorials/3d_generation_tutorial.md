## §0 TL;DR Cheat Sheet

> 💡 **9 句话搞定 3D Generation** — Embodied AI / AR / VR 面试核心要点（详见后文 §1–§11 推导）。

1. **三大表示**：**NeRF**（隐式神经场 + 体渲染）、**3DGS**（显式 Gaussian 点云 + 光栅化）、**Mesh / SDF**（显式表面 / 隐式距离场）。重建质量与速度的 sweet spot：3DGS（Kerbl 2023 SIGGRAPH Best Paper）。

2. **NeRF 核心公式**：$C(\mathbf{r}) = \int_{t_n}^{t_f} T(t)\sigma(\mathbf{r}(t))\mathbf{c}(\mathbf{r}(t),\mathbf{d})\,dt$，其中 $T(t) = \exp\!\left(-\int_{t_n}^{t}\sigma(\mathbf{r}(s))\,ds\right)$ 是 transmittance。离散化得到 $\alpha$-compositing：$C \approx \sum_i T_i (1-e^{-\sigma_i\delta_i})\mathbf{c}_i$。

3. **Instant-NGP** (Müller 2022 SIGGRAPH)：**多分辨率 hash 网格** + tiny MLP，约 4+ 个数量级加速；hash collision 由 MLP 在含碰撞表项上自动学习消歧（被 loss + 多尺度冗余共同压制）。

4. **3DGS 核心**：场景表示为一组 3D Gaussian $\{\mu_i, \Sigma_i, \alpha_i, c_i(\mathbf{d})\}$，**可微光栅化**通过将 3D 协方差用 Jacobian $J$ 投影到 2D：$\Sigma' = J W \Sigma W^\top J^\top$，按深度排序后做 front-to-back alpha-blending。

5. **DreamFusion SDS** (Poole et al. 2022 arXiv → ICLR 2023 Outstanding)：用 pretrained 2D diffusion 监督 3D 表示：$\nabla_\theta \mathcal{L}_\text{SDS} = \mathbb{E}_{t,\epsilon}[w(t)(\epsilon_\phi(x_t;y,t)-\epsilon)\,\partial x/\partial \theta]$，**故意去掉 U-Net 对 $x_t$ 求导的 Jacobian 项**，使得训练 simulation-free。代价：mode-seeking → over-saturation / Janus。

6. **VSD** (Wang 2023 NeurIPS, ProlificDreamer)：把 3D 参数 $\theta$ 视为 random variable $\mu(\theta)$，**最小化的是渲染加噪图像分布**之间的 KL：$\mathbb{E}_t\big[D_\text{KL}\big(q_\mu^t(x_t|y)\,\|\,p_\phi^t(x_t|y)\big)\big]$。梯度形式为 **relative score** $\nabla_\theta \approx (\epsilon_\phi(x_t;y,t) - \epsilon_\psi(x_t;y,t,\pi))\,\partial x/\partial\theta$，其中 $\epsilon_\psi$ 是 LoRA 微调的辅助 score。CFG 可从 100 降至 7.5。

7. **Single-image / Few-view 3D**：Zero-1-to-3 (Liu 2023 ICCV) 用 viewpoint conditioned diffusion；SyncDreamer / MVDream 学多视图联合一致性；TripoSR / InstantMesh / Stable Fast 3D 把 image-to-mesh 推到秒级（TripoSR ~0.5 秒、InstantMesh ~10 秒）。

8. **3D Foundation Models (2024-25 开源)**：**Trellis** (Microsoft 2024) 用 structured latent + flow matching；**Hunyuan3D-2** (Tencent 2025) shape→texture 两阶段；**CLAY** (Zhang 2024 SIGGRAPH, arXiv:2406.13897) 大尺度 latent diffusion = 多分辨率 VAE + latent DiT。

9. **Embodied AI 关键应用**：Sim2Real 资产生成、NeRF/3DGS 作为可微 simulator、language-conditioned 3D affordance。**面试常见交叉**：NeRF SLAM、Gaussian-Splat scene editing、3D 物理一致性。

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

## §7 Single-Image / Few-View 3D 生成

更实用的设定：**给一张图，生成 3D**。

### 7.1　Zero-1-to-3 范式（novel view via diffusion）

**Zero-1-to-3** (Liu 2023 ICCV)：用 Objaverse 上 finetune Stable Diffusion，让它接收 (input view, target camera) → output novel view。

- Input：单图 $x$ + 相对相机位姿 $\Delta R, \Delta T$
- Diffusion conditioning：image embedding (CLIP) + camera embedding (sinusoidal)
- Output：在 $\Delta R, \Delta T$ 视角下的图

**用法**：给一个 input view，sample 16-32 个 novel view，再用 NeRF / 3DGS 重建。

**衍生**：
- **Zero-1-to-3++** (Shi 2023)：固定生成 6 个 anchor view（方位角在 30°/90°/150°/210°/270°/330° 每 60° 均匀环绕一周，仰角在 -10° 与 20° 之间交替排列，即 interleaved elevation + uniform azimuth，而非"一个北极视角 + 四个平视角 + 一个俯视角"），减少 randomness
- **SyncDreamer** (Liu 2024 ICLR)：在 latent 上**联合**预测多视图（cross-attention 让 views 看到彼此），保证 3D 一致
- **MVDream** (Shi 2024 ICLR)：text-to-multi-view，4 视图同时生成；后接 SDS 精化

### 7.2　One-2-3-45 / InstantMesh / TripoSR / Stable Fast 3D

| 方法 | 输入 | 输出 | 速度 | 关键 |
| --- | --- | --- | --- | --- |
| **One-2-3-45** (Liu 2023 NeurIPS) | 单图 | mesh | 45 秒 | Zero-1-to-3 → SparseNeuS |
| **One-2-3-45++** (Liu 2024) | 单图 | mesh | 60 秒 | 多视图 + SDF |
| **TripoSR** (Tochilkin 2024, Stability+Tripo) | 单图 | NeRF/mesh | 0.5-2 秒 | LRM (Large Reconstruction Model) 风格 transformer |
| **InstantMesh** (Xu 2024, arXiv:2404.07191) | 单图 | mesh | ~10 秒 | Zero-1-to-3++ 多视图 → sparse-view recon transformer |
| **Stable Fast 3D** (SF3D, Stability 2024) | 单图 | textured mesh | ~0.5 秒 | TripoSR 后继；加 illumination disentangle + UV unwrap |

**LRM (Hong et al. 2023 arXiv → ICLR 2024) 设定**：把图当 token + Plucker ray embedding，transformer 输出 NeRF triplane。这是 TripoSR / InstantMesh 的母模型。

### 7.3　LRM Triplane 表示（**面试高频**）

- **Triplane** (Chan 2022 EG3D)：3 个轴对齐 2D 平面（XY, YZ, XZ），共 $3 \times C \times N \times N$ 维
- 查询 3D 点 $(x, y, z)$：在每个平面双线性插值 → **三个平面特征逐元素相加**（EG3D 是 sum，不是 concat）→ 小 MLP → $(\sigma, \mathbf{c})$
- 优点：比 voxel grid 显存少（$O(N^2)$ vs $O(N^3)$），比 hash grid 更 dense 适合 transformer 输出
- LRM / TripoSR / InstantMesh 都让 transformer 直接 regress triplane tokens

## §8 3D Foundation Models（2024 开源浪潮）

### 8.1　Trellis (Microsoft 2024, 开源)

**Trellis** (Xiang 2024 arXiv) 是首个尝试做"3D 的 Stable Diffusion"开源工作。

- **Structured Latent (SLAT)**：把 3D 资产编码到 voxel 上的稀疏 latent grid——既保留空间结构（适合 sparse conv / sparse attention），又紧凑（仅 active voxel 存 latent）
- **3D VAE**：把 mesh + texture (signed distance field 派生) → SLAT
- **Flow matching prior**：在 SLAT 上跑 rectified flow，conditioned on text/image
- **多 decoder**：从 SLAT decode 出 NeRF / 3DGS / mesh 三种表示（同一 latent，可选输出格式）
- **训练数据**：Objaverse-XL 子集 + 内部高质量集
- **效果**：text-to-3D / image-to-3D，几秒到几十秒，质量超过 SDS 系列

### 8.2　Hunyuan3D-1 / -2 (Tencent 2024-25, 开源)

**Hunyuan3D** 走 **shape-then-texture** 两阶段路线。

- **Hunyuan3D-1** (Yang 2024 arXiv)：
  - Stage 1: text/image → multi-view image (Zero-1-to-3 系)
  - Stage 2: multi-view → 3D mesh (LRM-like reconstructor)
  - 几秒到几十秒输出 textured mesh
- **Hunyuan3D-2** (Tencent 2025, arXiv 2501.12202)：
  - **Hunyuan3D-DiT**：geometry-only DiT 在 SDF latent 上生成 mesh
  - **Hunyuan3D-Paint**：multi-view PBR texture diffusion，UV space refinement
  - 高质量 PBR texture（实战可用于游戏 / VR 资产）
- **开源**：HuggingFace 上完整权重 + 推理代码

### 8.3　CLAY (Zhang 2024 SIGGRAPH, arXiv:2406.13897)

- **多分辨率 VAE + latent DiT**：3D shape VAE 把 mesh 编码到 neural field latent，再在 latent 上跑 DiT diffusion（区别于 3DShape2VecSet，Zhang et al. 2023, arXiv:2301.11445，那是一个单独的 vector-set 表示工作）
- 大规模训练（Objaverse-XL + 内部清洗集）
- 输出 SDF → marching cubes → mesh
- 加 PBR texture stage（类似 Hunyuan3D-2）

**Rodin** (Microsoft 2023, 商业)：早期 text-to-3D-avatar 产品级系统，diffusion on triplane，主打 character / avatar。

### 8.4　对比表

| 方法 | 表示 | Prior | 训练规模 | 开源 |
| --- | --- | --- | --- | --- |
| **Trellis** | Structured Latent (SLAT) + 多 decoder | Rectified Flow | Objaverse-XL 子集 | ✅ |
| **Hunyuan3D-2** | SDF latent (Shape DiT) + UV texture diff | Diffusion | 内部大规模集 | ✅ |
| **CLAY** | 多分辨率 VAE latent (latent DiT) | Diffusion | Objaverse-XL + 内部 | 部分 |
| **Rodin** | Triplane | Diffusion | 商业内部 | ❌ |
| **TripoSR / SF3D** | NeRF/mesh feedforward | 无 prior，纯 regression | Objaverse 类 | ✅ |

> 💡 **架构选择直觉** — 大 scene / general object 用 **Trellis 风格 SLAT**（保留空间结构）；高质量 single mesh 用 **CLAY 风格 latent DiT**（多分辨率 VAE latent，global attention）；快速推理用 **LRM/TripoSR feedforward**（不做 diffusion，直接 regress）。

## §9 复杂度 / 资源对比

| 方法 | 训练 | 推理 (一帧) | 显存 (训练) | 显存 (模型) |
| --- | --- | --- | --- | --- |
| NeRF vanilla | 1-2 天 | 数秒 | 8 GB | <10 MB MLP |
| Instant-NGP | 5 秒 - 5 分钟 | 30 fps+ | 4-12 GB | 100-500 MB hash |
| 3DGS | 10-30 分钟 | 100 fps+ | 6-24 GB | 100 MB - 1 GB Gaussian |
| 2DGS | 与 3DGS 接近 | 与 3DGS 接近 | 类似 | 类似 |
| DreamFusion (NeRF+SDS) | 2 hr / 物体 | — | 12 GB | NeRF 本身 |
| DreamGaussian (3DGS+SDS) | 2 分钟 / 物体 | — | 8-16 GB | — |
| ProlificDreamer (VSD) | 3-6 hr / 物体 | — | 24 GB | — |
| TripoSR feedforward | 训练 50 GPU 天 | 0.5 秒 (A100) | inference 6 GB | 1.5 GB |
| Trellis | 训练 100+ GPU 天 | 数秒 | inference 16 GB | 数 GB |
| Hunyuan3D-2 | 训练大集群 | 数十秒 | inference 24+ GB | 多模型组合 |

## §10 与相关方法对比 & Embodied AI 应用

### 10.1　3D-vs-2D 生成关键区别

| 维度 | 2D 生成 (Stable Diffusion) | 3D 生成 |
| --- | --- | --- |
| **数据量** | LAION-5B 50亿图 | Objaverse-XL 1000万件（小 500×） |
| **数据格式** | 图像（统一 RGB） | mesh / SDF / point cloud / NeRF / 3DGS（**碎片化**） |
| **训练 prior** | 直接 train diffusion | 用 2D diffusion 蒸馏 (SDS / Zero-1-to-3) **或** 用 3D-native diffusion (Trellis / CLAY) |
| **评测** | FID, CLIP score | Chamfer / IoU / PSNR (recon) + perceptual + user study |
| **下游** | 直接出图 | 出资产 → 渲染 / 仿真 / 编辑 |

### 10.2　Embodied AI / AR / VR 实战路线

| 任务 | 推荐表示 | 关键工具链 / 约束 |
| --- | --- | --- |
| **Sim2Real 资产** | mesh (PBR) | Trellis / Hunyuan3D-2 → IsaacSim / MuJoCo |
| **室内大场景** | 3DGS | COLMAP → 3DGS（chunk-wise 用 VastGS / CityGS） |
| **NeRF/3DGS as simulator** | NeRF / 3DGS + physics | DreamGaussian-Sim / Splatting Physics |
| **3D affordance / manipulation** | point cloud / 3DGS feature | OpenScene / LERF / RVT / 3D Diffuser Actor |
| **AR 物体扫描** | 3DGS（光照真实 + 实时）| mobile 算力（PostShot / Luma），剪枝 / 量化 |
| **VR 大场景** | 3DGS (large-scale) | 60 fps stereo + 6DoF |
| **Avatar** | mesh + LBS 或 3DGS avatar | 实时表情 / 头发 |
| **Object insertion** | mesh + PBR | 环境光照一致（IBL）|

> ⚠️ **Embodied AI 面试追问示例** — "做 NeRF 物理仿真器最大挑战？" 要点：NeRF 是 radiance，没 mass / friction → 需手动叠物理 prior；mesh 提取有 floater → 碰撞检测难；可微但 backward 慢；**业界更多用 3DGS / mesh 而非 vanilla NeRF**。

## §11 工程实战 & 易踩坑

### 11.1　COLMAP / SfM 前处理（重建必经）

输入多视图 → 输出内参 $K$ + 外参 $\{R_i, t_i\}$ + 稀疏点云；标准流程 SIFT → matching → incremental SfM → bundle adjustment。**常见坑**：texture-less / 镜面物体 SfM 失败；动态物体污染外参。

### 11.2　数值稳定（NeRF/3DGS 通用）

| 问题 | 症状 | 修复 |
| --- | --- | --- |
| Sigma 爆炸 | floater 充斥空间 | $\sigma$ 用 softplus 或 truncated；occupancy grid skip |
| Alpha 饱和 | 1-α 下溢 → T 全 0 | `(1-α).clamp(min=1e-10)` 或 log-space cumprod |
| Gaussian 退化 | 极小 scale / 极大 anisotropy | clamp scale lower bound；regularize anisotropy |
| Densify 爆炸 | Gaussian 数量飙到内存上限 | 加 max gaussian 数；周期 prune；reset opacity |
| SDS Janus | 多视角脸 / 头 | 加 view-conditioning（"front view" / "back view"）；MVDream |
| SDS over-sat | 颜色饱和 | CFG 降低；改用 VSD；或 negative prompt |

### 11.3　多机分布式 & 评测指标

**分布式**：NeRF / Instant-NGP / 3DGS 单 GPU 标准；大场景 3DGS 用 chunk-wise (VastGaussian, CityGaussian)；SDS/VSD 每 iter 跑 2 次 SD forward，8×A100 可显著提速；Trellis / Hunyuan3D 训练是大规模 multi-node DDP。

| 评测指标 | 用途 | 算法 |
| --- | --- | --- |
| **PSNR / SSIM / LPIPS** | 视图合成（重建）| 与真实视图对比 |
| **Chamfer Distance** | mesh 几何 | 两点云最近邻距离平均 |
| **F-Score (3D)** | mesh / point | precision + recall under threshold |
| **CLIP Score / CLIP-R-Prec** | text-to-3D 对齐 | render → CLIP 相似度 / 区分干扰 prompt |
| **User study** | 最终质量 | MTurk / lab-internal |

## §12 25 高频面试题

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

<summary>Q15.VSD 如何缓解 SDS 的 over-saturation？</summary>

- SDS：拉向 prior $p_\phi$ 的 mode；需 CFG=100 强化 → over-saturation

- **VSD**：把 3D 参数 $\theta$ 视为 random variable $\mu(\theta)$，最小化 KL(rendered dist || prior)

- 引入**辅助 score** $\epsilon_\psi$（LoRA 微调 SD）跟踪当前 $\mu$ 的 score

- gradient = $(\epsilon_\phi - \epsilon_\psi)\cdot \partial x/\partial \theta$ —— **relative score**，不需大 CFG

- $\epsilon_\psi$ 是 KL 目标对 $\theta$ 求导后 $q$ 分布自身 score 对应的必要项（丢了会退化回 SDS），不是只减方差的 RL actor-critic value baseline

说 VSD 用 "variational" 但讲不清 $\epsilon_\psi$ 替代 raw noise 的角色。

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

<summary>Q19.LRM 系列（TripoSR / InstantMesh）核心？</summary>

- **Triplane** 表示：3 个轴对齐 2D 平面，$O(N^2)$ 显存

- Transformer 把图像 token + Plucker ray embedding → regress triplane tokens

- 推理 feedforward（无 SDS / 无 iterative 优化），**秒级出 3D**（TripoSR ~0.5 秒、InstantMesh ~10 秒）

- TripoSR (Stability+Tripo 2024) / InstantMesh (Xu 2024) / SF3D (2024) 都属此族

把它们当成 SDS 系列——错，LRM 完全 feedforward；不算 distillation。

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

<summary>Q21.手推 NeRF 离散 $\alpha$-compositing。</summary>

- ODE $dT/dt = -\sigma(t) T(t)$，初值 $T(t_n) = 1$ → $T(t) = \exp(-\int_{t_n}^t \sigma\,ds)$

- 段内 $[t_i, t_{i+1}]$ 上 $\sigma$ 常数 $= \sigma_i$，所以 $T(t_{i+1}) = T(t_i)e^{-\sigma_i\delta_i}$

- 段间累积 $T_i = T(t_i) = \prod_{j<i} e^{-\sigma_j\delta_j} = \prod_{j<i}(1 - \alpha_j)$，其中 $\alpha_j = 1 - e^{-\sigma_j\delta_j}$

- 段内颜色贡献 $\int_{t_i}^{t_{i+1}} T(t)\sigma_i\mathbf{c}_i\,dt = \mathbf{c}_i T_i \int_0^{\delta_i}\sigma_i e^{-\sigma_i s}ds = T_i\mathbf{c}_i(1 - e^{-\sigma_i\delta_i}) = T_i\alpha_i\mathbf{c}_i$

- 合成 $C \approx \sum_i T_i\alpha_i \mathbf{c}_i$

- **关键**：$\alpha_i = 1 - e^{-\sigma_i\delta_i}$ 严格 vs $\alpha_i \approx \sigma_i\delta_i$ 一阶近似（在 $\sigma\delta \ll 1$ 时一致）

省略 ODE 推导直接套结论；或在 $\sigma\delta$ 大时用 $\sigma_i\delta_i$ 替 $\alpha_i$ 出错。

</details>

<details>

<summary>Q22.Instant-NGP hash collision 如何被 MLP 自动消歧？</summary>

- **冲突发生场景**：fine level 的格点数 $N_\ell^d > T$（hash 表大小），多个格点映到同一 entry

- **稀疏激活**：场景大部分 voxel 是 background，**仅 surface 附近 voxel 有非零监督梯度**——冲突的"两个 background entry"得不到信号，互不污染

- **多分辨率冗余**：粗 level $N_\ell^d \le T$ 保证 unique；细 level 提供 detail。即使细 level 冲突，粗 level 的非冲突特征已唯一识别该点

- **MLP 后处理**：tiny MLP 在 $L\times F$ 拼接特征上学非线性融合，遇到冲突 entry 时可以用**其他 level 的非冲突特征 disambiguate**

- **梯度自动调节**：训练时高梯度自然集中在 surface entry；冲突 entry 若同时在表面（罕见）会被 loss 推到妥协位置（取多次采样平均）

- **物理直觉**：与其代价昂贵地搞 perfect hash，不如允许冲突 + 用数据驱动 implicit 消歧（"lazy collision resolution"）

说"哈希冲突由 MLP 解决"但讲不清是哪些 mechanism（稀疏性 + 多尺度 + MLP 非线性）共同作用。

</details>

<details>

<summary>Q23.推导 3DGS 3D→2D 协方差投影 Jacobian。</summary>

- World → Camera：刚体变换 $\mathbf{x}_\text{cam} = W\mathbf{x} + t$；协方差只受旋转影响，$\Sigma_\text{cam} = W\Sigma W^\top$

- Camera → Screen：透视投影 $\pi(x, y, z) = (f_x x/z, f_y y/z)$ 非线性

- 在均值 $\mu_\text{cam}$ 处一阶 Taylor：$\pi(\mathbf{x}) \approx \pi(\mu_\text{cam}) + J(\mathbf{x} - \mu_\text{cam})$，$J = \partial\pi/\partial\mathbf{x}|_{\mu_\text{cam}}$

- $J = \begin{pmatrix} f_x/z & 0 & -f_x x/z^2 \\ 0 & f_y/z & -f_y y/z^2 \end{pmatrix} \in \mathbb{R}^{2\times 3}$

- $\text{Cov}[\pi(\mathbf{x})] = J\Sigma_\text{cam} J^\top = JW\Sigma W^\top J^\top \in \mathbb{R}^{2\times 2}$

- 这是 EWA splatting (Zwicker 2001) 的经典推论；3DGS 直接沿用

- 实际实现还加 $0.3 I$ low-pass filter（anti-aliasing）

不会做一阶 Taylor 把非线性投影线性化；或漏了 World→Cam 那步。

</details>

<details>

<summary>Q24.SDS gradient 丢掉哪项 Jacobian？为什么"反而 work"？</summary>

- **Naive diffusion training gradient**：

  $\nabla_\theta \mathcal{L}_\text{diff} = \mathbb{E}[w(t)\cdot 2(\epsilon_\phi - \epsilon)\cdot \underbrace{\partial \epsilon_\phi/\partial x_t}_{\text{U-Net Jacobian}}\cdot \alpha_t \cdot \partial x/\partial\theta]$

- **SDS** 扔掉 U-Net Jacobian $\partial \epsilon_\phi/\partial x_t$：

  $\nabla_\theta \mathcal{L}_\text{SDS} = \mathbb{E}[w(t)(\epsilon_\phi - \epsilon)\cdot \partial x/\partial\theta]$

- **为什么扔掉 reasonable**：
  - U-Net Jacobian 计算昂贵（H×W×3 输入 → H×W×3 输出的 second-order）
  - U-Net 没训练 second-order 稳定，Jacobian 数值差
  - $(\epsilon_\phi - \epsilon)$ 本身就是 score 的代理（$\epsilon_\phi/\sigma_t \approx -\nabla_{x_t}\log p_\phi$），扔掉 Jacobian 相当于用 first-order score 信号

- **代价**：SDS 数学上等价 mode-seeking KL（往 prior 的 mode 跑），加大 CFG (100) 才能逃出 mean-blur

- **症状**：over-saturation（颜色饱和）+ Janus（多视角同一面孔）+ over-smoothing（细节糊）

只说"为简化扔了 Jacobian"，不解释 mode-seeking 后果 + 为什么需要大 CFG。

</details>

<details>

<summary>Q25.VSD 为什么能在小 CFG 下避免 over-saturation？</summary>

- **SDS 视角**：把 $\theta$ 当点估计；gradient 拉向 $p_\phi(\cdot|y)$ mode；大 CFG 让 mode 更尖 → over-saturation

- **VSD 视角**：把 $\theta$ 当 random variable $\mu(\theta)$；**最小化的是渲染加噪图像分布**之间的 KL：$\mathbb{E}_t[D_\text{KL}(q_\mu^t(x_t|y) \,\|\, p_\phi^t(x_t|y))]$（不是直接在 $\theta$ 域上对一个 3D prior 求 KL）

- 引入**辅助 score** $\epsilon_\psi$（用 LoRA 微调 Stable Diffusion）跟踪当前 rendered 分布的 score

- **VSD gradient**：

  $\nabla_\theta \mathcal{L}_\text{VSD} = \mathbb{E}[w(t)(\epsilon_\phi - \epsilon_\psi)\cdot \partial x/\partial \theta]$

  即 **relative score**（target prior score $-$ current rendered score）

- **几何直觉**：从"我现在在哪"指向"prior 在哪"——是局部"梯度方向"而非全局 mode；不需大 CFG 锐化

- **不是 RL baseline**：$\epsilon_\psi$ 是 KL 目标对 $\theta$ 求导后 $q$ 分布自身 score 对应的必要成分，丢掉它会把梯度目标退化回 SDS 的 mode-seeking（改变期望梯度方向），不是 RL actor-critic 中只减方差、不改变期望梯度的 value baseline

- **效果（ProlificDreamer）**：CFG 可降至 7.5，颜色自然；几何更细；可维护多 mode（多样性）

只说"VSD 引入变分推断"不讲 $\epsilon_\psi$ 角色 + relative gradient 视角。

</details>

## §A 附录：代码完整骨架 + 参考文献

### A.1　完整 from-scratch 代码包含

`volume_render()` (NeRF α-compositing 含数值稳定) · `positional_encoding()` (γ(p) Fourier features) · `gaussian_splat_forward()` (3DGS 教学版前向 + 投影 Jacobian) · `densify_and_prune()` (3DGS densification 启发式) · `sds_loss()` (SDS gradient surrogate) · `marching_cubes_sketch()` (mesh 提取接口，用 scikit-image)。

### A.2　关键论文 reading list

- **NeRF 系**：Mildenhall 2020 ECCV (HM); Müller **Instant-NGP** SIGGRAPH 2022 Best; Barron **Mip-NeRF** / **360** ICCV 2021 / CVPR 2022; Wang **NeuS** + Yariv **VolSDF** NeurIPS 2021; Fridovich-Keil **Plenoxels** CVPR 2022; Chen **TensoRF** ECCV 2022.
- **3DGS 系**：Kerbl **3D Gaussian Splatting** SIGGRAPH 2023 Best; Huang **2D Gaussian Splatting** SIGGRAPH 2024; Luiten **Dynamic 3DGS** 3DV 2024; Wu **4DGS** CVPR 2024; Guédon **SuGaR** CVPR 2024.
- **Mesh / SDF**：Shen **DMTet** NeurIPS 2021 / **FlexiCubes** SIGGRAPH 2023.
- **SDS 系**：Poole **DreamFusion** arXiv 2022.09 → ICLR 2023 Outstanding; Wang **ProlificDreamer (VSD)** NeurIPS 2023 Spotlight; Lin **Magic3D** CVPR 2023; Chen **Fantasia3D** ICCV 2023; Tang **DreamGaussian** ICLR 2024; Yi **GaussianDreamer** CVPR 2024.
- **Single-image 3D**：Liu **Zero-1-to-3** ICCV 2023 / **One-2-3-45** NeurIPS 2023 / **SyncDreamer** ICLR 2024; Shi **Zero-1-to-3++** arXiv 2023 / **MVDream** ICLR 2024; Hong **LRM** arXiv 2023.11 → ICLR 2024; Tochilkin **TripoSR** arXiv 2024; Xu **InstantMesh** arXiv 2024; Boss **Stable Fast 3D** arXiv 2024.
- **3D Foundation Models**：Xiang **Trellis** arXiv 2024 (Microsoft); Tencent **Hunyuan3D-2** arXiv 2501.12202 (2025); Zhang **CLAY** SIGGRAPH 2024.

### A.3　Embodied AI / AR / VR 常见追问

3DGS 接物理引擎 → 先 2DGS / SuGaR 提 mesh → IsaacSim / MuJoCo；NeRF 动态化 → 4DGS / D-NeRF / K-Planes；AR 实时 3DGS → mobile-friendly (PostShot, Luma) + 剪枝 / 量化；3D 数据不足 → Objaverse-XL (Trellis) 或 2D 蒸馏 (DreamFusion 系) 或 multi-view 启发式 (MVDream)。

---

**3D Generation Quick Reference** · 主要参考：Mildenhall 2020 (NeRF), Müller 2022 (Instant-NGP), Kerbl 2023 (3DGS), Poole 2022/ICLR 2023 (DreamFusion), Wang 2023 (VSD), Xiang 2024 (Trellis), Tencent 2025 (Hunyuan3D-2). 涵盖：NeRF 体渲染推导、Instant-NGP hash 网格、3DGS 投影 Jacobian、SDS / VSD 梯度推导、single-image 3D、3D foundation models。Embodied AI / AR / VR 必备。
