## §0 TL;DR Cheat Sheet

> 💡 **9 句话搞定 Diffusion / Flow Distillation** — 把 50–1000 NFE 的 teacher 压到 1–4 NFE 的 student。一页拿下面试核心（详见后文 §1–§9 推导）。

1. **为什么**：diffusion 采样默认 50–1000 NFE，**网络前向占总延迟 >95%**；目标 ≤ 4 step 是 production 上线门槛（实时聊天 / 移动端 / 视频生成）。本文只讲 **few-step / one-step 蒸馏**，不涉 RL 后训练。

2. **Trade-off**：少 step 通常降质——naive uniform-skip DDIM 在 4 step 几乎不可用。蒸馏的本质是**用 teacher 的 50-step 轨迹/分布作为 supervision** 训练 student 一步直达。

3. **三大技术路线**：(a) **trajectory matching**（progressive distillation / CM / iCT / sCM / CTM / LCM / TCD）—— 让 student 复现 teacher ODE 解；(b) **distribution matching**（DMD / DMD2 / rCM）—— score gap 当 KL 梯度，匹配两个分布；(c) **adversarial**（ADD / LADD / SDXL-Lightning / FLUX-schnell）—— GAN loss + teacher 蒸馏。

4. **Consistency Models** (Song 2023 ICML)：学 $f_\theta(x_t, t) \to x_0$ 的 consistency function，**任一 $x_t$ 都映到同一 $x_0$**；boundary $f_\theta(x_{\sigma_\min}, \sigma_\min) = x_{\sigma_\min}$ 用 EDM-style precond 强制；CD（distillation，有 teacher）/ CT（training，无 teacher）。

5. **iCT** (Song-Dhariwal 2023)：**去 EMA target** + **pseudo-Huber loss**（替代 LPIPS）+ lognormal noise schedule + step-count curriculum，让 CT 接近 CD 质量。

6. **sCM / TrigFlow** (Lu-Song 2024 OpenAI)：连续时间 CM，**$x_t = \cos(t) x_0 + \sin(t) z$**（三角参数化让 EDM precond + PF-ODE + CM 同形式），1.5B ImageNet 512 2-step FID 1.88，**与最强 diffusion 差 <10%**。

7. **DMD** (Yin 2024 CVPR)：student 输出做"假分布"，**fake score** $s_\text{fake}$ 与 **real score** $s_\text{real}$ 之差当作 reverse-KL 梯度去推 student：$\nabla_\theta \text{KL}(p_\text{fake} \| p_\text{real}) = \mathbb{E}[(s_\text{fake} - s_\text{real}) \cdot \partial G_\theta / \partial \theta]$。**DMD2** (Yin 2024 NeurIPS) 去掉 regression loss、加 GAN、支持 multi-step student。

8. **ADD / LADD** (Sauer et al. 2023/2024 Stability)：teacher score distillation + **DINOv2 / VAE feature discriminator** 双重监督。**SDXL-Turbo** 1-step 1024、**SD3-Turbo** 4-step；**FLUX.1-schnell** 同样 LADD 系。

9. **LCM-LoRA** (Luo 2023)：把 Latent CM 训练成 **LoRA adapter**，~30 A100·h 就能让任意 SD 1.5 / SDXL fine-tune 用 4 step 出图，**不换 base model**。production 生态的关键启用器。

## §1 直觉 & 为什么需要蒸馏

### 1.1　采样成本是 diffusion 的阿喀琉斯之踵

| 模型 / sampler | 典型 NFE | 1024² 图像延迟（A100 fp16） |
|---|---|---|
| DDPM ancestral (1000 step) | 1000 | ~90 s |
| DDIM (50 step) | 50 | ~5 s |
| DPM-Solver++ (20 step) | 20 | ~2 s |
| **EDM Heun (35 step)** | ~35 | ~3.5 s |
| **LCM (4 step)** | 4 | ~0.4 s |
| **SDXL-Turbo (1 step)** | 1 | ~0.1 s |
| **DMD2 / FLUX-schnell (1-4 step)** | 1–4 | 0.1–0.4 s |

**production 要求**通常 < 0.5 s（实时聊天）或 < 1 s（手机端），原生 diffusion 远远超时。蒸馏不是"可选优化"——它是**让 diffusion 落地的必经之路**。

### 1.2　为什么 naive few-step 不行

把 50-step DDIM 改成 4-step uniform DDIM：sampler 的 $\Delta t$ 变大，**一阶 Euler 误差 $O(\Delta t)$ 急剧放大**，high-frequency 细节崩塌、噪声残留明显。即便用 EDM Heun 2nd-order，4-step 通常 FID > 15，远不可用。**根本原因**：teacher 的 ODE 轨迹是 curved（VP/VE path），4 步只能粗略折线近似。

> 💡 **蒸馏的核心 idea** — 不是改 sampler、不是降精度——而是**重新训一个 student**，让它学会"任意 $x_t$ 直接跳到 $x_0$"（CM 视角）或"输出分布匹配 teacher"（DMD 视角）或"输出图像骗过 discriminator"（ADD 视角）。三种视角对应三大流派。

### 1.3　蒸馏 vs 加速 sampler：本质区别

| | 加速 sampler（DDIM / DPM-Solver / Heun） | 蒸馏（CM / DMD / ADD） |
|---|---|---|
| 改训练？ | ❌ | ✅ 需新一轮训练 |
| 改网络？ | ❌（同一 $\epsilon_\theta$） | ✅ student 独立网络（或 teacher 的 fine-tune） |
| 极限 NFE | 10–20（解 ODE 精度极限） | 1–4 |
| 失败模式 | 离散化误差 | mode collapse / saturated colors / 缺多样性 |

**互补关系**：production pipeline 一般是 **"先选 sampler 类型 → 再蒸馏"**——比如 SD3 用 RF（Euler 友好）+ LADD 蒸到 4-step，FLUX 用 RF + LADD-schnell 蒸到 1-4 step。

### 1.4　全文 convention

| 符号 | 含义 |
|---|---|
| $x_0$ | 干净数据 |
| $x_t$ ($t \in [0, T]$ 或 $\sigma \in [\sigma_\min, \sigma_\max]$) | 加噪样本 |
| $z, \epsilon$ | $\mathcal{N}(0, I)$ 噪声 |
| $\theta$ / $\phi$ | student 参数 / teacher 参数 |
| $f_\theta(x_t, t)$ | CM 的 consistency function (CM 输出) |
| $G_\theta(z, t)$ | one-step / few-step student generator |
| $s_\theta(x, t) \approx \nabla \log p_t(x)$ | score |
| $D_\psi$ | discriminator (ADD/LADD 用) |
| NFE | Number of Function Evaluations |

> ⚠️ **时间方向陷阱（必须先 disambiguate）** — CM 系列论文用 EDM 的 $\sigma$-time（$\sigma_\min = 0.002$, $\sigma_\max = 80$），DDPM 用 $t \in [0, T]$，FM 用 $t \in [0, 1]$。本文按章节统一：§2 CM/iCT/sCM 用 $\sigma$-time；§3 DMD 用 $t \in [0, T]$；§4 ADD/LADD 用 $\sigma$-time（沿 EDM）；§5 Flow 系用 $t \in [0, 1]$（$t=0$ 噪声 / $t=1$ 数据）。

## §2 Consistency Models 家族

### 2.1　Consistency Models (CM, Song et al. 2023 ICML, arXiv:2303.01469)

**核心定义**：consistency function $f: (x_t, t) \mapsto x_{\sigma_\min}$ 沿 PF-ODE 轨迹**自洽**——

$$\boxed{\;f_\theta(x_t, t) = f_\theta(x_{t'}, t')\quad\text{对同一 ODE 轨迹上任意 } t, t' \in [\sigma_\min, \sigma_\max]\;}$$

由此可一步生成：$x_0 \approx f_\theta(z \cdot \sigma_\max, \sigma_\max)$，其中 $z \sim \mathcal{N}(0, I)$。

**Boundary condition**：要求 $f_\theta(x, \sigma_\min) = x$（最低噪声处恒等映射）——用 EDM-style precond 强制：

$$f_\theta(x, \sigma) = c_\text{skip}(\sigma)\, x + c_\text{out}(\sigma)\, F_\theta(x, \sigma)$$

其中 $c_\text{skip}(\sigma_\min) = 1$, $c_\text{out}(\sigma_\min) = 0$。Song 2023 的具体取值（与 EDM Karras 同形式）：

$$c_\text{skip}(\sigma) = \frac{\sigma_\text{data}^2}{(\sigma - \sigma_\min)^2 + \sigma_\text{data}^2},\quad c_\text{out}(\sigma) = \frac{\sigma_\text{data}\,(\sigma - \sigma_\min)}{\sqrt{\sigma_\text{data}^2 + \sigma^2}}$$

**Consistency Loss（核心）**：取相邻噪声级 $t_n < t_{n+1}$，要求 student 对 $(x_{t_n}, t_n)$ 和 $(x_{t_{n+1}}, t_{n+1})$ 输出一致——

$$\boxed{\;\mathcal{L}_\text{CD}(\theta) = \mathbb{E}\left[\lambda(t_n)\, d\!\Big(f_\theta(x_{t_{n+1}}, t_{n+1}),\; f_{\theta^-}(\hat x_{t_n}, t_n)\Big)\right]\;}$$

- $\theta^-$：EMA target（类似 BYOL，防止 representation collapse）
- $\hat x_{t_n} = x_{t_{n+1}} - (t_{n+1} - t_n) \cdot v_\phi(x_{t_{n+1}}, t_{n+1})$：teacher 一步 ODE 反向
- $d$：L2 或 LPIPS（CM 原文 ImageNet 64 用 LPIPS）
- $\lambda(t_n)$：权重，CM 原文取 1

> 💡 **CD vs CT 的关键区别** — CD (Consistency **Distillation**) 用 pretrained teacher $v_\phi$ 算 $\hat x_{t_n}$；CT (Consistency **Training**) 完全无 teacher，用 $\hat x_{t_n} = x_0 + t_n \epsilon$（同一 noise sample 加不同噪声水平）。CD 用 LPIPS + EMA 可达 FID 3.55（CIFAR-10），CT 只能到 8.7——直到 iCT 才追平。

### 2.2　从 PF-ODE 推 Consistency Loss（必考推导）

考虑 PF-ODE $\frac{dx}{dt} = v_\phi(x_t, t)$（teacher）。Consistency 定义要求沿轨迹自洽 $f_\theta(x_{t+\Delta t}, t+\Delta t) = f_\theta(x_t, t)$。一阶 Taylor 展开（在 trajectory 上）：

$$f_\theta(x_{t+\Delta t}, t+\Delta t) \approx f_\theta(x_t, t) + \Delta t \cdot \frac{d f_\theta}{dt}$$

其中 $\frac{d f_\theta}{dt} = \partial_t f_\theta + \partial_x f_\theta \cdot v_\phi$。所以 **continuous-time consistency loss**：

$$\mathcal{L}_\text{cont} = \mathbb{E}\left\|\frac{d f_\theta}{dt}\right\|^2 = \mathbb{E}\left\|\partial_t f_\theta + (\partial_x f_\theta)\, v_\phi(x_t, t)\right\|^2$$

**离散化**：用 $f_{\theta^-}$（EMA）当 stop-gradient 锚点，$\hat x_{t_n}$ 由 teacher 一步 ODE 得到：

$$\mathcal{L}_\text{CD} \approx \mathbb{E}\|f_\theta(x_{t_{n+1}}, t_{n+1}) - f_{\theta^-}(\hat x_{t_n}, t_n)\|^2$$

> ⚠️ **不能去掉 EMA 锚点** — 如果两侧都用 $\theta$，loss 退化为 $\|f_\theta - f_\theta\| = 0$，**网络无信号**。EMA 提供"过去的自己"做 supervision，类似 BYOL 防 collapse 的机制。iCT 论文（§2.3）证明在合适的 noise schedule + pseudo-Huber loss 下可以**去掉 EMA**——这是 iCT 的核心贡献之一。

### 2.3　iCT / Improved Techniques (Song-Dhariwal 2023, arXiv:2310.14189)

CT (Consistency Training) 原本质量远低于 CD。iCT 改进四件事：

| 改动 | 原 CT | iCT |
|---|---|---|
| Target | EMA $\theta^- = \tau \theta^- + (1-\tau) \theta$ | **直接 stop-grad**（不用 EMA） |
| Loss | LPIPS | **Pseudo-Huber** $d(a, b) = \sqrt{\lVert a-b \rVert^2 + c^2} - c$ |
| Noise sched | uniform discrete $\sigma_n$ | **Lognormal**：$\log \sigma \sim \mathcal{N}(P_\text{mean}, P_\text{std}^2)$ |
| Step count | fixed $N$ | **Curriculum**：$N(k) = \lceil N_\min \cdot (N_\max/N_\min)^{k/K} \rceil$ |

**Pseudo-Huber 的设计动机**：

- LPIPS 引入对 ImageNet pretrained 特征的**bias**——eval 时 FID 看起来好，但实际 distribution shift
- L2 对 outlier 敏感、训练不稳
- Pseudo-Huber $\sqrt{\|a-b\|^2 + c^2} - c$：小残差时 ≈ $\|a-b\|^2/(2c)$（L2），大残差时 ≈ $\|a-b\|$（L1）——**自适应 robust**

**结果**：iCT 在 CIFAR-10 **1-step FID 2.83 / 2-step FID 2.46**（论文摘要数字；更深的 iCT-deep 进一步到 2.51 / 2.24），且**不依赖 teacher**——彻底打开 from-scratch consistency training 的天花板。

### 2.4　sCM / TrigFlow (Lu-Song 2024 OpenAI, arXiv:2410.11081)

**问题**：离散时间 CM 有两大病——(i) 离散化误差（$N$ 越大越准但越慢）、(ii) 各种 hyper-parameter（noise schedule / EMA decay / loss curriculum）调起来很玄。

**TrigFlow 参数化**：把 forward path 写成三角形式——

$$\boxed{\;x_t = \cos(t)\, x_0 + \sin(t)\, z,\quad t \in [0, \pi/2],\; z \sim \mathcal{N}(0, I)\;}$$

边界：$t = 0$ 时 $x_t = x_0$（数据），$t = \pi/2$ 时 $x_t = z$（标准高斯）。

**为什么三角形式？** 这是同时让以下四件事**形式简洁**的唯一参数化（Lu-Song 2024 Theorem 1）：

- EDM precond：$D_\theta(x_t, t) = \cos(t)\, x_t - \sin(t)\, F_\theta$，自动满足 boundary
- PF-ODE：$\frac{dx_t}{dt} = -\sin(t) x_0 + \cos(t) z$，干净表达
- CM 输出：$f_\theta(x_t, t) = \cos(t) x_t - \sin(t) (\sigma_d F_\theta(x_t / \sigma_d, c_\text{noise}(t)))$（$\sigma_d$ 是 data std）
- Continuous-time consistency loss：直接梯度可写成 closed-form

**Continuous-Time Consistency Loss (sCM 核心)**：sCM 把连续时间 CM 梯度改写为 **stop-gradient MSE surrogate**（不是把 target 简化为 $r\cdot\mathrm{JVP}$ —— 那会在 warmup $r=0$ 时变成 self-reference 零信号）。正确形式：

$$\mathcal{L}_\text{sCM}(\theta, \phi) = \mathbb{E}_{x, t}\!\left[\frac{e^{w_\phi(t)}}{D}\Big\|F_\theta(x_t/\sigma_d, t) - \operatorname{sg}\!\big(F_{\theta^-}(x_t/\sigma_d, t) + g_{\theta^-}(x_t, t)\big)\Big\|_2^2 - w_\phi(t)\right]$$

其中 $F_{\theta^-}$ 是 EMA / stop-grad copy。**TrigFlow consistency function**：$f_\theta(x_t, t) = \cos t\, x_t - \sin t\, \sigma_d F_\theta(x_t/\sigma_d, t)$。令 $\hat v_t = dx_t/dt$（sCT 中 $= \cos t\, z - \sin t\, x_0$，sCD 中由 teacher PF-ODE 给出），**JVP-rearranged tangent target**：

$$g = -\cos^2(t)\,(\sigma_d F_{\theta^-} - \hat v_t) - r\cos(t)\sin(t)\!\left(x_t + \sigma_d \frac{dF_{\theta^-}}{dt}\right),\quad g \leftarrow \frac{g}{\|g\|_2 + c}.$$

warmup $r: 0 \to 1$ **只打开第二项**；当 $r=0$ 时仍有 $-\cos^2(t)(\sigma_d F_{\theta^-} - \hat v_t)$，因此退化为 velocity / diffusion matching，**不是零 loss**。

**关键技巧**：

- **Adaptive double normalization**：把 input/output 都按 $\sigma_d$ + $\sigma(t)$ 归一化，让网络的 effective scale 不依赖 $t$
- **Tangent warmup**（不是关全部 tangent）：$r$ 控制第二项 $-r\cos t\sin t(\cdots)$，第一项始终在；adaptive weighting $w_\phi(t)$ 与 tangent normalization 一起降方差
- **JVP via forward-mode autodiff**：PyTorch `torch.func.jvp`，**比 backward 算 Jacobian 快 ~2×**

**结果**：1.5B 参数，ImageNet 512×512 **2-step FID 1.88**，与最强 diffusion baseline 差 <10%——首次让 CM 在大规模 high-res 上拿到顶级数字。

### 2.5　CTM / Consistency Trajectory Models (Kim et al. 2024 ICLR, arXiv:2310.02279)

**问题**：CM 只能映 $(x_t, t) \to x_{\sigma_\min}$（轨迹终点），无法做中间点跳跃；step 数固定。

**CTM 的扩展**：学一个 $G(x_t, t, s)$——从 $(x_t, t)$ **跳到任意 $s < t$**：

$$G_\theta(x_t, t, s) \approx \text{ODE-solver}(x_t, t \to s)$$

- $s = \sigma_\min$ 时退化为 CM
- $s = t$ 时退化为 identity
- 中间 $s$ 让 user 自由选 NFE：3-step = $G(z, T, t_1) \to G(\cdot, t_1, t_2) \to G(\cdot, t_2, 0)$

**Loss**：trajectory matching——

$$\mathcal{L}_\text{CTM} = \mathbb{E}\Big[d\big(G_\theta(x_t, t, s),\; \text{ODE-solver}^\text{teacher}(x_t, t \to s)\big)\Big] + \lambda\, \mathcal{L}_\text{score}$$

- 第一项：trajectory consistency，让 student 复现 teacher ODE
- 第二项：辅助 score matching（避免 trivial solution）

**结果**：CIFAR-10 1-step FID 1.73, ImageNet 64 1.92——SOTA。**核心贡献**：把 step 数从"hard-coded"变成"runtime 可选"。

### 2.6　LCM / Latent Consistency Models (Luo et al. 2023, arXiv:2310.04378)

**LCM = CM on latent diffusion**（SD 1.5 / SDXL）。三大改进：

1. **Latent 空间**：在 VAE latent ($f=8$) 上做 CM，省 $64\times$ 计算
2. **CFG 蒸进 student**：训练时随机采样 guidance scale $w \in [w_\min, w_\max]$，把 $w$ 作为额外 condition——$f_\theta(x_t, t, c, w)$。**推理时无需双 forward** 跑 conditional + unconditional
3. **Skipping-Step Distillation**：取 $k$-step skip 的 teacher (如 $k=20$ 跳到 50/20 ≈ 2.5)，加速收敛

**结果**：4-step SD-XL 出图，FID 与 50-step SDXL 接近（同 base model）。

### 2.7　LCM-LoRA (Luo et al. 2023, arXiv:2311.05556)

**核心 idea**：LCM 训练的"差异权重" $\Delta \theta = \theta_\text{LCM} - \theta_\text{SD}$ 可以参数化为 LoRA——

$$\Delta W = B A,\quad B \in \mathbb{R}^{d \times r},\; A \in \mathbb{R}^{r \times k},\; r \in \{8, 16, 32, 64\}$$

只需训 $A, B$ 即可（~22M 参数 / SDXL），merge 时 $W' = W + \alpha B A$。

> ✅ **LCM-LoRA 的生态价值** — SD 1.5 / SDXL 生态有上万 fine-tune 模型（DreamShaper / RealisticVision / 各种角色 LoRA）。LCM-LoRA **不要求重训各家 base model**，用户只需挂上 LCM-LoRA + 自家原 LoRA 就能 4-step 出图。这点是 LCM 比 DMD/ADD 在 production 普及度高得多的原因。

### 2.8　TCD / Trajectory Consistency Distillation (Zheng et al. 2024, arXiv:2402.19159)

**TCD = LCM + trajectory-aware** 改进。两大贡献：

1. **Trajectory consistency function**：把 boundary condition 放宽到"沿轨迹任意点"，而非单一 $\sigma_\min$。具体用 **semi-linear consistency function**（exponential integrator 推导）减小参数化误差
2. **Strategic stochastic sampling**：multi-step inference 时**显式控制随机性**——通过 $\gamma \in [0, 1]$ 参数加可控扰动，避免 accumulated error 把分布拖偏

**实际效果**：低 NFE（4 step）质量高于 LCM；**高 NFE（8+ step）比 teacher 自己还细致**（因为 stochastic sampling 加了 expressivity）。

### 2.9　rCM / Score-Regularized Continuous-Time CM (2025, arXiv:2510.08431)

> 📍 **本文写作时（2025-2026）最新的 CM 工作之一** — rCM = "Score-Regularized Continuous-Time Consistency Model"，arXiv:2510.08431 verified。

**动机**：sCM 在 fine detail 上有质量瓶颈——作者归因于 **forward-divergence 的 mode-covering 性质**（KL(p_data ‖ p_student) 倾向覆盖所有 mode，导致细节模糊）。

**rCM 做法**：在 sCM loss 上加一项 **score distillation regularizer**（reverse-divergence flavor，类似 DMD 的 KL 梯度），让 student 兼具**mode-seeking**（清晰细节）+ mode-covering（多样性）。

**结果**：Cosmos-Predict2、Wan 2.1（14B）上 1-4 step 出 5 秒视频，质量持平 DMD2 + 多样性更好。

## §3 Distribution Matching Distillation (DMD 家族)

### 3.1　DMD 核心：reverse-KL via score gap (Yin et al. 2024 CVPR, arXiv:2311.18828)

**问题视角**：student $G_\theta$ 把 noise 直接映成图，要让它**输出分布 $p_\text{fake}$ 匹配 teacher 分布 $p_\text{real}$**。直接优化 $\text{KL}(p_\text{fake} \| p_\text{real})$ 的梯度——

$$\nabla_\theta \text{KL}(p_\text{fake}^\theta \| p_\text{real}) = -\mathbb{E}_{x \sim p_\text{fake}^\theta}\!\left[\big(\nabla_x \log p_\text{real}(x) - \nabla_x \log p_\text{fake}(x)\big) \cdot \frac{\partial G_\theta}{\partial \theta}\right]$$

**关键观察**：$\nabla_x \log p_\text{real}$ 就是 teacher score $s_\text{real}$（teacher diffusion 模型现成），$\nabla_x \log p_\text{fake}$ 用一个**fake score model** $s_\text{fake}$（在 student 当前输出上训出来的小 diffusion）。

**DMD Loss**（两个 loss 联训，**记号严格**：$\mu$ 是 denoiser/mean predictor，$s_\mu(x_t,t) = (\alpha_t \mu(x_t,t) - x_t)/\sigma_t^2$ 是从 denoiser 转出的 score；DMD 论文用 denoiser，而非裸 score）：

$$
\boxed{\;
\begin{aligned}
\nabla_\theta \mathcal{L}_\text{DMD}^G &= \mathbb{E}_{z, t, \epsilon}\!\left[w_t\,\alpha_t\,(s_\text{fake}(x_t, t) - s_\text{real}(x_t, t))^\top\,\tfrac{\partial G_\theta(z)}{\partial\theta}\right] \quad\text{// student, surrogate} \\
\mathcal{L}_\text{fake}(\phi_f) &= \mathbb{E}\!\left[\lambda_t\,\|\mu_{\phi_f}(x_t, t) - \operatorname{sg}(G_\theta(z))\|_2^2\right] \quad\text{// fake denoiser DSM target 是 student 当前输出}
\end{aligned}
\;}
$$

其中 $x_t = \alpha_t G_\theta(z) + \sigma_t \epsilon$。**辅助 regression loss**：DMD v1 还加一项 $\mathbb{E}\|G_\theta(z) - \text{ODE-solver}^\text{teacher}(z)\|^2$（teacher pair 监督）防止 student 跑偏——但这要预生成大批 teacher pair，**贵且 mode 受限**；DMD2 去掉了这项。

### 3.2　从 reverse-KL 推 DMD 梯度（必考推导）

设 student $G_\theta(z) \mapsto x$，$z \sim \mathcal{N}(0, I)$。fake 分布 $p_\text{fake}^\theta(x) = G_\theta \# \mathcal{N}(0, I)$（push-forward）。

reverse KL：

$$\text{KL}(p_\text{fake} \| p_\text{real}) = \mathbb{E}_{x \sim p_\text{fake}}[\log p_\text{fake}(x) - \log p_\text{real}(x)]$$

求 $\nabla_\theta$：

$$\nabla_\theta \text{KL} = \mathbb{E}_z\!\left[\nabla_\theta \log p_\text{fake}^\theta(G_\theta(z)) - \nabla_\theta \log p_\text{real}(G_\theta(z))\right]$$

第二项 chain rule：$\nabla_\theta \log p_\text{real}(G_\theta(z)) = \nabla_x \log p_\text{real}(x)\big|_{x=G_\theta(z)} \cdot \partial G_\theta / \partial \theta$。

第一项展开后 + 利用 $\mathbb{E}_{p_\text{fake}}[\nabla_\theta \log p_\text{fake}] = 0$（score function trick），整理：

$$\nabla_\theta \text{KL} = -\mathbb{E}_z\!\left[(\nabla_x \log p_\text{real} - \nabla_x \log p_\text{fake})\big|_{x=G_\theta(z)} \cdot \partial_\theta G_\theta(z)\right]$$

**但 $p_\text{real}, p_\text{fake}$ 在 high-dim 上 score 不连续 / 不光滑**——DMD 的 trick 是**在所有 noise level $t$ 上对 $x_t = G_\theta(z) + \sigma_t \epsilon$ 算 score**，把估计移到 smooth 的 $p_t$ 上——这就是为什么 DMD 既要 real diffusion teacher 又要 fake diffusion（两者都是"在不同 noise level 给 score"）。

> 💡 **DMD vs CM 的本质区别** — CM 是 **trajectory matching**（让 student 复现 teacher ODE 解），DMD 是 **distribution matching**（让两个分布的 score 处处相等）。**CM 需要 step alignment（noise schedule 对齐），DMD 不需要**——DMD 的 student 可以是任意 generator 架构，只要可微。

### 3.3　DMD2 (Yin et al. 2024 NeurIPS, arXiv:2405.14867)

**改进四件事**：

| 改动 | DMD | DMD2 |
|---|---|---|
| Regression loss | 需要 teacher pair（贵） | **去掉** |
| GAN | ❌ | **加 GAN classifier**：接在 fake diffusion denoiser bottleneck 上，**在 noised real / noised fake 上判别**（不是 clean image） |
| TTUR | 1:1 | fake denoiser **每个 generator step 更新约 5 次**（论文 ImageNet 默认 5:1） |
| Student | 1-step only | **multi-step backward simulation**：训练时按 inference schedule 跑当前 student 拿到中间 noisy states，再在那些 states 上算 DMD/GAN loss，对齐训练/推理分布 |

**DMD2 总 loss**（generator 侧）：

$$\mathcal{L}_\text{DMD2}^G = \underbrace{\mathcal{L}_\text{DMD}^G(\theta)}_{\text{score gap surrogate}} + \lambda_\text{GAN} \cdot \mathcal{L}_\text{adv}^G(\theta)$$

**判别器侧**：DMD2 的 D 通常是 fake denoiser bottleneck 上的 classifier head，输入是 noised image $x_t = \alpha_t x + \sigma_t\epsilon$（不是 clean $x$）：

$$\mathcal{L}_D = \mathbb{E}_{x \sim p_\text{data}, t}\!\left[\text{softplus}(-D_\psi(x_t, t))\right] + \mathbb{E}_{z, t}\!\left[\text{softplus}(D_\psi(\hat x_t^{\text{fake}}, t))\right]$$

**Multi-step backward simulation**（关键，比简单 unroll 强）：训练时按 $K$-step inference schedule 跑当前 student 得到中间 noisy intermediate states $x_{t_k}$，再在这些 states 上调用 student / 算 DMD-GAN loss。这保证训练输入分布 = 推理时第 $k$ 步看到的输入分布，**不是简单的 noised real image**。

**结果**：ImageNet 64 1-step FID **1.28**（DMD v1 是 2.62），**首次让 one-step diffusion 超过 GAN**。production：DMD2-SDXL 1-step 出 1024×1024 megapixel image。

### 3.4　Score gap 的统计物理直觉

reverse-KL 的"score gap" $s_\text{real} - s_\text{fake}$ 在物理上对应**两个 Gibbs 分布的"force diff"**——

$$s_\text{real} - s_\text{fake} = \nabla_x \log\frac{p_\text{real}}{p_\text{fake}} = -\nabla_x [V_\text{real}(x) - V_\text{fake}(x)]$$

把 student 当 particle，$s_\text{real} - s_\text{fake}$ 是把它从 $p_\text{fake}$ 推向 $p_\text{real}$ 的"力"。**这是 DMD 与 GAN 的本质区别**——GAN 用 discriminator 给 binary 信号，DMD 用 score gap 给**dense vector field 信号**，sample efficiency 高得多。

## §4 Adversarial Distillation (ADD / LADD 家族)

### 4.1　ADD / SDXL-Turbo (Sauer et al. 2023, arXiv:2311.17042)

**Stability AI 2023.11，让 SDXL 1-step 出 512² 图**。两大监督：

$$\boxed{\;\mathcal{L}_\text{ADD} = \mathcal{L}_\text{adv}^G(\theta, \psi) + \lambda \cdot \mathcal{L}_\text{distill}(\theta, \phi)\;}$$

- **$\mathcal{L}_\text{adv}$**：hinge loss + **DINOv2 vision backbone 当 discriminator**（不是 from-scratch 训 D，而是 fix DINOv2 + 多个 head）
- **$\mathcal{L}_\text{distill}$**：student 1-step 输出 vs teacher multi-step 输出的 MSE（"score distillation" 的离散形式）

**DINOv2 discriminator** 是 ADD 的关键：

- 普通 GAN 训 D from-scratch，**对 1-step generator 不稳**（mode collapse 严重）
- DINOv2 提供"pretrained 高级 perceptual features"，把判别问题 anchor 到**强语义空间**
- 多个 head（不同 layer feature）+ hinge loss → 训练稳定

> ⚠️ **Distillation loss 的实际形式** — ADD 论文里的 $\mathcal{L}_\text{distill}$ 是 **score-distillation 风格**（用 teacher denoiser 在 noisy student output 上估计目标），而**不是**简单的 pixel-space MSE 也**不是** KL。这里前文教学版的 $\|G_\theta - \text{ODE}\|^2$ 是 illustrative simplification；细节见原论文 Eq.(6-7)。ADD 靠 GAN loss 补 mode 多样性。

**结果**：SDXL-Turbo 在 **512×512** 上 1-step ≈ 100ms / image (A100)，CLIP score 与 4-step SDXL 相当；高分辨率 **1024×1024** 在 SDXL-Turbo 上质量有限，主要由后续的 **LADD / SD3-Turbo / Lightning** 解决。

### 4.2　LADD / SD3-Turbo (Sauer et al. 2024, arXiv:2403.12015)

**问题**：ADD 在 pixel space 算 distill loss + DINOv2 D，对**高分辨率（1024+）和 latent diffusion 不友好**——pixel 解码贵、DINOv2 输入分辨率限制 224 / 518。

**LADD = Latent ADD**：把 discriminator 直接搬到 latent space——

| | ADD | LADD |
|---|---|---|
| Discriminator backbone | DINOv2 (pixel) | **teacher diffusion 自己的中间 layer feature**（latent） |
| Distillation | pixel MSE | latent space distill |
| 分辨率 scale | 受 DINOv2 限制 | latent 任意尺寸 |
| 应用模型 | SDXL | **SD3 (8B)、FLUX (12B)** |

**Discriminator 设计**：把 teacher 的 MM-DiT block 抽出来 fine-tune 成 D 的 backbone——理由是 **diffusion 训练中的 intermediate feature 已经隐式学到了"什么样是真实的 latent"**。

**结果**：

- **SD3-Turbo** = SD3 8B + LADD → 4-step 1024² 媲美 multi-step SD3
- **FLUX.1-schnell** = FLUX 12B + LADD → 1-4 step 1024² 出图（Apache 2.0 开源）

### 4.3　SDXL-Lightning (Lin et al. 2024, arXiv:2402.13929)

ByteDance 的开源 SDXL 蒸馏方案，**progressive + adversarial 双管**：

- **Progressive (halving)**：从 teacher 多 step 开始，每阶段把 step 数 **减半**（$T \to T/2 \to T/4 \to \dots$），每段都用 MSE 拟合上一阶段的 teacher（这是 Salimans-Ho 2022 progressive distillation 的 lineage）。最终能 1/2/4/8-step 多档可选
- **Adversarial**：每阶段末用 GAN loss 提升 fidelity
- **Discriminator**：自训（不像 ADD/LADD 借现成 backbone）

**结果**：SDXL 1024² 1-step/2-step/4-step 多档可选，**开源 LoRA 形式**（与 LCM-LoRA 类似），生态友好。

### 4.4　ADD/LADD/Lightning 对比

| 方法 | Discriminator | Distill loss | 应用 | 1-step quality |
|---|---|---|---|---|
| **ADD** | DINOv2 (pixel) | pixel MSE | SDXL | 中（512²) |
| **LADD** | teacher MM-DiT feat (latent) | latent score-distill | SD3, FLUX | 高（1024²） |
| **Lightning** | self-trained CNN | progressive MSE | SDXL | 中-高 |
| **DMD2** | self-trained + score gap | reverse-KL via score | SDXL | 高（含多样性） |

> 💡 **production 选型 cheat sheet** — 如果 base 是 SD 1.5 / SDXL，用 **LCM-LoRA**（生态最广）或 **SDXL-Lightning**（开源稳定）；如果 base 是 SD3 / FLUX，**LADD** 是官方路线；想要 **GAN-free + score-based** 选 **DMD2**；学术想刷 SOTA 选 **sCM / rCM**。

## §5 Flow / Rectified Flow 蒸馏

### 5.1　Rectified Flow + Reflow 路线 (Liu et al. 2022, arXiv:2209.03003)

**Rectified Flow path**：$x_t = (1-t) x_0 + t\, x_1$，$x_0 \sim \mathcal{N}(0, I)$（噪声端），$x_1 \sim p_\text{data}$，target $u_t = x_1 - x_0$（**常数 vector**）。

**Reflow 算法**（reverse 用 ODE，配对再训）：

1. 训 $v_\theta^{(1)}$ 用独立 pair $(x_0, x_1) \sim p_0 \otimes p_\text{data}$
2. 用 $v_\theta^{(1)}$ 跑 ODE 生成 coupled pair $(x_0, x_1^{(1)})$，即 $x_1^{(1)} = x_0 + \int_0^1 v_\theta^{(1)}(x_t, t)\, dt$
3. 用 coupled pair 重训 $v_\theta^{(2)}$——**新轨迹更直**（transport cost 非增定理）

**为什么 reflow 让 trajectory 变直？**

考虑 transport cost $\mathbb{E}[\|x_1 - x_0\|^2]$ 当 coupling。独立 pair 的 cost 大；reflow 后 $(x_0, x_1^{(1)})$ 已经被 ODE 自然配对，是当前 $v_\theta^{(1)}$ 下的"最优传输"。Liu 2022 证：再训一次后总 transport cost 不增（实际上往往严格减），且**曲线"直"等价于 vector field 不依赖 $t$**——$v(t, x) = $ const 沿轨迹 → 1-step 生成。

**InstaFlow (2023, arXiv:2309.06380)**：第一个把 reflow 用到 SD 上，1-step 出图 FID 23.3（512²）。

### 5.2　Reflow 的"直线"极限

**理想极限**：若 reflow 收敛到完全 straight，则 $v_\theta(t, x) = v_\theta(x)$（与 $t$ 无关），1-step Euler 可达——

$$x_1 = x_0 + 1 \cdot v_\theta(x_0)$$

**实际**：1-2 次 reflow 后已足够"直"以支撑 4-step Euler 媲美 50-step；完全 1-step 需要更多 reflow + adversarial 微调（如 SD3-Turbo / FLUX-schnell）。

### 5.3　SD3-Turbo / FLUX-schnell = RF + LADD

production 实际栈：

```
SD3 / FLUX (Rectified Flow, ~50-step 1024²)
  │ pretrain
  ↓
LADD distillation (teacher 蒸馏 + latent discriminator)
  ↓
SD3-Turbo / FLUX-schnell (1-4 step 1024²)
```

**不是单纯 reflow**——LADD 提供 adversarial fidelity，比 pure reflow 在 high-res production 上更稳。

### 5.4　Flow-OPD（arXiv:2605.08063, 2026）— **out-of-scope sidebar**

> 📍 **澄清范围** — Flow-OPD 的主要 contribution 是 **multi-reward RL alignment + on-policy specialist distillation**，**不是** few-step inference distillation。把它放在这里是因为它名字带 "Distillation" 且涉及 flow models；但本 cheat sheet §2-§4 的核心主线（CM/DMD/ADD 把 50 step → 1-4 step）与 Flow-OPD 的 RL alignment 目标不同。
>
> 与本文相邻的姊妹篇 **`diffusion_post_training_tutorial.md`** 有完整 RL alignment 讨论（Flow-GRPO / Diffusion-DPO / DDPO 等），Flow-OPD 在那个语境下更准确。

**简要 idea（仅 sidebar，深入讨论见 post-training 教程 + 原 paper）**：用多个 reward-specific teacher（每个 reward GRPO fine-tuned）做 on-policy distillation supervision，student 在 inference 时仍是 few-step + 同时获得 multi-reward alignment。

**Loss 形式**（简化 sketch，请以原 paper 为准）：

$$\mathcal{L}_\text{OPD-sketch} = \mathbb{E}_{x \sim \pi_\theta}\!\left[\sum_k w_k(x) \cdot \|v_\theta(x_t, t) - v_{\phi_k}(x_t, t)\|^2\right]$$

其中 $w_k$ 是 task-aware weighting；这只是结构示意，**不主张** Flow-OPD 与 DMD 数学上"退化等价"（这层关系无可靠依据，请不要在面试中那样断言）。

**论文报告结果**：基于 SD 3.5 Medium，GenEval 63 → 92，OCR 59 → 94（细节参考原 paper Table）。

### 5.5　Rectified Diffusion / 后续工作

> 📍 **Rectified Diffusion (arXiv:2410.07303) 在 main scope 之外的相关工作 brief mention**

后续工作（如 Rectified Diffusion）挑战"straightness 是不是必须"——发现**直线不是必要条件**，只要 ODE 解空间 sufficiently expressive 即可。这条线和 sCM 的"continuous-time CM"在 mathematical formulation 上有趋同迹象。

## §6 CFG 蒸馏

### 6.1　为什么 CFG 要单独蒸馏

CFG 推理：

$$\tilde\epsilon(x, c) = (1 + w) \epsilon_\theta(x, c) - w\, \epsilon_\theta(x, \emptyset)$$

**每步两次 forward**（conditional + unconditional），延迟翻倍。所以 production CFG-aware 模型要 **把 CFG 蒸进 single forward**。

### 6.2　Guidance Distillation (Meng et al. 2023 CVPR, arXiv:2210.03142)

**Stage 1 — Guidance distillation**：训一个 student $\tilde\epsilon_\theta(x, c, w)$，**输入加上 guidance scale $w$ 作为额外 condition**，让它直接输出 CFG 后的 score：

$$\mathcal{L}_\text{guide} = \mathbb{E}\!\left[\|\tilde\epsilon_\theta(x_t, c, w) - \tilde\epsilon^*(x_t, c, w)\|^2\right]$$

其中 $\tilde\epsilon^*$ 是 teacher 显式跑两次 forward 得到的 CFG 输出。Student 只跑一次 forward。

**Stage 2 — Step distillation**：在 stage 1 基础上叠 progressive distillation，把 step 数从 32 蒸到 4 → 2 → 1。

**LCM 的 CFG-aware 设计**继承自此——把 $w$ 作 condition 喂进网络，是 LCM-LoRA 的关键。

### 6.3　Step-distillation vs Trajectory-distillation 区别

| | Step-distillation (Salimans-Ho 2022) | Trajectory-distillation (CM/CTM) |
|---|---|---|
| 目标 | 把 $N$-step student 蒸到 $N/2$-step | 学 trajectory function $f(x_t, t) \to x_0$ |
| 训练阶段 | 多 stage progressive | 单 stage |
| Step 数 | 每次减半（32→16→8→4→2→1） | 任意（1-step 直接训） |
| Boundary condition | 无需特殊 | 必须 $f(x_{\sigma_\min}, \sigma_\min) = x$ |
| Teacher | 上一阶段 student（自我蒸馏） | 原始 diffusion |

> 💡 **历史轨迹** — 2022 progressive distillation 是首个让 diffusion 4-step 可用的方法；2023 CM 通过 trajectory function 直接 1-step；2024 sCM / DMD2 把 1-step 推到 SOTA。**思路演化**：迭代逼近（progressive）→ 函数拟合（CM）→ 分布匹配（DMD）→ 三角参数化（sCM）。

## §7 From-Scratch PyTorch 代码

### 7.1　Code 1: Consistency Distillation Loss (CD, base CM)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def edm_precond(F_net, x, sigma, sigma_data=0.5, sigma_min=0.002):
    """EDM-style precond，让 boundary f(x, sigma_min) = x 自动满足"""
    c_skip = sigma_data**2 / ((sigma - sigma_min)**2 + sigma_data**2)
    c_out  = sigma_data * (sigma - sigma_min) / torch.sqrt(sigma_data**2 + sigma**2)
    c_in   = 1.0 / torch.sqrt(sigma_data**2 + sigma**2)
    c_noise = 0.25 * torch.log(sigma)
    # 广播到 [B, 1, 1, 1]（图像）
    c_skip = c_skip.view(-1, 1, 1, 1)
    c_out  = c_out.view(-1, 1, 1, 1)
    c_in   = c_in.view(-1, 1, 1, 1)
    F_x = F_net(c_in * x, c_noise)
    return c_skip * x + c_out * F_x

@torch.no_grad()
def teacher_ode_step(x_t1, t1, t0, teacher):
    """teacher 一步 Heun (EDM 2nd-order) 反向: t1 -> t0"""
    d1 = (x_t1 - teacher(x_t1, t1)) / t1  # 当前梯度
    x_euler = x_t1 + (t0 - t1) * d1       # Euler 预测
    d2 = (x_euler - teacher(x_euler, t0)) / t0
    return x_t1 + 0.5 * (t0 - t1) * (d1 + d2)

def consistency_distillation_loss(student, student_ema, teacher,
                                   x_0, sigmas, N=18):
    """
    student / student_ema: 同 architecture，ema 是 stop-grad 版本
    teacher: pretrained diffusion (EDM denoiser)
    x_0: clean image batch [B, C, H, W]
    sigmas: noise schedule，**索引递增 = noise 递增** (sigmas[0]=sigma_min, sigmas[N]=sigma_max)
    
    !!! 教学版示意：约定 sigmas 递增方便 t_{n+1} > t_n。
        生产实现请参考 EDM 官方代码（karras/edm: 通常 sigmas 递减）+ 论文 Eq. 形式。
    """
    B = x_0.shape[0]
    # 1) 随机选相邻噪声级 n ~ U{0, N-1}
    n = torch.randint(0, N, (B,), device=x_0.device)
    t_n1 = sigmas[n + 1]    # higher noise (per 上文 convention: sigmas 递增)
    t_n  = sigmas[n]        # lower noise
    
    # 2) 采样 x_{t_{n+1}} = x_0 + t_{n+1} * eps
    eps = torch.randn_like(x_0)
    x_tn1 = x_0 + t_n1.view(-1, 1, 1, 1) * eps
    
    # 3) teacher 一步 ODE 反向得到 x_{t_n}
    with torch.no_grad():
        x_tn = teacher_ode_step(x_tn1, t_n1, t_n, teacher)
    
    # 4) student / student_ema 都过 EDM precond
    f_online = edm_precond(student, x_tn1, t_n1)
    with torch.no_grad():
        f_target = edm_precond(student_ema, x_tn, t_n)
    
    # 5) consistency loss (LPIPS / L2 二选一; 这里用 L2)
    loss = F.mse_loss(f_online, f_target)
    return loss

def update_ema(ema_model, model, decay=0.9999):
    """EMA target，类似 BYOL"""
    with torch.no_grad():
        for p_ema, p in zip(ema_model.parameters(), model.parameters()):
            p_ema.data.mul_(decay).add_(p.data, alpha=1 - decay)
```

### 7.2　Code 2: iCT (去 EMA + Pseudo-Huber + Lognormal + Curriculum)

```python
def pseudo_huber(a, b, c=0.00054):
    """Pseudo-Huber loss: sqrt(||a-b||^2 + c^2) - c
    小残差 ≈ L2/2c, 大残差 ≈ L1. iCT 论文 c=0.00054 (CIFAR-10)"""
    return torch.sqrt((a - b).pow(2).sum(dim=(1, 2, 3)) + c**2).mean() - c

def lognormal_sigma(B, P_mean=-1.1, P_std=2.0, sigma_min=0.002, sigma_max=80.0):
    """iCT 用 lognormal 而不是 uniform 采样 sigma
    log_sigma ~ N(P_mean, P_std)"""
    log_sigma = torch.randn(B) * P_std + P_mean
    sigma = torch.exp(log_sigma).clamp(sigma_min, sigma_max)
    return sigma

def get_curriculum_N(step, total_steps, N_min=10, N_max=1280, schedule='exp'):
    """Step-count curriculum: N 从 10 渐增到 1280
    K 步训练里, N(k) = ceil(N_min * (N_max/N_min)^(k/K))"""
    k = step / total_steps
    if schedule == 'exp':
        N = N_min * (N_max / N_min) ** k
    else:
        N = N_min + (N_max - N_min) * k
    return int(math.ceil(N))

def ict_loss(student, x_0, step, total_steps):
    """iCT: no EMA, no LPIPS, no teacher
    consistency loss on (x_0 + sigma_n*eps, x_0 + sigma_{n+1}*eps) with SAME eps"""
    B = x_0.shape[0]
    device = x_0.device
    
    # 1) curriculum N
    N = get_curriculum_N(step, total_steps)
    
    # 2) 选相邻 sigma_n, sigma_{n+1}（从 lognormal 离散化的 N+1 个点里选）
    # !!! 约定 sigmas 升序，所以 sigmas[n+1] > sigmas[n]（与 §2.3 CD 代码一致）
    sigmas = lognormal_sigma(N + 1).to(device).sort(descending=False).values
    n_idx = torch.randint(0, N, (B,), device=device)
    t_n1 = sigmas[n_idx + 1]  # higher noise
    t_n  = sigmas[n_idx]      # lower noise
    
    # 3) 关键：同一个 epsilon 加两个不同噪声水平（不需要 teacher）
    eps = torch.randn_like(x_0)
    x_tn1 = x_0 + t_n1.view(-1, 1, 1, 1) * eps
    x_tn  = x_0 + t_n.view(-1, 1, 1, 1) * eps
    
    # 4) student 都跑 (无 EMA, 但 target 用 stop_grad)
    f_online = edm_precond(student, x_tn1, t_n1)
    with torch.no_grad():
        f_target = edm_precond(student, x_tn, t_n)
    
    # 5) Pseudo-Huber
    loss = pseudo_huber(f_online, f_target, c=0.00054)
    return loss
```

### 7.3　Code 3: sCM Continuous-Time Loss (TrigFlow)

```python
import torch.func as tfunc

def trigflow_xt(x_0, z, t):
    """TrigFlow path: x_t = cos(t) x_0 + sin(t) z, t in [0, π/2]"""
    cos_t = torch.cos(t).view(-1, 1, 1, 1)
    sin_t = torch.sin(t).view(-1, 1, 1, 1)
    return cos_t * x_0 + sin_t * z

def scm_loss(F_net, x_0, sigma_data=0.5, r_warmup=0.5):
    """
    Simplified continuous-time CM loss (sCM, Lu-Song 2024).
    F_net: student network F_θ(x_t / σ_d, t)
    r_warmup: NCS warmup ratio (0=pure score, 1=pure CM)
    """
    import math
    B = x_0.shape[0]
    device = x_0.device

    # 1) lognormal t (TrigFlow 时间 t in [0, π/2])
    log_t = torch.randn(B, device=device) * 1.0 - 0.4  # σ ≈ 1, mean shift
    t = torch.sigmoid(log_t) * (math.pi / 2 - 0.001) + 0.001  # 避开 boundary

    # 2) 采样 x_t
    z = torch.randn_like(x_0)
    x_t = trigflow_xt(x_0, z, t)

    # 3) PF-ODE tangent direction (TrigFlow: dx_t/dt = -sin(t) x_0 + cos(t) z)
    cos_t = torch.cos(t).view(-1, 1, 1, 1)
    sin_t = torch.sin(t).view(-1, 1, 1, 1)
    dxdt = -sin_t * x_0 + cos_t * z

    # 4) F_θ 在 (x_t/σ_d, t) 的输出 + JVP（forward-mode autodiff，比 backward 算 Jac 快）
    def net_fn(xt_norm, t_):
        return F_net(xt_norm, t_)

    x_t_norm = x_t / sigma_data

    # 4b) student forward（带 grad）以拿 F_out
    F_out = net_fn(x_t_norm, t)

    # 5) sCM target: stop_grad(F_minus + normalized tangent g)
    # 第一项 velocity matching 在 r=0 时仍有信号；第二项由 warmup r 渐渐打开。
    # JVP tangent direction = dx/dt（PF-ODE 方向），JVP 输出即 dF/dt。
    tangent_x = dxdt / sigma_data        # tangent of (x_t/σ_d) 沿 dx/dt 方向
    tangent_t = torch.ones_like(t)       # dt/dt = 1

    with torch.no_grad():
        F_minus, dFdt = tfunc.jvp(
            net_fn,
            (x_t_norm, t),
            (tangent_x, tangent_t),
        )
        # First term: velocity / diffusion matching（r=0 时也有信号）
        g = -(cos_t ** 2) * (sigma_data * F_minus - dxdt)
        # Second term: consistency tangent，warmup 渐渐打开（cos·sin 因子仅出现一次）
        g = g - r_warmup * (cos_t * sin_t * x_t + sigma_data * cos_t * sin_t * dFdt)
        # Normalize tangent for stability
        g = g / (g.flatten(1).norm(dim=1).view(-1, 1, 1, 1) + 0.1)
        target = F_minus + g

    # 6) surrogate MSE loss (adaptive w_phi(t) omitted in tutorial code)
    loss = F.mse_loss(F_out, target)
    return loss
```

### 7.4　Code 4: DMD Loss (Distribution Matching via Score Gap)

```python
class DMDTrainer:
    """
    DMD v1 (Yin 2024 CVPR): 三个网络
    - G_θ: 1-step student generator (z -> x)
    - s_real: pretrained teacher diffusion (frozen)
    - s_fake: fake diffusion, trained on G_θ outputs
    """
    def __init__(self, G, s_fake, s_real_frozen, sigma_data=0.5):
        self.G = G
        self.s_fake = s_fake  # trainable
        self.s_real = s_real_frozen  # frozen
        self.opt_G = torch.optim.AdamW(G.parameters(), lr=1e-5)
        self.opt_f = torch.optim.AdamW(s_fake.parameters(), lr=1e-5)
        self.sigma_data = sigma_data

    def student_loss(self, z):
        """DMD student loss: alpha_t * (s_fake - s_real)^T · ∂G/∂θ
        其中 s_*(x_t,t) = (alpha_t * mu_*(x_t,t) - x_t) / sigma_t^2 由 denoiser 转出。
        EDM/VE 约定 alpha_t = 1；VP/DDPM 需用 scheduler alpha_t。
        """
        x = self.G(z)               # G_θ(z), 1-step output
        B = x.shape[0]
        # 随机 noise level
        sigma = torch.exp(torch.randn(B, device=x.device) * 1.6 - 1.0).view(-1, 1, 1, 1)
        eps = torch.randn_like(x)
        alpha = 1.0                     # EDM/VE; for VP use scheduler.alpha(t)
        x_t = alpha * x + sigma * eps   # 与论文 x_t = alpha*x + sigma*eps 一致

        with torch.no_grad():
            mu_real = self.s_real(x_t, sigma.squeeze())  # frozen denoiser / mean predictor
            mu_fake = self.s_fake(x_t, sigma.squeeze())  # trainable fake denoiser

            # DMD 的 score-gap：s_fake - s_real = α(μ_fake - μ_real)/σ²。
            # 配 DMD 权重 w_t ∝ σ²/α 抵消 1/σ²，得 w_t(s_fake - s_real) = α(μ_fake - μ_real)。
            # 再用 mean-abs normalization 稳数值（DMD 论文 Eq.(8)）。
            grad_proxy = alpha * (mu_fake - mu_real)
            grad_proxy = grad_proxy / (
                (x.detach() - mu_real).abs().mean(dim=(1, 2, 3), keepdim=True) + 1e-6
            )

        # surrogate：loss = +(x · grad_proxy.detach()).sum()，
        # backward 得 ∇L = grad_proxy · ∂G/∂θ = ∇_θ KL(p_fake‖p_real)，
        # optimizer step θ -= η∇L 即 minimize KL。
        loss_G = (x * grad_proxy.detach()).sum(dim=(1, 2, 3)).mean()
        return loss_G

    def fake_score_loss(self, z):
        """fake denoiser 用 DSM；target 是 student 当前输出（不是 -eps/sigma 这种 score-head 形式）"""
        with torch.no_grad():
            x = self.G(z)               # detached student output
        B = x.shape[0]
        sigma = torch.exp(torch.randn(B, device=x.device) * 1.6 - 1.0).view(-1, 1, 1, 1)
        eps = torch.randn_like(x)
        alpha = 1.0
        x_t = alpha * x + sigma * eps

        pred_x0 = self.s_fake(x_t, sigma.squeeze())   # denoiser output
        target_x0 = x.detach()                          # student output as DSM target
        loss_f = ((pred_x0 - target_x0) ** 2).flatten(1).mean(1).mean()
        return loss_f

    def step(self, z_batch):
        # 1) update fake score (在 G 当前输出上)
        self.opt_f.zero_grad()
        loss_f = self.fake_score_loss(z_batch)
        loss_f.backward()
        self.opt_f.step()
        
        # 2) update G via score gap
        self.opt_G.zero_grad()
        loss_G = self.student_loss(z_batch)
        loss_G.backward()
        self.opt_G.step()
        return loss_G.item(), loss_f.item()
```

### 7.5　Code 5: DMD2 Loss (去 regression + GAN + multi-step)

```python
class DMD2Trainer(DMDTrainer):
    """DMD2: 去 regression + 加 noised-input GAN + multi-step backward simulation。
    D 在原论文里是 fake denoiser bottleneck 上的 classifier head（共享 backbone），
    输入是 noised image x_t = alpha*x + sigma*eps。此处用独立 D 当教学近似。
    TTUR：fake denoiser + D 每 generator step 更新约 5 次（论文 ImageNet 默认 5:1）。
    """
    def __init__(self, G, s_fake, s_real_frozen, D, sigma_data=0.5,
                 lambda_gan=1.0, num_steps_train=4, ttur_ratio=5):
        super().__init__(G, s_fake, s_real_frozen, sigma_data)
        self.D = D
        self.opt_D = torch.optim.AdamW(D.parameters(), lr=1e-5)
        self.lambda_gan = lambda_gan
        self.K = num_steps_train
        self.ttur_ratio = ttur_ratio

    def _sample_multistep(self, z, K=None, with_grad=False):
        """Backward simulation：按 inference schedule 跑 student，**返回每步 clean denoised output**。
        with_grad=True 整条链保留 grad（用于 generator loss）；False 时 detach 用于 D / fake denoiser。
        返回:
          x_finals: list of [B, ...] clean denoised outputs（含 final, len = K）
          x_noised_inputs: list of [B, ...] 喂给下一步 G 的 noisy inputs（len = K, 第一个 = z）
        论文里实际是按 EDM/TrigFlow schedule re-noise；此处用 sigma_next = t_next 当 placeholder。
        """
        K = K or self.K
        ts = torch.linspace(1.0, 0.0, K + 1, device=z.device)
        x_finals = []
        x_noised_inputs = []
        x_input = z
        ctx = torch.enable_grad() if with_grad else torch.no_grad()
        with ctx:
            for k in range(K):
                t_k = ts[k].expand(z.shape[0])
                x_noised_inputs.append(x_input)
                x_clean = self.G(x_input, t_k)   # denoised output
                x_finals.append(x_clean)
                if k < K - 1:
                    sigma_next = ts[k + 1]
                    # re-noise clean output 到下一 timestep 对应的 noisy state
                    x_input = x_clean + sigma_next * torch.randn_like(x_clean)
        return x_finals, x_noised_inputs

    def student_loss_dmd2(self, z, real_batch):
        # 1) backward simulation with grad：拿每步 clean denoised output
        x_finals, _ = self._sample_multistep(z, K=self.K, with_grad=True)
        # 2) DMD score gap：对每个 clean output 都算并平均（DMD2 论文 multi-step loss）
        loss_score = sum(self._score_gap_loss(x_c) for x_c in x_finals) / len(x_finals)
        # 3) GAN generator loss：D 在 final clean output 的 noised 版本上判别
        x_final = x_finals[-1]
        sigma = torch.exp(torch.randn(x_final.shape[0], device=x_final.device) * 1.6 - 1.0).view(-1, 1, 1, 1)
        x_fake_t = x_final + sigma * torch.randn_like(x_final)
        d_fake_logit = self.D(x_fake_t, sigma.squeeze())
        loss_adv = F.softplus(-d_fake_logit).mean()  # non-saturating
        return loss_score + self.lambda_gan * loss_adv

    def fake_score_loss(self, z):
        """Override：fake denoiser DSM target = student 的 multi-step backward-simulated outputs。
        DMD2 论文要 fake denoiser 学的是 generator 的整个 simulated 分布，而不是 1-step G(z)。
        """
        with torch.no_grad():
            x_finals, _ = self._sample_multistep(z, K=self.K, with_grad=False)
        # 在所有 K 步 clean outputs 上做 DSM
        loss_total = 0.0
        for x_c in x_finals:
            B = x_c.shape[0]
            sigma = torch.exp(torch.randn(B, device=x_c.device) * 1.6 - 1.0).view(-1, 1, 1, 1)
            alpha = 1.0
            x_t = alpha * x_c + sigma * torch.randn_like(x_c)
            pred_x0 = self.s_fake(x_t, sigma.squeeze())
            target_x0 = x_c.detach()
            loss_total = loss_total + ((pred_x0 - target_x0) ** 2).flatten(1).mean(1).mean()
        return loss_total / len(x_finals)

    def step(self, z_batch, real_batch):
        """DMD2 step：每 generator update 配 ttur_ratio 个 fake denoiser + D update。"""
        for _ in range(self.ttur_ratio):
            self.opt_f.zero_grad()
            loss_f = self.fake_score_loss(z_batch)
            loss_f.backward()
            self.opt_f.step()

            self.opt_D.zero_grad()
            loss_D = self.discriminator_loss(z_batch, real_batch)
            loss_D.backward()
            self.opt_D.step()
        # generator update
        self.opt_G.zero_grad()
        loss_G = self.student_loss_dmd2(z_batch, real_batch)
        loss_G.backward()
        self.opt_G.step()
        return loss_G.item(), loss_f.item(), loss_D.item()

    def _score_gap_loss(self, x_fake):
        # 复用 DMDTrainer.student_loss 的 denoiser-based score gap（见 §7.4），
        # 唯一差异：input 是 multi-step student output 而非 1-step。
        B = x_fake.shape[0]
        sigma = torch.exp(torch.randn(B, device=x_fake.device) * 1.6 - 1.0).view(-1, 1, 1, 1)
        alpha = 1.0
        x_t = alpha * x_fake + sigma * torch.randn_like(x_fake)
        with torch.no_grad():
            mu_real = self.s_real(x_t, sigma.squeeze())
            mu_fake = self.s_fake(x_t, sigma.squeeze())
            # 见 §7.4 推导：w_t(s_fake - s_real) = α(μ_fake - μ_real)
            grad_proxy = alpha * (mu_fake - mu_real)
            grad_proxy = grad_proxy / (
                (x_fake.detach() - mu_real).abs().mean(dim=(1, 2, 3), keepdim=True) + 1e-6
            )
        return (x_fake * grad_proxy.detach()).sum(dim=(1, 2, 3)).mean()

    def discriminator_loss(self, z, real_batch):
        """D 在 noised image 上判别 real vs student output。
        Softplus / non-saturating loss + 共享 fake denoiser backbone（教学版用独立 D）。
        """
        x_finals, _ = self._sample_multistep(z, with_grad=False)
        x_fake = x_finals[-1]
        B = real_batch.shape[0]
        sigma = torch.exp(torch.randn(B, device=real_batch.device) * 1.6 - 1.0).view(-1, 1, 1, 1)
        x_real_t = real_batch + sigma * torch.randn_like(real_batch)
        x_fake_t = x_fake.detach() + sigma * torch.randn_like(x_fake)
        d_real = self.D(x_real_t, sigma.squeeze())
        d_fake = self.D(x_fake_t, sigma.squeeze())
        return F.softplus(-d_real).mean() + F.softplus(d_fake).mean()
```

### 7.6　Code 6: ADD (Adversarial Diffusion Distillation, SDXL-Turbo 风格)

```python
import torchvision  # for DINOv2 backbone

class ADDTrainer:
    """ADD (Sauer 2023): pretrained DINOv2 backbone 当 discriminator"""
    def __init__(self, G, teacher_diffusion, sigma_data=0.5,
                 lambda_distill=1.0):
        self.G = G
        self.teacher = teacher_diffusion  # frozen
        # DINOv2 backbone + multiple discriminator heads
        self.dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
        self.dino.eval()
        for p in self.dino.parameters():
            p.requires_grad = False
        # multi-layer head: 从 DINOv2 不同 block 抽 feature, 各接一个 1x1 conv head
        self.disc_heads = nn.ModuleList([
            nn.Sequential(nn.Conv1d(1024, 1, 1), nn.Flatten())
            for _ in range(4)
        ])
        self.opt_G = torch.optim.AdamW(G.parameters(), lr=1e-5)
        self.opt_D = torch.optim.AdamW(self.disc_heads.parameters(), lr=1e-5)
        self.lambda_distill = lambda_distill

    def get_dino_features(self, x):
        """从 DINOv2 多层抽特征"""
        # 简化: dinov2_vitl14 的中间 layer hooks (实际需用 register_forward_hook)
        # 这里返回一个 list of features for each disc head
        x_resized = F.interpolate(x, size=224, mode='bilinear')
        # mock: 假设 backbone 给出 [B, 1024, N_patch] 序列, 4 个 layer
        feats = self.dino.get_intermediate_layers(x_resized, n=4)
        return feats

    def adv_loss_G(self, x_fake):
        feats = self.get_dino_features(x_fake)
        loss = 0
        for feat, head in zip(feats, self.disc_heads):
            logit = head(feat.transpose(1, 2))  # [B, ?]
            loss += -logit.mean()  # non-saturating
        return loss / len(self.disc_heads)

    def adv_loss_D(self, x_real, x_fake):
        feats_real = self.get_dino_features(x_real)
        feats_fake = self.get_dino_features(x_fake.detach())
        loss = 0
        for fr, ff, head in zip(feats_real, feats_fake, self.disc_heads):
            d_r = head(fr.transpose(1, 2))
            d_f = head(ff.transpose(1, 2))
            loss += F.relu(1 - d_r).mean() + F.relu(1 + d_f).mean()
        return loss / len(self.disc_heads)

    def distill_loss(self, z, x_fake):
        """teacher multi-step ODE output 做 supervision"""
        with torch.no_grad():
            x_teacher = self.teacher_ode_sample(z, steps=4)
        # pixel-level MSE
        return F.mse_loss(x_fake, x_teacher)

    @torch.no_grad()
    def teacher_ode_sample(self, z, steps=4):
        """teacher 跑 K-step ODE 出图，作为 student 的 distillation target"""
        # ... EDM Heun sampler, 省略具体实现
        return self.teacher.sample(z, num_steps=steps)

    def step(self, z, x_real):
        # 1) G output (1-step)
        x_fake = self.G(z)
        # 2) D loss
        self.opt_D.zero_grad()
        loss_D = self.adv_loss_D(x_real, x_fake)
        loss_D.backward()
        self.opt_D.step()
        # 3) G loss (adversarial + distill)
        x_fake = self.G(z)  # recompute (D updated)
        self.opt_G.zero_grad()
        loss_adv = self.adv_loss_G(x_fake)
        loss_dist = self.distill_loss(z, x_fake)
        loss_G = loss_adv + self.lambda_distill * loss_dist
        loss_G.backward()
        self.opt_G.step()
        return loss_G.item(), loss_D.item()
```

### 7.7　Code 7: LCM-LoRA 挂载到 SDXL

```python
# 假设有 diffusers 风格的 SDXL pipeline
from diffusers import StableDiffusionXLPipeline, LCMScheduler
from peft import LoraConfig, get_peft_model

def attach_lcm_lora(sdxl_pipe, lcm_lora_path="latent-consistency/lcm-lora-sdxl"):
    """LCM-LoRA: 把 LCM 蒸馏的差异权重作为 LoRA 挂上去。
    diffusers 当前 (>=0.24) `LCMScheduler` 的 teacher-step 参数名是 `original_inference_steps`，
    放在 scheduler config 里；老 community pipeline / 早期 dreamshaper 示例才用 `lcm_origin_steps`。
    """
    # 1) 切换 scheduler 为 LCM 风格；teacher-equivalent step 数放进 config
    sdxl_pipe.scheduler = LCMScheduler.from_config(
        sdxl_pipe.scheduler.config,
        original_inference_steps=50,   # 当前 diffusers LCMScheduler API
    )
    # 2) load LCM-LoRA weights
    sdxl_pipe.load_lora_weights(lcm_lora_path)
    # 3) 可选: 同时挂用户自己的 LoRA (e.g. character LoRA)
    # sdxl_pipe.load_lora_weights("path/to/user_lora", adapter_name="char")
    # sdxl_pipe.set_adapters(["default", "char"], adapter_weights=[1.0, 0.8])
    return sdxl_pipe

# 推理: 只需 4 step
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16
).to("cuda")
pipe = attach_lcm_lora(pipe)
images = pipe(
    prompt="a cat sitting on a chair",
    num_inference_steps=4,         # 关键: LCM 只需 4 step
    guidance_scale=0.0,            # HF LCM-LoRA 当前推荐 0.0；1.0-2.0 也可
).images
```

### 7.8　Code 8: Reflow (Rectified Flow distillation)

```python
@torch.no_grad()
def reflow_generate_pairs(v_net, num_samples, sample_shape, steps=50, device='cuda'):
    """用当前 v_θ 跑 ODE 生成 coupled (x_0, x_1) pair, 用于 reflow 重训。
    sample_shape: tuple，如 (D,) 用于 toy data，或 (C, H, W) 用于图像 latent。
    """
    x_0 = torch.randn(num_samples, *sample_shape, device=device)
    x = x_0.clone()
    ts = torch.linspace(0, 1, steps + 1, device=device)
    for i in range(steps):
        t = ts[i].expand(num_samples)
        dt = ts[i + 1] - ts[i]
        x = x + dt * v_net(x, t)
    return x_0, x  # x_1 = ODE(x_0; v_θ), 自然 coupled

def reflow_loss(v_net, x_0, x_1, t_dist='uniform'):
    """RF + reflow loss: 用 (x_0, x_1^{(k)}) coupled pair 重新训 v_θ^{(k+1)}"""
    B = x_0.shape[0]
    if t_dist == 'uniform':
        t = torch.rand(B, device=x_0.device)
    else:  # logit-normal (SD3 风格)
        t = torch.sigmoid(torch.randn(B, device=x_0.device))

    # 广播到任意 rank: t shape -> (B, 1, 1, ..., 1) 与 x_0 对齐
    t_view = t.view(B, *([1] * (x_0.ndim - 1)))
    x_t = (1 - t_view) * x_0 + t_view * x_1
    # target: u_t = x_1 - x_0 (常数)
    target = x_1 - x_0

    pred = v_net(x_t, t)
    return F.mse_loss(pred, target)

# 完整 reflow 训练流程
def train_with_reflow(v_net, data_loader, num_reflow_rounds=2, device='cuda'):
    """1st round: 独立 pair; 后续 rounds: coupled pair (reflow)"""
    # Round 0: 独立 pair (普通 RF 训练)
    for batch in data_loader:
        x_1 = batch[0] if isinstance(batch, (tuple, list)) else batch
        x_0 = torch.randn_like(x_1)
        loss = reflow_loss(v_net, x_0, x_1)
        # ... optimizer step

    # 从 data_loader 推断 sample shape（不依赖自定义 .dim 属性）
    first_batch = next(iter(data_loader))
    first_x1 = first_batch[0] if isinstance(first_batch, (tuple, list)) else first_batch
    sample_shape = tuple(first_x1.shape[1:])  # e.g. (D,) or (C, H, W)

    # Round 1, 2, ...: reflow
    for k in range(num_reflow_rounds):
        # 1) 用当前 v_net 生成 coupled pair
        x_0_pool, x_1_pool = reflow_generate_pairs(
            v_net, num_samples=10_000, sample_shape=sample_shape, device=device
        )
        # 2) 在 coupled pair 上重训
        from torch.utils.data import TensorDataset, DataLoader
        coupled_loader = DataLoader(
            TensorDataset(x_0_pool, x_1_pool), batch_size=64, shuffle=True
        )
        for x_0_b, x_1_b in coupled_loader:
            loss = reflow_loss(v_net, x_0_b, x_1_b)
            # ... optimizer step
    return v_net
```

## §8 Production Landscape

### 8.1　主流 production few-step 模型清单（2024-2026）

| 模型 | 蒸馏方法 | Base | Step | 分辨率 | 开源 |
|---|---|---|---|---|---|
| **LCM-SDXL / LCM-LoRA** | LCM (consistency on latent) | SDXL | 4–8 | 1024² | ✅ |
| **SDXL-Turbo** | ADD | SDXL | 1 | 512² | ✅ (weights only) |
| **SDXL-Lightning** | progressive + GAN | SDXL | 1/2/4/8 | 1024² | ✅ LoRA |
| **TCD-SDXL** | trajectory CD | SDXL | 4–8 | 1024² | ✅ |
| **DMD2-SDXL** | DMD2 (score gap + GAN) | SDXL | 1/4 | 1024² | ✅ |
| **SD3-Turbo** | LADD | SD3 8B | 4 | 1024² | API only |
| **FLUX.1-schnell** | LADD-style | FLUX 12B | 1–4 | 1024² | ✅ Apache 2.0 |
| **PixArt-LCM / PixArt-α-Lightning** | LCM / Lightning | PixArt-α | 4–8 | 1024² | ✅ |
| **SDXS** | feature alignment + GAN | SDXL | 1 | 512² | ✅ |

> ⚠️ **"开源"标 ✅ 不代表完全可商用** — SDXL-Turbo 早期非商用 license；FLUX-schnell Apache 2.0 商用 OK 但 FLUX-pro (teacher) 闭源。Production 上线前必查 license。

### 8.2　Video Distillation 现状

视频 diffusion 蒸馏在 2024-2025 才起步：

- **AnimateLCM** (Wang et al. 2024)：把 LCM 套到 AnimateDiff motion module，4-step 视频
- **VideoCrafter-LCM** / **CogVideoX-LCM**：类似套法
- **Hunyuan-Video-Lightning** / **Wan-Lightning**：用 Lightning 风格 + temporal-aware D
- **rCM** (arXiv:2510.08431)：把 sCM 扩到 Wan 2.1 14B / Cosmos-Predict2，1-4 step 5 秒视频

> 💡 **video 蒸馏的难点** — 静态图蒸馏的 D 直接看单帧；video 必须 D 看 **temporal coherence**——一种做法是 D 输入是 video clip（3D conv backbone），另一种是把单帧 D + flow-consistency loss 加在一起。这块工程经验比图像少得多。

### 8.3　部署 cheat sheet

| 场景 | 推荐方案 | 理由 |
|---|---|---|
| **移动端 / WebGPU** | SDXL-Turbo / FLUX-schnell 1-step | latency < 200 ms |
| **服务器 batch（API）** | DMD2-SDXL 4-step / SD3-Turbo | 质量+多样性平衡 |
| **二次开发（角色 LoRA）** | LCM-LoRA / SDXL-Lightning LoRA | 不破坏现有生态 |
| **学术 baseline** | sCM / CD / iCT | 数学清晰、复现性强 |
| **视频实时** | rCM / Wan-Lightning | 当前 SOTA |

## §9 失败模式 & 选型决策

### 9.1　常见 failure mode

| 现象 | 可能原因 | 对策 |
|---|---|---|
| **Mode collapse**（输出多样性低） | 1-step + 纯 MSE distill；ADD 没 GAN | 加 DMD score gap 或 GAN loss |
| **Saturated colors**（红黄过浓） | CFG 蒸进去 + step 太少 | 降 $w$；用 LCM-LoRA 4 step 而非 1 step |
| **High-freq detail blurry** | sCM mode-covering | 用 rCM 加 mode-seeking reg |
| **Text alignment 退化** | one-step CFG distill 不准 | 用 multi-condition $w$ 训练（LCM 风格） |
| **EMA collapse**（loss 卡住） | EMA decay 太高 / 太低 | 0.9999 起调，看 spectral norm |
| **Pseudo-Huber c 选错** | 太小 → L1 主导（不光滑）；太大 → 退化 L2 | iCT 论文 $c = 0.00054$（CIFAR），$c \propto \sqrt{D}$（D=维度） |
| **JVP NaN**（sCM） | warmup ratio $r$ 太快上 | NCS warmup：前 ~5% steps $r=0$，再渐增 |

### 9.2　选型决策树

```
Q1: base model 是什么?
  ├─ SD 1.5 / SDXL → LCM-LoRA (生态) 或 SDXL-Lightning
  ├─ SD3 / FLUX → LADD (官方 SD3-Turbo / FLUX-schnell)
  ├─ DiT / 自训 → sCM (continuous-time) 或 DMD2
  └─ Pixel-space (CIFAR/ImageNet) → CD / iCT / EDM teacher

Q2: 目标 NFE?
  ├─ 1-step → DMD2 / ADD / sCM / iCT
  ├─ 2-4 step → LCM / TCD / LADD
  └─ 8 step OK → progressive distillation / EDM Heun sufficient

Q3: 是否需要 CFG?
  ├─ 是 (text-to-image) → 用支持 CFG 蒸馏的方案 (LCM / LADD)
  └─ 否 (unconditional) → CD / DMD 直接用

Q4: 是否需要多样性?
  ├─ 高 (商业产品) → DMD2 / rCM (有 reverse KL / mode seeking)
  └─ 低 (固定 prompt) → ADD / Lightning 即可
```

### 9.3　Evaluation 指标

- **FID** (Fréchet Inception Distance)：图像质量 + 多样性的标准指标；越低越好
- **CLIP Score** / **CLIPSim**：text-image alignment
- **GenEval** (SD3)：对象计数 / 颜色 / 位置等结构化评估
- **HPSv2 / ImageReward**：人类偏好评分
- **PRD / Precision-Recall**：分别衡量"假图质量"和"covering 多样性"
- **Step-wise FID**：not just 1-step，还要 2/4/8-step 都看

> ⚠️ **FID 的陷阱** — FID 在 mode collapse 上**不敏感**——只算 mean/cov，可能漏掉只生成 50% mode 的 student。**必须配合 Precision-Recall** 或 IS / Coverage 指标交叉验证。

## §10 25 高频面试题（L1 必会 · L2 进阶 · L3 顶级 lab）

### L1 必会题（任何 ML / diffusion 岗位都可能问）

<details>
<summary>Q1. 为什么 diffusion 需要蒸馏？直接降 step 行不行？</summary>

- Diffusion sampling 50–1000 NFE，**网络前向占总延迟 >95%**，production 要 < 1 s 实时

- 直接减 step（如 50→4）会让 ODE 离散化误差爆炸：1-step Euler 误差 $O(\Delta t)$，4-step 时 $\Delta t$ 大 12.5×，图像高频细节崩塌

- 蒸馏的本质：**重新训一个 student**，让它学会"任意 $x_t$ 直接跳 $x_0$"（CM）或"输出分布匹配 teacher"（DMD）或"输出骗过 D"（ADD）

只说"diffusion 慢"不说网络前向是瓶颈；以为 DPM-Solver 就够了（10-NFE 是其物理极限）。

</details>

<details>
<summary>Q2. 写出 Consistency Models 的 consistency loss。</summary>

$$\mathcal{L}_\text{CD} = \mathbb{E}\big[d\big(f_\theta(x_{t_{n+1}}, t_{n+1}),\; f_{\theta^-}(\hat x_{t_n}, t_n)\big)\big]$$

- $\theta^-$ 是 EMA target

- $\hat x_{t_n} = x_{t_{n+1}} - (t_{n+1} - t_n) v_\phi(x_{t_{n+1}}, t_{n+1})$，由 teacher 一步 ODE 得

- $d$ = L2 或 LPIPS

混淆 EMA target 与 stop-gradient（前者可学，后者纯停梯度）；忘 boundary 用 EDM precond。

</details>

<details>
<summary>Q3. CD vs CT 区别？</summary>

- **CD (Consistency Distillation)**：有 teacher diffusion，用它一步 ODE 算 $\hat x_{t_n}$

- **CT (Consistency Training)**：无 teacher，用 $\hat x_{t_n} = x_0 + t_n \epsilon$（同 epsilon 加不同 noise level）

- 原始 CT 质量远低于 CD（CIFAR FID 8.7 vs 3.55），iCT 通过 pseudo-Huber + lognormal sigma + curriculum 把 CT 提到 2.83，**反超 CD**

只说"CT 不用 teacher" 不说 iCT 的改进；以为 CD 一定比 CT 好（已被 iCT 反例）。

</details>

<details>
<summary>Q4. DMD 的核心思想是什么？</summary>

- 把 student $G_\theta(z) \to x$ 当作直接的 generator

- 优化 reverse-KL：$\text{KL}(p_\text{fake}^\theta \| p_\text{real})$，梯度 = score gap × ∂G/∂θ

  $$\nabla_\theta \text{KL} = -\mathbb{E}[(s_\text{real} - s_\text{fake}) \cdot \partial G_\theta]$$

- $s_\text{real}$ = teacher diffusion（frozen），$s_\text{fake}$ = 在 $G_\theta$ 输出上训的 fake diffusion

只说"DMD 是 distribution matching" 不会写梯度；不知道 $s_\text{fake}$ 也是个 diffusion model。

</details>

<details>
<summary>Q5. DMD 和 GAN 的本质区别？</summary>

- GAN 用 discriminator 给 **binary 信号**（real/fake），sample efficiency 低

- DMD 用 **score gap = ∇log(p_real/p_fake)** 给 **dense vector field 信号**，告诉 student 每个点该往哪移动

- 物理直觉：score gap 就是把 student 从 $p_\text{fake}$ 推向 $p_\text{real}$ 的"力"

- DMD2 实际把 GAN loss 也加上当 fidelity 辅助

不知道 score gap 的物理含义；以为 DMD 只是 GAN 的 variant。

</details>

<details>
<summary>Q6. ADD（SDXL-Turbo）为什么用 DINOv2 当 discriminator？</summary>

- 普通 GAN 训 D from-scratch，对 1-step generator **不稳定**（mode collapse / 训不动）

- DINOv2 提供"pretrained 高级 perceptual feature"，anchor 判别问题到强语义空间

- 多个 layer head + hinge loss 让训练稳定

- 同时省去 D 的训练成本（D 主体冻结，只训 1×1 conv heads）

只说"DINOv2 好用"不说为什么不能 from scratch；不知道 head 是多层的。

</details>

<details>
<summary>Q7. LCM 和 CM 的核心区别？</summary>

- **空间**：LCM 在 VAE latent 空间（节省 64×），CM 在 pixel space

- **CFG**：LCM 把 guidance scale $w$ 作为额外 condition $f_\theta(x_t, t, c, w)$ 喂进网络，**推理时无需双 forward**；CM 原文不处理 CFG

- **Skipping-step distillation**：LCM 用 $k$-step skip 的 teacher 加速收敛

混淆 LCM 和 LCM-LoRA（后者是把 LCM 写成 LoRA adapter）。

</details>

<details>
<summary>Q8. Rectified Flow 的 reflow 算法？</summary>

1. 用独立 pair $(x_0, x_1) \sim p_0 \otimes p_\text{data}$ 训 $v_\theta^{(1)}$

2. 用 $v_\theta^{(1)}$ 跑 ODE 生成 coupled pair $(x_0, x_1^{(1)})$

3. 用 coupled pair 重训 $v_\theta^{(2)}$，新轨迹更"直"

- **transport cost 非增定理**：每次 reflow 总传输成本不增

- 1-2 次 reflow 后 1-step Euler 可媲美 50-step

只说"reflow 让轨迹变直" 不会推 transport cost 单调性；忘 InstaFlow 是 reflow 在 SD 上的应用。

</details>

<details>
<summary>Q9. SDXL-Turbo 和 SD3-Turbo / FLUX-schnell 的方法区别？</summary>

- **SDXL-Turbo (ADD)**：DINOv2 pixel-space discriminator + teacher MSE distillation

- **SD3-Turbo / FLUX-schnell (LADD)**：把 D 搬到 latent space，用 teacher MM-DiT 中间层 feature 当 D backbone，**支持高分辨率 + 高参数量 base**

- ADD 受 DINOv2 input 分辨率限制（≤ 518），LADD 无此限制

- FLUX-schnell 是 LADD 的 RF 版本

不知道 LADD 是 ADD 的 latent 版；以为 FLUX-schnell 是普通 CM 蒸馏。

</details>

<details>
<summary>Q10. LCM-LoRA 为什么生态价值大？</summary>

- LCM 训练的"差异权重" $\Delta\theta = \theta_\text{LCM} - \theta_\text{SD}$ 可参数化为 LoRA（$r \in [8, 64]$）

- 用户原 SD 1.5 / SDXL fine-tune（DreamShaper / 角色 LoRA）**无需重训**，挂上 LCM-LoRA 就能 4-step 出图

- 生态侧：SD 一家有上万 fine-tune 模型，LCM-LoRA 是**唯一不破坏现有生态的加速方案**

- 训练成本低（~30 A100 hours / SDXL）

只说 LCM-LoRA 是"LCM 的 LoRA 版" 不说生态意义；忘 LCM-LoRA 训练 cost 远小于 LCM。

</details>

### L2 进阶题（research-oriented · 需熟悉 diffusion 训练细节）

<details>
<summary>Q11. iCT 比 CT 提升的四个改动是什么？为什么 EMA 可以去掉？</summary>

四改动：

1. **去 EMA**：直接用 stop_grad 做 target

2. **Pseudo-Huber loss**：$\sqrt{\|a-b\|^2 + c^2} - c$ 替代 LPIPS，自适应 robust

3. **Lognormal noise schedule**：$\log\sigma \sim \mathcal{N}(P_\text{mean}, P_\text{std}^2)$ 替代 uniform

4. **Step-count curriculum**：$N$ 从 10 渐增到 1280

**为什么可以去 EMA**：原 CT 的 EMA 防止"网络输出对自身求导收敛到 trivial $f \equiv 0$"。pseudo-Huber + lognormal sigma 让 loss surface 更"凸"（small-residual region 主导），stop_grad 就足够防 collapse。

只背改动名不知道原因；以为 EMA 必须有（仍是误区）。

</details>

<details>
<summary>Q12. sCM 的 TrigFlow 参数化为什么能同时简化 EDM precond / PF-ODE / CM？</summary>

$$x_t = \cos(t) x_0 + \sin(t) z,\; t \in [0, \pi/2]$$

- **EDM precond**：$D_\theta = \cos(t) x_t - \sin(t) F_\theta$，boundary 自动满足（$t=0$ 时 $D = x_0$）

- **PF-ODE**：$dx_t/dt = -\sin(t) x_0 + \cos(t) z$，干净

- **CM**：consistency function $f_\theta = \cos(t) x_t - \sin(t)(\sigma_d F_\theta)$，形式与 EDM 同构

- 关键：$\cos^2 + \sin^2 = 1$（variance preservation），且 $d\cos/dt = -\sin$ 给出"自然"的 ODE 项

只说"用 sin cos 简单"不说为什么"恰好"四件事都简化；不知道 $\sigma^2 + \alpha^2 = 1$ 是 VP 条件。

</details>

<details>
<summary>Q13. sCM 的 NCS warmup 是什么？为什么需要？</summary>

- NCS = Noise → Consistency → Score（warmup 顺序）

- 训练初期 $r \approx 0$，sCM loss 退化为标准 score matching（学 $F_\theta \approx \epsilon$）

- 渐增 $r$，consistency 项（JVP）接管

- **没有 warmup 直接 $r = 1$**：网络还没学到 score，JVP 是噪声方向，训练 NaN

- 类似 GAN 训练里"先训 D 再 alternate"，先建立 base representation 再加难

只说 warmup 是"训练 trick"不说背后是 score 先于 consistency；不知道 JVP 不收敛会 NaN。

</details>

<details>
<summary>Q14. DMD2 比 DMD 改了哪些？为什么这些改动重要？</summary>

三改动：

1. **去掉 regression loss**：DMD v1 需要预生成 teacher pair（贵 + mode 受限）；DMD2 完全靠 score gap + GAN

2. **加 GAN loss**：判别器看真实数据 + student 输出，提供 high-freq detail 监督

3. **Multi-step student**：训练时模拟 $K$-step inference trajectory，让同一权重支持 1/2/4-step

**重要性**：

- 去 regression → 数据量解锁（不再依赖 teacher pair）
- 加 GAN → 与 DMD score gap 互补（score 给 distribution-level signal，GAN 给 sample-level fidelity）
- multi-step → production 灵活性（同一模型 1-step / 4-step 切换）

不知道为什么需要 multi-step（"1-step 就够了吗"）；忘 GAN 在 DMD2 里是 auxiliary 而非主 loss。

</details>

<details>
<summary>Q15. CFG 蒸馏的两阶段流程？</summary>

**Stage 1 - Guidance distillation** (Meng 2023)：训 $\tilde\epsilon_\theta(x, c, w)$，把 $w$ 作 condition 喂进网络——

$$\mathcal{L}_\text{guide} = \|\tilde\epsilon_\theta(x_t, c, w) - \tilde\epsilon^*(x_t, c, w)\|^2$$

其中 $\tilde\epsilon^* = (1+w) \epsilon_\theta(x, c) - w \epsilon_\theta(x, \emptyset)$ 是 teacher 跑两次 forward 得到的 CFG 输出。Student 只跑一次。

**Stage 2 - Step distillation**：在 stage 1 基础上叠 progressive distillation，把 32 step 蒸到 4/2/1 step。LCM 直接同时做 stage 1 + stage 2。

只知道有 CFG 蒸馏不会写两阶段；不知道 LCM-LoRA 的 $w$-condition 来自这。

</details>

<details>
<summary>Q16. EDM preconditioning 在 CM 里的作用？</summary>

$$f_\theta(x, \sigma) = c_\text{skip}(\sigma) x + c_\text{out}(\sigma) F_\theta(c_\text{in} x, c_\text{noise})$$

具体取值（Song 2023）：

$c_\text{skip} = \sigma_d^2 / ((\sigma - \sigma_\min)^2 + \sigma_d^2)$, $c_\text{out} = \sigma_d (\sigma - \sigma_\min) / \sqrt{\sigma_d^2 + \sigma^2}$

**作用**：

1. **Boundary 自动满足**：$\sigma = \sigma_\min$ 时 $c_\text{skip} = 1, c_\text{out} = 0$，所以 $f(x, \sigma_\min) = x$（identity）

2. **Unit-variance**：让 $F_\theta$ 输入输出方差与 $\sigma$ 无关，训练稳定

不知道 $c_\text{skip}(\sigma_\min) = 1$ 是 boundary 的关键；混淆 EDM precond 与 score-based reparam。

</details>

<details>
<summary>Q17. ADD 的 distillation loss 用 pixel MSE 而不是 score gap，会有什么问题？</summary>

- Pixel MSE 是 **mode-covering** + **blurry**：student 输出 = teacher mean，丢细节

- 加 GAN loss 才能补 high-freq → ADD 必须 GAN（不像 DMD 可纯 score gap）

- 这就是为什么 ADD 的 distill loss 只是 "anchor"（防 mode 严重塌缩），主战场是 GAN

- 对比 DMD：用 score gap → dense per-pixel gradient，不需 GAN 也能出图（但 DMD2 加 GAN 进一步提升）

只说"MSE 模糊"不说为什么；以为 ADD 不需要 GAN 也能 work。

</details>

<details>
<summary>Q18. CTM 比 CM 多了什么能力？</summary>

- CM：只学 $f(x_t, t) \to x_0$（轨迹终点）

- CTM：学 $G(x_t, t, s)$，**任意 $s < t$ 都可跳**

- 实际收益：

  - inference step 数 runtime 可选（CM 固定）
  - 中间状态可控（适合做 image-to-image / inpainting）
  - 训练 + score matching auxiliary loss 防 trivial

- FID：CIFAR 1-step 1.73 / ImageNet 64 1.92（SOTA）

只说"CTM 是 CM 的 trajectory 版" 不说为什么"任意 s"有用；混淆 CTM 和 TCD（后者是 LCM 改进）。

</details>

<details>
<summary>Q19. Reflow 的 transport cost 单调性怎么证？</summary>

**setup**：考虑独立 pair $(x_0, x_1) \sim p_0 \otimes p_1$，初始 cost $C^{(0)} = \mathbb{E}\|x_1 - x_0\|^2$。

**reflow**：用 $v_\theta^{(1)}$ 跑 ODE 得 coupled $(x_0, x_1^{(1)})$，cost $C^{(1)} = \mathbb{E}\|x_1^{(1)} - x_0\|^2$。

**关键观察**：

- $x_1^{(1)} = x_0 + \int_0^1 v_\theta^{(1)}(x_t, t)\, dt$

- 在 $L^2$ 下 $\|x_1^{(1)} - x_0\| = \|\int v\, dt\| \le \int \|v\|\, dt$（Cauchy-Schwarz）

- 而 $v_\theta^{(1)}$ 训练目标是 $\mathbb{E}\|v - (x_1 - x_0)\|^2$ 最小化 → 期望意义下 $\|v\| \approx \|x_1 - x_0\|$

- 严格定理（Liu 2022 Theorem 3.6）：$C^{(k+1)} \le C^{(k)}$（OT 视角下 reflow 不增 transport cost）

直觉：**直线是 OT 解** ⇒ 反复 reflow 推向 OT 解。

只说"轨迹变直"不会写 transport cost；不知道 Cauchy-Schwarz 直觉。

</details>

<details>
<summary>Q20. 蒸馏后的 student 怎么 evaluate？只看 FID 够吗？</summary>

**为什么 FID 不够**：

- FID 只算 Inception feature 的 mean + cov，对 **mode collapse 不敏感**（生成 50% mode 的 student FID 可能仍低）

- 对 high-freq detail 不敏感（Inception backbone 在 224×224 上 pool 严重）

**需要的辅助指标**：

- **Precision / Recall**（Kynkäänniemi 2019）：分别衡量"假图质量"和"覆盖多样性"

- **CLIP Score**：text-image alignment

- **HPSv2 / ImageReward / PickScore**：人类偏好

- **Step-wise FID**：1/2/4/8-step 都看，避免只优化 1-step

- **Mode count / coverage**：直接数生成图覆盖几个真实 cluster

只说"FID 就够"；忘人类偏好评估在 production 上线必看。

</details>

### L3 顶级 lab 题（research 深度 · 需会推导）

<details>
<summary>Q21. 从 PF-ODE 推 Consistency loss 的连续时间形式。</summary>

**PF-ODE**：$dx_t/dt = v_\phi(x_t, t)$（teacher）。

**Consistency 定义**：$f_\theta(x_{t+\Delta t}, t+\Delta t) = f_\theta(x_t, t)$ 沿同一 ODE 轨迹。

**一阶 Taylor**：

$$f_\theta(x_{t+\Delta t}, t+\Delta t) = f_\theta(x_t, t) + \Delta t \cdot \frac{d f_\theta}{dt} + O(\Delta t^2)$$

其中 $\frac{d f_\theta}{dt} = \partial_t f_\theta + (\nabla_x f_\theta)^\top \cdot \dot x_t = \partial_t f_\theta + (\nabla_x f_\theta)^\top v_\phi$（chain rule + PF-ODE 代入）。

**连续时间 consistency loss**：

$$\mathcal{L}_\text{cont}(\theta) = \mathbb{E}\!\left[\Big\|\partial_t f_\theta(x_t, t) + \nabla_x f_\theta(x_t, t) \cdot v_\phi(x_t, t)\Big\|^2\right]$$

**离散化**（CM 原版）：用 $\hat x_{t_n} = x_{t_{n+1}} + (t_n - t_{n+1}) v_\phi(\cdot)$ 当 teacher Euler，$f_{\theta^-}$ 当 target——

$$\mathcal{L}_\text{CD} \approx \mathbb{E}\|f_\theta(x_{t_{n+1}}, t_{n+1}) - f_{\theta^-}(\hat x_{t_n}, t_n)\|^2$$

只会写离散 loss 不会推连续形式；混淆 $\partial_t$ 和 $d/dt$（前者偏导后者全导）。

</details>

<details>
<summary>Q22. DMD 两个 score 的物理意义？为什么必须用 fake score 而不是 zero？</summary>

**物理意义**：

- $s_\text{real}(x, t) = \nabla_x \log p_\text{real}(x_t)$：把 $x_t$ 推向 real data 的"力"

- $s_\text{fake}(x, t) = \nabla_x \log p_\text{fake}(x_t)$：student 当前输出分布的 score

- 差 $s_\text{real} - s_\text{fake} = \nabla_x \log(p_\text{real}/p_\text{fake})$：reverse-KL 的梯度方向

**为什么 fake score 必要**：

- 如果只用 $s_\text{real}$（即 $s_\text{fake} \equiv 0$）：等价于把 student 推向 "$p_\text{real}$ 的 mode"——**mode collapse**

- $s_\text{fake}$ 提供"已经覆盖的位置不需要再推"的信号，类似 GAN 的 D 提供 contrastive feedback

- 数学：$\mathbb{E}_{p_\text{fake}}[s_\text{real} - s_\text{fake}]$ 是 Stein discrepancy，正确的 distribution matching 信号

**实现**：

- $s_\text{real}$ = teacher diffusion（frozen）
- $s_\text{fake}$ = 一个小 diffusion model，**在 $G_\theta$ 当前输出上做 DSM**，与 $G_\theta$ 联训

只说"DMD 用 score" 不说两个的角色区别；不知道 $s_\text{fake}$ 需要联训。

</details>

<details>
<summary>Q23. ADD vs LADD 的 scale 差异本质在哪？为什么 ADD 上不到 SD3 8B / FLUX 12B？</summary>

**ADD bottleneck**：

1. **DINOv2 input 分辨率**：ADD 用 DINOv2 base（518²）当 D backbone，超过此分辨率必须 patch / downsample，1024² 输入受限

2. **Pixel-space distill**：`MSE(G(z), teacher_ode(z))` 需 VAE decode，**back-prop 通过 VAE 贵且不稳**

3. **Discriminator capacity**：DINOv2 ViT-L 1B 参数远小于 SD3 8B / FLUX 12B 的 base，D 表达力不够

**LADD 解法**：

1. **Latent space**：D 直接在 VAE latent 上跑（128×128×16 for SD3），分辨率无关

2. **Teacher 自身的 MM-DiT block 当 D backbone**：把 SD3 自己的 transformer block 抽出来 fine-tune 成 D，**capacity 自动匹配 base 规模**

3. **Score distill in latent**：避开 VAE back-prop

**结果**：SDXL-Turbo（2.6B SDXL ADD）做到 1024² 已是 ADD 上限；SD3-Turbo（8B LADD）/ FLUX-schnell（12B LADD-style）需要 LADD 才能稳定训出。

只说"LADD 在 latent space" 不说为什么 ADD 上不到大模型；忘 DINOv2 分辨率限制是 hard cap。

</details>

<details>
<summary>Q24. Flow-OPD（2026 arXiv:2605.08063）与 DMD 在数学上有什么联系？</summary>

> 📍 **澄清**：Flow-OPD 主要是 multi-reward RL alignment paper，与本文 few-step inference distillation 主线略偏；这里出现是因为 name 包含 "Distillation"，详细讨论见 [diffusion_post_training_tutorial.md](diffusion_post_training_tutorial.md)。

**DMD**：reverse-KL 梯度（在 student 输出分布上），single teacher，single objective = match teacher distribution。

**Flow-OPD**：on-policy distillation with multiple **reward-specific** teachers（每个 reward GRPO fine-tuned 一个 specialist），是 **alignment paper**（多 reward 对齐）而非 inference distillation 论文。

**说"DMD 退化到 OPD"是错的**：DMD 的 reverse-KL 与 OPD 的 multi-teacher vector-field weighting 是**不同的数学目标**——一个是分布匹配，一个是 reward-aware policy supervision。两者目标侧重不同（**single-teacher distribution match vs multi-reward alignment**），没有 reduction 关系。面试中**不要**说"DMD 是 OPD 的特例"或反之，无可靠数学依据。

**实践意义**（safer 表述）：
- **DMD 更适合 few-step inference**（单一目标：match teacher 分布）
- **Flow-OPD 更适合 multi-reward alignment**（多 reward 对齐 + on-policy 训练）
- 两者解决不同问题，并非替代关系；详细 alignment 内容见 [`diffusion_post_training_tutorial.md`](diffusion_post_training_tutorial.md)

只把 Flow-OPD 当"另一种 inference distillation"是混淆 — 它的 multi-reward / RL 性质是核心；同样不要把"reduction to DMD"作为既定数学结论。

</details>

<details>
<summary>Q25. 设计一个能在 4 step 跑 1024² 视频 + 保 temporal coherence 的蒸馏方案。给出 loss 和 D 设计。</summary>

**Setup**：

- Teacher：50-step video diffusion (e.g. Wan 2.1 14B, Rectified Flow)
- Student：4-step video generator $G_\theta(z_{1:T}, c)$
- Target：1024² × 5 sec

**Loss 组合**（rCM-style + LADD-style）：

$$\mathcal{L}_\text{total} = \underbrace{\mathcal{L}_\text{sCM}^\text{trig}}_{\text{video CM, JVP-based}} + \lambda_1 \cdot \underbrace{\mathcal{L}_\text{score-reg}}_{\text{mode-seeking via score gap}} + \lambda_2 \cdot \underbrace{\mathcal{L}_\text{adv}^\text{video}}_{\text{temporal D}}$$

**Video Discriminator 设计**：

- **Backbone**：teacher 自己的 3D MM-DiT block（latent space，避开 VAE decode）

- **两个 head**：
  - **Spatial head**：单帧 latent → real/fake 信号（图像 quality）
  - **Temporal head**：连续 $k$-frame latent stack → real/fake（motion realism）

- **Optical flow consistency loss**（辅助）：
  $$\mathcal{L}_\text{flow} = \mathbb{E}\|f_\text{flow}(\hat x_{t}, \hat x_{t+1}) - f_\text{flow}(x_t^\text{real}, x_{t+1}^\text{real})\|$$

**训练 tricks**：

- **Multi-stage**：先在静态图（$T = 1$）上预训 → 再加 temporal D → 最后 fine-tune full video
- **Curriculum on T**：短片段先训（$T = 8$ frame）→ 长片段（$T = 80$ frame）
- **EMA on G**：避免 student 输出在不同 step 间 drift

**Evaluation**：

- VBench (静态质量 + 动态质量 16 维)
- FVD (Fréchet Video Distance)
- 人类对照（rCM-style）

**对比 baseline**：rCM 已在 Wan 2.1 14B 上做到接近——这是 production-grade direction，2026 还在快速发展。

需要把"图像蒸馏 + temporal 监督 + 大 base"三件事融合；只用单 D 看单帧会 motion 崩；只用 score-gap 没 GAN 会 detail 模糊。

</details>

## §A 附录：参考文献

**Consistency Models 家族**：

- Song et al. 2023, "Consistency Models", ICML 2023, [arXiv:2303.01469](https://arxiv.org/abs/2303.01469)
- Song & Dhariwal 2023, "Improved Techniques for Training Consistency Models" (iCT), [arXiv:2310.14189](https://arxiv.org/abs/2310.14189)
- Lu & Song 2024, "Simplifying, Stabilizing and Scaling Continuous-Time Consistency Models" (sCM / TrigFlow), ICLR 2025, [arXiv:2410.11081](https://arxiv.org/abs/2410.11081)
- Kim et al. 2023, "Consistency Trajectory Models: Learning Probability Flow ODE Trajectory of Diffusion" (CTM), ICLR 2024, [arXiv:2310.02279](https://arxiv.org/abs/2310.02279)
- Luo et al. 2023, "Latent Consistency Models" (LCM), [arXiv:2310.04378](https://arxiv.org/abs/2310.04378)
- Luo et al. 2023, "LCM-LoRA: A Universal Stable-Diffusion Acceleration Module", [arXiv:2311.05556](https://arxiv.org/abs/2311.05556)
- Zheng et al. 2024, "Trajectory Consistency Distillation" (TCD), [arXiv:2402.19159](https://arxiv.org/abs/2402.19159)
- "Large Scale Diffusion Distillation via Score-Regularized Continuous-Time Consistency" (rCM), [arXiv:2510.08431](https://arxiv.org/abs/2510.08431) (rCM acronym verified)

**Distribution Matching Distillation**：

- Yin et al. 2024, "One-step Diffusion with Distribution Matching Distillation" (DMD), CVPR 2024, [arXiv:2311.18828](https://arxiv.org/abs/2311.18828)
- Yin et al. 2024, "Improved Distribution Matching Distillation for Fast Image Synthesis" (DMD2), NeurIPS 2024, [arXiv:2405.14867](https://arxiv.org/abs/2405.14867)

**Adversarial Distillation**：

- Sauer et al. 2023, "Adversarial Diffusion Distillation" (ADD / SDXL-Turbo), [arXiv:2311.17042](https://arxiv.org/abs/2311.17042)
- Sauer et al. 2024, "Fast High-Resolution Image Synthesis with Latent Adversarial Diffusion Distillation" (LADD / SD3-Turbo), [arXiv:2403.12015](https://arxiv.org/abs/2403.12015)
- Lin et al. 2024, "SDXL-Lightning: Progressive Adversarial Diffusion Distillation", [arXiv:2402.13929](https://arxiv.org/abs/2402.13929)

**Flow / Rectified Flow**：

- Liu, Gong & Liu 2022, "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow", ICLR 2023, [arXiv:2209.03003](https://arxiv.org/abs/2209.03003)
- Liu et al. 2023, "InstaFlow: One Step is Enough for High-Quality Diffusion-Based Text-to-Image Generation", ICLR 2024, [arXiv:2309.06380](https://arxiv.org/abs/2309.06380)
- "Flow-OPD: On-Policy Distillation for Flow Matching Models", [arXiv:2605.08063](https://arxiv.org/abs/2605.08063) (Flow-OPD 主要是 multi-reward RL alignment paper，与本文 few-step inference distillation 主线略偏；详见 `diffusion_post_training_tutorial.md`)

**CFG / Step Distillation**：

- Meng et al. 2023, "On Distillation of Guided Diffusion Models", CVPR 2023, [arXiv:2210.03142](https://arxiv.org/abs/2210.03142)
- Salimans & Ho 2022, "Progressive Distillation for Fast Sampling of Diffusion Models", ICLR 2022, [arXiv:2202.00512](https://arxiv.org/abs/2202.00512)

**Foundations**：

- Ho, Jain & Abbeel 2020, "Denoising Diffusion Probabilistic Models", NeurIPS 2020 (DDPM)
- Song et al. 2021, "Score-Based Generative Modeling through Stochastic Differential Equations", ICLR 2021
- Karras et al. 2022, "Elucidating the Design Space of Diffusion-Based Generative Models" (EDM), NeurIPS 2022, [arXiv:2206.00364](https://arxiv.org/abs/2206.00364)
- Lipman et al. 2023, "Flow Matching for Generative Modeling", ICLR 2023

**Production models**：

- Stable Diffusion XL: Podell et al. 2024 ICLR
- Stable Diffusion 3: Esser et al. 2024 ICML
- FLUX.1: Black Forest Labs 2024 (technical report)

**Diffusion / Flow Distillation Cheat Sheet** · 主要参考：Song 2023 (CM), Lu-Song 2024 (sCM), Yin 2024 (DMD/DMD2), Sauer 2023/2024 (ADD/LADD), Liu 2022 (RF)
