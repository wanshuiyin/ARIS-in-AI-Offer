## §0 TL;DR Cheat Sheet

> 💡 **一句话** — 本文的三种方法共享 GRPO 式的 **group sampling** 框架（同一 prompt 采 $G$ 张，组内归一化 reward），分岔只在一处：**reward 怎么进梯度**。

1. **Flow-GRPO**（Liu et al. 2025，arXiv 2505.05470）：把 ODE 改写成保边缘的 **SDE**，得到可算的 Gaussian 单步转移密度，然后做 **advantage 加权的 PPO-clip 策略梯度**。代价：训练 rollout 必须 SDE（推理仍可 ODE）、要存整条轨迹、每步算 likelihood、有离散化偏差。
2. **DGPO**（Luo, Hu, Tang 2025，arXiv 2510.08425，ICLR 2026）：**彻底不要策略梯度**，把 DPO 从"一对样本"推广到"一对子组"——正组 vs 负组进一个 sigmoid；likelihood 用 Diffusion-DPO 那套 **DSM 差**代替。于是可以用**确定性 ODE** 采样，只存干净图 + reward。报告比 Flow-GRPO 快约 20×。
3. **DiffusionNFT**（Zheng, Chen et al. 2025，arXiv 2509.16117，ICLR 2026 Oral）：在**前向加噪过程**上做 RL——一个 flow-matching 回归 loss，正例拉向 target、负例通过关于 old 的**反射参数化**推开。不需要 likelihood、不依赖 solver、推理 **CFG-free**。报告在 GenEval 上比 Flow-GRPO 快最多 25×。
4. 三者都**不要 critic**；但它们分别是 policy gradient / group preference logistic / reward-weighted 双支回归，不能笼统叫"三种 policy gradient"。
5. **old ≠ ref**：old 是采样时的行为策略（importance ratio 的分母、NFT 的反射中心），ref 是 KL 约束基准（Flow-GRPO 的 KL、DGPO 的 DSM 差）。NFT 的核心 loss 不需要固定 ref。
6. 三个数字的口径要分清：Flow-GRPO 自报 GenEval 0.63→0.95（CFG base）、PickScore 21.72→23.31（带 KL；无 KL 是 23.41）；DGPO 0.97；NFT 0.24→0.98（**0.24 是 CFG-free base**）。DGPO 的 3.74 vs 3.66 是 **UnifiedReward**，不是 PickScore。
7. 面试杀手题：**reverse SDE 的符号**（生成时间递减，drift 是 $v-\tfrac12 g^2 s$）、**DGPO 为什么能消掉 $\log Z$**（组内 advantage 和为零 ⇒ 正负权重总量相等）、**NFT 的负支为什么是 $(1+\beta)$**（关于 old 的反射）。

---

## §1 直觉：三条路的一句话

### 1.1　共同的起点：LLM GRPO

GRPO（Shao et al. 2024，DeepSeekMath）去掉了 PPO 的 critic：对同一个 prompt 采 $G$ 个回答，用组内均值当 baseline：

$$\hat A_i = \frac{R_i - \operatorname{mean}_j R_j}{\operatorname{std}_j R_j}$$

然后做 PPO-clip。搬到 diffusion 上，"一个回答"变成"一张图"，reward 是 GenEval 这类规则验证器、OCR 准确率或 PickScore 这类人偏好模型。三种方法共享同一个采集框架：$P$ 个 prompt × $G$ 张图 × 打分 × 组内归一化（solver、reward 归一化和 collector 的更新方式各有不同）。组内 reward 全同时标准差为零，约定 advantage 取零。

### 1.2　分岔点：advantage 落在哪个量上

| 方法 | reward 进梯度的方式 | 训练时需要 | 采样器 |
| --- | --- | --- | --- |
| Flow-GRPO | 加权每一步转移的 $\log p_\theta(x_{k+1}\mid x_k)$（策略梯度） | 整条轨迹 + 每步 old log-prob | 训练 rollout 必须 SDE |
| DGPO | 正组、负组的 reference-relative DSM 差进一个 sigmoid（偏好学习） | 干净图 + reward，重新加噪 | 任意（ODE 可） |
| DiffusionNFT | 正例/负例两支 flow-matching 回归的权重（监督学习形式） | 干净图 + reward，重新加噪 | 任意（黑盒） |

### 1.3　为什么 ODE 会卡住 Flow-GRPO 而不卡另外两个

ODE 采样**不是没有随机性**——不同初始噪声就给出不同样本。卡住的是另一件事：ODE 的单步转移 $x_k \to x_{k+1}$ 是确定的（Dirac），没有密度可写，PPO 的 importance ratio $p_\theta/p_\text{old}$ 无从谈起。Flow-GRPO 的解法是把 ODE 换成同边缘的 SDE；DGPO 和 NFT 的解法是**根本不用采样路径的 likelihood**——训练时对干净图重新加噪 $x_t = (1-t)x_0 + t\epsilon$，用的是前向过程 $q(x_t\mid x_0)$，和采样器无关。

> ⚠️ **不要说"ODE 无法探索"** — 探索靠初始噪声就够了。缺的是可计算的随机单步密度，以及中间步骤的额外探索。

---

## §2 Convention（全文统一）

### 2.1　Rectified Flow 记号

$$x_t = (1-t)\,x_0 + t\,\epsilon,\qquad t\in[0,1],\quad x_0\sim p_\text{data},\ \epsilon\sim\mathcal N(0,I)$$

- $t=0$ 是干净图，$t=1$ 是纯噪声；**生成沿 $t: 1\to 0$**。
- 训练配对的 target 是 $v = \epsilon - x_0$；网络学的是条件期望 $v^*(x_t,t,c) = \mathbb E[\epsilon - x_0 \mid x_t, t, c]$，**不是**某一个配对的随机 target。
- SDE 的扩散系数记 $g(t)$，和前向加噪系数 $\sigma_\text{path}(t) = t$ 分开写。三种方法各有自己的 $\beta$，含义不同，**不能跨方法比大小**。

### 2.2　换算表（背下来）

由 $x_t = (1-t)x_0 + t\epsilon$ 和 $v = \epsilon - x_0$：

$$\hat x_0 = x_t - t\,v_\theta,\qquad \hat\epsilon = x_t + (1-t)\,v_\theta,\qquad s_\theta(x_t) := -\frac{\hat\epsilon}{t}$$

验证 $\hat\epsilon$：$x_t + (1-t)v = (1-t)x_0 + t\epsilon + (1-t)\epsilon - (1-t)x_0 = \epsilon$。score 用的是**预测的条件均值** $\hat\epsilon$，不是真实的 $\epsilon$（那是未知量）。$s_\theta$ 只在 $v_\theta = v^*$ 时等于真实的 $\nabla_x\log p_t$——§3.2 和 Q24 说的「保边缘不严格继承」就是这一点。

### 2.3　三个模型的角色

| 名字 | 谁 | 干什么 | Flow-GRPO | DGPO | NFT |
| --- | --- | --- | --- | --- | --- |
| **policy** $\theta$ | 正在更新的 | 被优化 | ✅ | ✅ | ✅ |
| **old** $\theta_\text{old}$ | 采样时的 | ratio 分母 / 反射中心 | ✅（rollout） | collector（可 EMA） | ✅（EMA） |
| **ref** $\theta_\text{ref}$ | 固定不动的 | KL / DSM 差的基准 | ✅（KL 项） | ✅（DSM 差） | ❌（核心 loss 不需要） |

---

## §3 Flow-GRPO：把 ODE 变成 SDE，然后做 PPO

Liu et al. 2025, *Flow-GRPO: Training Flow Matching Models via Online RL*, arXiv 2505.05470（Kling / CUHK / 清华）。

### 3.1　保边缘的 SDE：推导

**目标**：找一条 SDE，它在每个 $t$ 的边缘分布 $p_t$ 和原 ODE $\dot x = v$ 完全一样，但单步转移是 Gaussian。

用**正向生成时钟** $u = 1 - t$（$u$ 从 0 到 1，对应 $t$ 从 1 到 0）推导最清楚。原 ODE 在 $u$ 上是 $dX_u = -v\,du$。加上一对互相抵消的项：

$$dX_u = \big[-v + \tfrac12 g^2 s\big]\,du + g\,dW_u$$

Fokker–Planck 方程里，drift 中的 $\tfrac12 g^2 s$ 贡献 $-\tfrac12 g^2\nabla\cdot(p\,s)$，扩散项贡献 $+\tfrac12 g^2\Delta p$；因为 $p\,s = p\nabla\log p = \nabla p$，两项恰好抵消，剩下的正是 ODE 的密度演化。所以边缘不变。

换回 $t$ 时钟（$dt = -du$）：

$$dx_t = \big[v - \tfrac12 g^2 s\big]\,dt + g\,d\bar W_t \qquad (t \text{ 递减})$$

这就是论文 Eq. 7。代入 §2.2 的 $s = -\hat\epsilon/t = -(x_t + (1-t)v)/t$，得论文 Eq. 8：

$$dx_t = \Big[v + \frac{g^2}{2t}\big(x_t + (1-t)v\big)\Big]dt + g\,d\bar W_t$$

> ⚠️ **符号取决于时间方向** — 时间**递增**的保边缘 SDE drift 是 $v + \tfrac12 g^2 s$，时间**递减**的 reverse SDE drift 是 $v - \tfrac12 g^2 s$。旧版教程把生成用的 SDE 写成 $+\tfrac12\sigma^2 s$，是错的。

### 3.2　离散化：统一用正步长

令 $h = t_k - t_{k+1} > 0$，Euler–Maruyama：

$$x_{k+1} = x_k - h\Big[v_\theta + \frac{g^2}{2t_k}\big(x_k + (1-t_k)v_\theta\big)\Big] + g\sqrt{h}\,\xi,\qquad \xi\sim\mathcal N(0,I)$$

论文 Eq. 9 用带符号的 $\Delta t$ 写，直接照抄会出现"对负步长开平方"的困惑——用 $h>0$ 写清楚。

**噪声 schedule**：$g(t) = a\sqrt{t/(1-t)}$，默认 $a = 0.7$。从 $t=1$ 起步时分母为零，官方实现用相邻时间点处理首步，不是直接算无穷大。

**保边缘是连续时间性质**：要求 $v$ 和 $s$ 对应同一族边缘；学习误差、加了 CFG 的场、有限步离散化都不自动继承严格相等。这是为什么 3.5 的 denoising reduction 是经验观察而非定理。

### 3.3　单步策略、log-prob、ratio

**Euler 离散后的**单步转移是各向同性 Gaussian（一般非线性 SDE 的精确有限时转移并不是）：$p_\theta(x_{k+1}\mid x_k) = \mathcal N\big(\mu_\theta(x_k),\ g^2 h\,I\big)$，$\mu_\theta$ 就是 3.2 去掉噪声项的那一行。

$$\log p_\theta(x_{k+1}\mid x_k) = -\frac{\lVert x_{k+1} - \mu_\theta\rVert^2}{2g^2h} - \frac d2\log(2\pi g^2 h)$$

对 latent 的全部 $d$ 维**求和**。官方代码按维度平均 log-prob，得到的是 $\rho^{1/d}$（密度 ratio 的 $d$ 次方根），不是对 ratio 乘常数——和论文的密度公式要区分。

Importance ratio 的分母是 **old**（rollout 时的策略），不是 ref：

$$\rho_k = \exp\big(\log p_\theta(x_{k+1}\mid x_k) - \log p_\text{old}(x_{k+1}\mid x_k)\big)$$

### 3.4　Loss：PPO-clip + 闭式 KL

$$\mathcal L_\text{Flow-GRPO} = -\,\mathbb E\Big[\min\big(\rho_k \hat A,\ \operatorname{clip}(\rho_k, 1-\varepsilon_c, 1+\varepsilon_c)\hat A\big)\Big] + \beta_\text{KL}\,\mathbb E\big[\mathrm{KL}(p_\theta\Vert p_\text{ref})\big]$$

负 advantage 时仍用 $\min$，不换 $\max$（这是 PPO 的标准形式）。$\hat A$ 是终态图的组内归一化 reward，整条轨迹的每一步共享。

**KL 有闭式**——两个同协方差 Gaussian 的 KL 只剩均值差：

$$\mathrm{KL}(p_\theta\Vert p_\text{ref}) = \frac{\lVert\mu_\theta - \mu_\text{ref}\rVert^2}{2g^2h}$$

代入 3.2 的 drift，$\mu_\theta - \mu_\text{ref} = -h\Big(1 + \frac{g^2(1-t)}{2t}\Big)(v_\theta - v_\text{ref})$，所以

$$\mathrm{KL} = \frac h2\Big(\frac1g + \frac{g(1-t)}{2t}\Big)^2\lVert v_\theta - v_\text{ref}\rVert^2$$

系数恒非负。这是**条件单步 KL**；沿链求和对应路径 KL，**不等于**终态图像分布的 KL。旧版教程说"KL 用 K3 估计"——原文给的是闭式，不需要 K3。

### 3.5　Denoising reduction

采一组 $G$ 条轨迹，每条 $T$ 步，成本 $\propto GT$。Flow-GRPO 训练时 $T_\text{train} = 10$，评估时 $T_\text{infer} = 40$；论文报告"超过 4×"加速——来自 40→10 步、无 KL 的消融训练曲线，不是普适定律。为什么少步训练不掉点：连续边缘与步数无关，RL 学的是 $v_\theta$ 的方向修正，与具体步数耦合较弱——这是经验观察加近似，不是严格等价。

### 3.6　梯度边界（面试常追问）

- **固定**：rollout 的状态序列、old log-prob、reward、advantage。
- **重算**：当前策略在这些状态上的 log-prob。
- **stop-grad**：ref 的输出。
- "存整条轨迹"存的是状态，**不是**整条 rollout 的反传图。

正 advantage 的几何意义：把 transition mean $\mu_\theta$ 推向实际采到的下一状态 $x_{k+1}$。要翻译成"$v_\theta$ 往哪推"，得经过负时间步和完整的 drift 系数——不能直接说"$v_\theta$ 指向下一张 latent"。

### 3.7　结果与代价

SD3.5-M 上（每行是**不同 reward 分别训练**的模型）：

| Reward | base | Flow-GRPO | $\beta_\text{KL}$ |
| --- | --- | --- | --- |
| GenEval | 0.63 | **0.95** | 0.04 |
| OCR | 0.59 | **0.92** | 0.04 |
| PickScore | 21.72 | **23.31** | 0.01 |

$G = 24$，$a = 0.7$。代价清单：训练 rollout 只能 SDE 采样（评估关掉额外噪声、用 ODE）；轨迹 buffer $O(PGTd)$；每步一次 old/policy/ref 三份前向；PPO surrogate + 离散化两层近似。

> 💡 **相关工作只记三条** — **DanceGRPO** 把 GRPO 推到图像/视频、diffusion/RF 各种 backbone；**MixGRPO / Flow-GRPO-Fast** 只在一段时间窗口内用 SDE + GRPO 更新，窗口外用 ODE（所以"SDE 必须覆盖整条链"是过度概括）；**GRPO-Guard** 指出 importance ratio 的分布随 timestep 漂移会削弱 clipping，用 RatioNorm 和梯度重加权修（§8 再提）。

---

## §4 DGPO：不要策略梯度，把 DPO 推广到组

Luo, Hu, Tang 2025, *Reinforcing Diffusion Models by Direct Group Preference Optimization*, arXiv 2510.08425, ICLR 2026（CUHK-SZ / HKUST-GZ）。

### 4.1　动机

Flow-GRPO 的一切代价都来自"需要随机单步转移密度"。DGPO 问：能不能保留 group sampling 的 advantage 信号，但**用偏好学习代替策略梯度**？DPO 已经证明单对样本可以；把"一对样本"换成"一对子组"就是 DGPO。

### 4.2　从 DPO 到 group DPO

KL 正则 RL 的最优策略 $p^*(x_0\mid c) \propto p_\text{ref}(x_0\mid c)\,e^{R(x_0,c)/\beta}$，反解出隐式 reward：

$$r_\theta(c, x_0) = \beta\log\frac{p_\theta(x_0\mid c)}{p_\text{ref}(x_0\mid c)} + \beta\log Z(c)$$

DPO 把两个样本的 $r_\theta$ 差放进 Bradley–Terry，$\log Z(c)$ 相减抵消。DGPO 定义**组级 reward** 为加权和：

$$R_\theta(\mathcal G\mid c) = \sum_{x_0\in\mathcal G} w(x_0)\, r_\theta(c, x_0)$$

目标：

$$\max_\theta\ \mathbb E\,\log\sigma\big(R_\theta(\mathcal G^+\mid c) - R_\theta(\mathcal G^-\mid c)\big)$$

### 4.3　正负组怎么分、$\log Z$ 怎么消

同一 prompt 的同一组 $G$ 张图，组内归一化 $A_i = (R_i - \bar R)/s_R$，然后

$$\mathcal G^+ = \{x_0^i: A_i > 0\},\qquad \mathcal G^- = \{x_0^i: A_i \le 0\},\qquad w(x_0^i) = \lvert A_i\rvert$$

**关键一步**：因为 $\sum_i A_i = 0$，

$$\sum_{\mathcal G^+}\lvert A_i\rvert = \sum_{\mathcal G^+} A_i = -\sum_{\mathcal G^-} A_i = \sum_{\mathcal G^-}\lvert A_i\rvert$$

正负权重**总量相等**。于是 $\beta\log Z(c)$ 乘以 $\big[\sum_{\mathcal G^+}w - \sum_{\mathcal G^-}w\big] = 0$，被消掉。

> ⚠️ **平衡的是权重总量，不是人数** — 正负组通常人数不等（比如 1:3）。如果把两边各自按人数取平均，抵消就坏了，目标也变了。

### 4.4　DSM 代替 likelihood

$r_\theta$ 里的 $\beta\log\frac{p_\theta(x_0\mid c)}{p_\text{ref}(x_0\mid c)}$ 对 diffusion 不可算。沿用 Diffusion-DPO 的路：把这个终态 log-ratio 用沿加噪路径的 reference-relative 量 $\beta\,\mathbb E_{q(x_{1:T}\mid x_0)}\log\frac{p_\theta(x_{0:T}\mid c)}{p_\text{ref}(x_{0:T}\mid c)}$ 近似（是近似，不是精确等式），再化成每个 timestep 的**去噪回归误差之差**。记

$$d_i = L^\theta_{\text{dsm},i} - L^\text{ref}_{\text{dsm},i}$$

最终 loss（论文 Eq. 17 的紧凑写法）：

$$\mathcal L_\text{DGPO} = -\,\mathbb E_{t,\epsilon}\log\sigma\Big(-\lambda_t\,\beta\,T\sum_i A_i\, d_i\Big)$$

（$A_i$ 自带符号，所以正组减负组已经包含在内。）**符号自检**：preferred 样本的当前误差比 ref 降低，或 dispreferred 的当前误差比 ref 升高，都让 logit 变大——整个组共享一个 logistic 梯度系数。

两点细节：

- DSM 定义在去噪回归误差上。换成 RF 的 velocity 参数化时 $\lVert\hat x_0 - x_0\rVert^2 = t^2\lVert v_\theta - v\rVert^2$，系数要跟着换，不能换了预测目标还说 loss 逐字相同。
- **Jensen 的方向**：$-\log\sigma(\mathbb E z) \le \mathbb E[-\log\sigma(z)]$。单个 timestep 上算的是 surrogate，不是把期望搬出去后仍严格相等。
- 组内**共享 $t$ 和 $\epsilon$**。这是方差缩减的实现选择，不参与 $\log Z$ 的抵消证明；外层有 sigmoid 非线性，所以也别套"换 noise coupling 永远只改方差、原目标仍无偏"的旧说法。

### 4.5　和 Diffusion-DPO 的关系

$G = 2$ 时正负各一张、权重相等，就回到 Diffusion-DPO 的形式（尺度吸进 $\beta$）。一般的 DGPO 是"**组差进一个 sigmoid**"，不是枚举所有正负 pair 逐对求 loss 再平均。

### 4.6　为什么 ODE 可以用

训练时对存下来的干净图**重新构造** $q(x_t\mid x_0)$，从不计算采样路径的 likelihood。所以 collection 用什么 solver 都行——论文用 10 步 Flow-DPM-Solver。要存的只有干净图、prompt、reward。仍然需要固定的 $p_\text{ref}$；collector 可以是当前策略或 EMA。

### 4.7　Timestep clipping（必须知道）

少步生成的图有伪影（模糊等），低噪声 timestep 的回归会把这些伪影当特征学走。所以 DGPO 只在 $[t_\min, 1]$ 上采 $t$ 训练。**这是训练 timestep 的截断，不是 PPO 的 ratio clipping。**

### 4.8　配方与结果

论文默认：$G = 24$，10 步 Flow-DPM-Solver，$\beta = 100$（受 loss 尺度约定影响，不能和别人的 $\beta$ 比），前 200 步 collector 直接跟随 policy，之后 EMA decay 0.3。

SD3.5-M（DGPO 论文 Table 2，同表复现 Flow-GRPO）：

| 指标 | base | Flow-GRPO（DGPO Table 2 所列） | DGPO |
| --- | --- | --- | --- |
| GenEval | 0.63 | 0.95 | **0.97** |
| OCR | 0.59 | 0.92 | **0.96** |
| PickScore | 21.72 | 23.31 | **23.89** |
| UnifiedReward（偏好训练模型） | 3.33 | 3.66 | **3.74** |

"约 20×"是论文报告的整体训练效率，GenEval 曲线另称接近 30×——**不是**一次 backward 快 20×。

---

## §5 DiffusionNFT：在前向过程上做 RL

Zheng, Chen, Ye, Wang, Zhang, Jiang, Su, Ermon, Zhu, Liu 2025, *DiffusionNFT: Online Diffusion Reinforcement with Forward Process*, arXiv 2509.16117, ICLR 2026 Oral（清华 / NVIDIA / Stanford）。

### 5.1　出处：LLM 的 NFT

Chen et al. 2025, *Bridging Supervised Learning and Reinforcement Learning in Math Reasoning*（arXiv 2505.18116）：只在正例上做 SFT（RFT）会丢掉负例的信息；NFT 把 old policy 分解成正负两支，负例通过一个**隐式负策略**进入监督 loss。DiffusionNFT 沿用的是这个 negative-aware 思路——注意离散 token 概率的外推公式不能直接搬成 velocity 公式，下面是 diffusion 自己的推导。

### 5.2　"前向过程上的 RL"是什么意思

- **采集**：任何黑盒 solver（ODE/SDE、任意步数）出 $K$ 张图，打分，**只存干净图 $x_0$ 和 reward**。
- **训练**：重新采 $t, \epsilon$，构造 $x_t = (1-t)x_0 + t\epsilon$ 和 target $v = \epsilon - x_0$，做加权 flow-matching 回归。
- 不反传穿过 solver，不沿着采集轨迹训练，没有任何 likelihood。

### 5.3　Reward 归一化与正负分解

每张图的 reward 映射到 $[0,1]$：

$$r_i = \tfrac12 + \tfrac12\operatorname{clip}\Big(\frac{R_i - \bar R_c}{Z_c},\ -1,\ 1\Big)$$

$Z_c$ 是归一化尺度（论文允许用全局 reward std）；它**不是** DPO 那个 partition function。$r_i$ 解释为"optimality probability"——**是每个样本的量**，不是全局成功率。

建模上，记 $\bar r_c = \mathbb E_{\pi_\text{old}}[r]$，隐式正负策略：

$$\pi^+ = \frac{r\,\pi_\text{old}}{\bar r_c},\qquad \pi^- = \frac{(1-r)\,\pi_\text{old}}{1-\bar r_c}$$

实现不必真的把数据分成两堆，直接用软权重 $r$、$1-r$ 即可。

### 5.4　噪声态的分解与 improvement direction

在噪声态 $x_t$ 上，正例占比是后验

$$\alpha(x_t) = \mathbb E[r(x_0,c)\mid x_t, c] = \bar r_c\,\frac{\pi_t^+(x_t\mid c)}{\pi_t^\text{old}(x_t\mid c)}$$

（既不是单样本 $r_i$，也不是全局成功率，更不是加噪 schedule 的 $\alpha_t$。）真实正负分布对应的场满足

$$v_\text{old} = \alpha\,v^+ + (1-\alpha)\,v^-,\qquad \Delta := \alpha\,(v^+ - v_\text{old}) = (1-\alpha)(v_\text{old} - v^-)$$

$\Delta$ 是"reinforcement guidance"方向。这些是真实场；下面的是**参数化的两支**，记号分开。

### 5.5　Implicit mixing：负支为什么是 $(1+\beta)$

用一个网络 $v_\theta$ 同时参数化两支，关于 old **对称反射**：

$$v_\theta^+ = v_\text{old} + \beta\,(v_\theta - v_\text{old}),\qquad v_\theta^- = v_\text{old} - \beta\,(v_\theta - v_\text{old})$$

展开第二式就是论文写的 $(1+\beta)v_\text{old} - \beta v_\theta$。$\beta$ 是 mixing 参数。

### 5.6　Loss 与梯度（面试要能展开）

$$\mathcal L(\theta) = \mathbb E_{c,\,x_0\sim\pi_\text{old},\,t,\,\epsilon}\Big[r\,\lVert v_\theta^+ - v\rVert^2 + (1-r)\,\lVert v_\theta^- - v\rVert^2\Big]$$

两项都是**非负 MSE**——负反馈来自反射参数化，不是给普通 MSE 一个负权重。令 $d = v_\theta - v_\text{old}$，$e = v_\text{old} - v$，逐样本展开：

$$\ell = \lVert e\rVert^2 + \beta^2\lVert d\rVert^2 + 2\beta(2r-1)\,e^\top d$$

对 $d$ 求梯度：$2\beta^2 d + 2\beta(2r-1)e$。刚开始 $v_\theta = v_\text{old}$（$d=0$）时，梯度是 $2\beta(2r-1)(v_\text{old} - v)$：正例（$r=1$）沿 $-\nabla$ 走就是**朝 target $v$ 更新**，负例（$r=0$）反向，$r=\tfrac12$ 只剩对 old 的二次约束。

### 5.7　Theorem 3.2 与 CFG-free

逐样本最优 $d^* = (2r-1)(v - v_\text{old})/\beta$；取条件期望，用 $v_\text{old} = \mathbb E[v\mid x_t]$ 和 $\mathbb E[r\,v\mid x_t] = \alpha v^+$：

$$v_\theta^* = v_\text{old} + \frac{2}{\beta}\,\Delta$$

系数是 $2/\beta$：正负两支对同一个 improvement direction 各贡献一份。**$1/\beta$ 控制 guidance 强度**——$\beta$ 越小位移越大。定理依赖理想分布与容量假设，不保证任意 $\beta$、任意有限训练都提升 reward。

**CFG 的重新解释**：$v_\text{cfg} = v_\text{uncond} + w(v_\text{cond} - v_\text{uncond})$ 也是"沿一个方向外推"——cond 是正信号、uncond 是负信号，CFG 是**离线**的 reinforcement guidance。NFT 把这件事在线化了：从**只有条件分支**的模型出发，采集、训练、推理全程 CFG-free。论文 Table 1 的多 reward 模型训练约 1.7k 步后取得表中结果，超过带 CFG 的 baseline。

### 5.8　实践版 loss（不能只写 Eq. 5）

- **old 用 EMA 更新**：$\theta_\text{old} \leftarrow \eta_i\theta_\text{old} + (1-\eta_i)\theta$，在一个优化阶段内固定其输出。NFT 需要 old，但核心 loss **不需要**另存一个始终固定的初始 ref。
- **Adaptive weighting**：把 velocity 回归改写成带 stop-grad 归一化分母的 $x_0$ 回归，$\lVert x_\theta - x_0\rVert^2 / \operatorname{sg}\big(\operatorname{mean}\lvert x_\theta - x_0\rvert\big)$（受 DMD 蒸馏启发）。这是实践中的 loss 设计，不是定理逐字覆盖。
- **相对 RWR / RFT 的差异**在于显式利用负反馈：论文的去负支消融很快 collapse——这是该实验的观察，不是"所有 diffusion 正例微调必崩"的定理。

### 5.9　结果

- 单 reward：SD3.5-M GenEval **0.24 → 0.98**，约 1k 步。**0.24 是 CFG-free base**；Flow-GRPO 常引的 0.63 是 CFG base，两者不能直接并排。
- 效率：论文 §4.3 的口径是 **wall-clock time**——与 Flow-GRPO head-to-head（同为 10 步 rollout），GenEval 上最高约 **25×**，各 reward 在 3–25× 之间。这是训练曲线上达到同一 reward 所需时间的比较，不是单步吞吐、推理速度或梯度估计本身快 25×。
- 多 reward（GenEval / OCR / PickScore / ClipScore / HPSv2.1 分阶段联合，1.7k 步，采集 40 步）：GenEval **0.94**、OCR **0.91**——别拿单 reward 的 0.98 填这一行。在多项域内/域外指标上超过 SD3.5-L 和 FLUX.1-Dev，但**不是逐列全胜**（例如 ClipScore 0.293 低于 FLUX 的 0.295）。

---

## §6 三者对比

| 维度 | Flow-GRPO | DGPO | DiffusionNFT |
| --- | --- | --- | --- |
| 数学形式 | advantage 加权策略梯度 | 组偏好 logistic | reward 加权双支回归 |
| 采样器 | 训练 rollout 必须 SDE | 任意，默认 ODE（10 步 DPM） | 任意黑盒 |
| 存什么 | 整条轨迹 $O(PGTd)$ | 干净图 $O(PGd)$ | 干净图 $O(PGd)$ |
| 需要 likelihood | 每步 Gaussian | 否（DSM 差） | 否 |
| old | rollout 策略 | collector（EMA） | EMA 反射中心 |
| ref | KL 项 | DSM 差基准 | 核心 loss 不需要 |
| CFG | 原配置带 CFG，实现也支持 CFG-free | 论文未写；官方实现默认 rollout 带 CFG | 全程 CFG-free |
| 主要近似 | 离散化 + PPO surrogate | 路径 likelihood + DSM + Jensen | 无 likelihood 近似；仍有模型/采样/优化误差 |
| 报告效率 | 基准 | ~20× | 最高 ~25× |

存储比较不含权重、优化器和 activation；"多一个 ref"不等于显存翻倍。

---

## §7 Code Patterns

三段各 ~20 行的核心 loss，完整实验在 [`code/diffusion_online_rl.py`](code/diffusion_online_rl.py)。三段共用 `import math, torch; import torch.nn.functional as F`。

### 7.1　Flow-GRPO：SDE 一步 + log-prob + ratio

```python
def sde_step(v_theta, x, t, h, a=0.7, xi=None):
    """One reverse-time Euler-Maruyama step, t -> t - h, h > 0.  Returns (x_next, mu, v)."""
    g2 = a**2 * t / (1 - t)                                   # g(t)^2, t in (0,1)
    v = v_theta(x, t)
    eps_hat = x + (1 - t) * v                                 # §2.2
    mu = x - h * (v + g2 / (2 * t) * eps_hat)                 # drift, positive step
    xi = torch.randn_like(x) if xi is None else xi
    return mu + (g2 * h) ** 0.5 * xi, mu, v

def gauss_logp(x_next, mu, g2h):
    d = x_next.numel() // x_next.shape[0]
    return -((x_next - mu) ** 2).flatten(1).sum(1) / (2 * g2h) - 0.5 * d * math.log(2 * math.pi * g2h)

# per step: states x_k, x_{k+1}, old_logp, adv are FIXED from rollout; t_k, h are scalars
g = (0.7**2 * t_k / (1 - t_k)) ** 0.5                          # g(t_k), same a as sde_step
g2h = g**2 * h
_, mu, v_cur = sde_step(policy, x_k, t_k, h, xi=None)          # one policy call: mean for the ratio, v for the KL
logp = gauss_logp(x_k1, mu, g2h)
rho = torch.exp(logp - old_logp)
pg = -torch.min(rho * adv, rho.clamp(1 - eps_c, 1 + eps_c) * adv).mean()
with torch.no_grad():
    v_ref = ref(x_k, t_k)
kl = (h / 2) * (1 / g + g * (1 - t_k) / (2 * t_k)) ** 2 * ((v_cur - v_ref) ** 2).flatten(1).sum(1)
loss = pg + beta_kl * kl.mean()
```

### 7.2　DGPO：组内 DSM 差进一个 sigmoid

```python
def dgpo_loss(policy, ref, x0_group, c, adv, t, eps, beta, T, lam_t=1.0):
    """x0_group: [G, ...] clean samples of ONE prompt; adv: [G] centred advantages (sum ~ 0).
    Shared t, eps across the group (variance reduction, not needed for the log Z proof)."""
    x_t = (1 - t) * x0_group + t * eps
    v = eps - x0_group
    l_theta = ((policy(x_t, t, c) - v) ** 2).flatten(1).mean(1)            # per-sample DSM
    with torch.no_grad():
        l_ref = ((ref(x_t, t, c) - v) ** 2).flatten(1).mean(1)
    d = l_theta - l_ref                                                     # [G]
    # scale note: the paper's DSM is an x0-space error; this snippet uses per-dim mean
    # velocity error. With D latent dims, d_x0 = D * t**2 * d_here. To keep the paper's
    # objective set lam_t_here = D * t**2 * lam_t_paper; dropping the factor changes the
    # timestep weighting (a constant beta cannot absorb t**2) and needs its own tuning.
    w = adv.abs()                                                           # w = |A|; sum_{+} w == sum_{-} w
    logit = -lam_t * beta * T * (adv * d).sum()                             # sign carried by adv
    return -F.logsigmoid(logit)
```

### 7.3　DiffusionNFT：反射双支回归

```python
def nft_loss(policy, old, x0, c, r, t, eps, beta):
    """x0: [B, ...] clean samples; r: [B] in [0,1] (0.5 + 0.5*clip(normalised reward)).
    t is a shared scalar here; with per-sample t of shape [B], reshape it to [B,1,...] for
    the noising line and keep the [B] form for the network call."""
    x_t = (1 - t) * x0 + t * eps
    v = eps - x0
    with torch.no_grad():
        v_old = old(x_t, t, c)                                # EMA of policy, frozen inside a phase
    d = policy(x_t, t, c) - v_old
    v_pos = v_old + beta * d                                   # (1-beta) v_old + beta v_theta
    v_neg = v_old - beta * d                                   # (1+beta) v_old - beta v_theta
    per = r * ((v_pos - v) ** 2).flatten(1).mean(1) + (1 - r) * ((v_neg - v) ** 2).flatten(1).mean(1)
    return per.mean()
```

---

## §8 Online RL 的失败模式

- **Reward hacking 看三样**：原始 reward（不是组归一化分数——它不代表跨轮绝对进步）、图本身、多样性。KL、clipping、负支 loss 都**不保证**消除 hacking。
- **Flow-GRPO 的 ratio 漂移**（GRPO-Guard）：importance ratio 的分布随 timestep 偏移，固定 $\varepsilon_c$ 的 clip 在某些 $t$ 上失效；RatioNorm / 梯度重加权是优化层面的修法。
- **DGPO 的低噪声伪影**：忘了 timestep clipping，模型学会少步采样的模糊。
- **NFT 的 $\beta$**：它同时决定两件事——$d=0$ 处的梯度正比于 $\beta$，而理想最优位移正比于 $1/\beta$；较小的 $\beta$ 意味着更大的 guidance 目标。old 更新过快在论文实验里会不稳（反射本身在 old 冻结时仍给出非零梯度，失效的不是反射而是稳定性）。
- **"域外"的口径**：论文里的 OOD 通常指 DrawBench 上没参与训练的评估指标，不是分布外图像域。

---

## §9 vs LLM 的 GRPO / NFT

| | LLM | Diffusion |
| --- | --- | --- |
| 单步策略 | token 的 categorical，密度天然可算 | ODE 是 Dirac；要 SDE 化（Flow-GRPO）或绕开（DGPO/NFT） |
| 序列长度 | 数百到数千 token | 10–40 步，但每步是一整张 latent |
| NFT 的负策略 | 离散概率外推 | velocity 关于 old 的反射 |
| 参考模型 | GRPO 有 KL-ref | Flow-GRPO 有；NFT 核心 loss 没有 |
| CFG | 无对应物 | NFT 把 CFG 解释成离线 guidance 并在线化 |

共同教训：group baseline 去掉 critic 之后，剩下的难点全在"密度能不能算"——LLM 天然能算，diffusion 要么造一个（SDE），要么换一个不需要密度的目标。

---

## §10 25 高频面试题

### L1 必会题（10 题）

<details>

<summary>Q1. Flow-GRPO / DGPO / DiffusionNFT 真正的区别是什么？</summary>

三者共享 group sampling + 组内归一化的采集框架。区别在 reward 怎么进梯度：Flow-GRPO 加权每步转移的策略梯度；DGPO 把正负组的 reference-relative DSM 差放进一个 sigmoid（偏好学习）；NFT 用 reward 加权正负两支 flow-matching 回归。三者都没有 critic，但不是"三种 policy gradient"。

只答"都是 GRPO 变体"不得分。

</details>

<details>

<summary>Q2. ODE 采样已经有随机 seed，为什么 Flow-GRPO 还要改成 SDE？</summary>

seed 提供的是样本级随机性；PPO 需要的是**单步转移密度** $p_\theta(x_{k+1}\mid x_k)$ 来算 importance ratio。ODE 的单步是 Dirac，没有密度。SDE 化后 Euler 离散的单步变成 Gaussian，并顺便增加中间步骤的探索。

说"ODE 不能探索"是错的。

</details>

<details>

<summary>Q3. old 和 ref 能混用吗？</summary>

不能。old 是采样时的行为策略——在 Flow-GRPO 里是本轮 rollout 的策略快照（ratio 的分母），在 NFT 里是反射中心、用 EMA 更新；ref 是固定的约束基准——Flow-GRPO 的 KL、DGPO 的 DSM 差。NFT 的核心 loss 用的是更新中的 old，不需要固定初始 ref。

</details>

<details>

<summary>Q4. NFT 里的 $r$ 是成功率吗？</summary>

不是。$r_i = \tfrac12 + \tfrac12\operatorname{clip}((R_i - \bar R_c)/Z_c, -1, 1)$ 是**每个样本**映射到 $[0,1]$ 的 reward，解释为 optimality probability。全局的量是 $\bar r_c = \mathbb E_{\pi_\text{old}}[r]$，噪声态上的是后验 $\alpha(x_t)$。

</details>

<details>

<summary>Q5. 写出 RF 的 $\hat x_0$、$\hat\epsilon$、score 换算。</summary>

$x_t = (1-t)x_0 + t\epsilon$，$v = \epsilon - x_0$；用网络预测 $v_\theta$：$\hat x_0 = x_t - tv_\theta$，$\hat\epsilon = x_t + (1-t)v_\theta$，$s_\theta := -\hat\epsilon/t$。score 用预测的条件均值，只在 $v_\theta = v^*$ 时等于真实 score。

</details>

<details>

<summary>Q6. DGPO 的正负组怎么分？权重是什么？</summary>

同一 prompt 的组内 $A_i = (R_i - \bar R)/s_R$；$A_i > 0$ 进 $\mathcal G^+$，$A_i \le 0$ 进 $\mathcal G^-$；$w_i = \lvert A_i\rvert$。因为 $\sum A_i = 0$，两组权重总量相等。

</details>

<details>

<summary>Q7. Flow-GRPO 的 denoising reduction 是什么？</summary>

训练时 SDE 用 10 步采样，评估用 40 步。论文报告"超过 4×"加速。理由是经验性的：连续边缘与步数无关，RL 学的方向修正对步数不敏感——不是严格等价。

</details>

<details>

<summary>Q8. DiffusionNFT 训练需要存什么？</summary>

只存干净图 $x_0$、它的 prompt $c$ 和 reward。训练时重新采 $t,\epsilon$ 构造 $x_t$。不存轨迹、不算 likelihood、不反传穿过 solver。

</details>

<details>

<summary>Q9. 为什么 DiffusionNFT 推理可以不用 CFG？</summary>

CFG 是 cond（正）和 uncond（负）两个模型之间的外推——离线的 reinforcement guidance。NFT 用在线的正负样本学出一个能替代 CFG 作用的 reward guidance（方向不保证和 CFG 相同），模型本身就带 guidance。从只有条件分支的模型出发，全程 CFG-free。

</details>

<details>

<summary>Q10. GenEval 0.63、0.24、0.95、0.97、0.98 各是什么口径？</summary>

0.63：SD3.5-M **带 CFG** 的 base；0.24：**CFG-free** base；0.95：Flow-GRPO；0.97：DGPO；0.98：NFT 单 reward 约 1k 步。NFT 多 reward 模型是 0.94。

把两者当作相同 CFG 设置下的 baseline 来比不得分（标明 CFG 口径后当然可以并列展示，论文自己就是这么做的）。

</details>

### L2 进阶题（10 题）

<details>

<summary>Q11. 推导保边缘 SDE，并说明生成方向的符号。</summary>

正向时钟 $u = 1-t$：$dX_u = [-v + \tfrac12 g^2 s]du + g\,dW_u$。Fokker–Planck 中 $-\tfrac12 g^2\nabla\cdot(ps)$ 与 $+\tfrac12 g^2\Delta p$ 抵消（因 $ps = \nabla p$），剩 ODE 的密度演化。换回递减的 $t$：drift 是 $v - \tfrac12 g^2 s$。离散噪声尺度用 $\sqrt{h}$，$h = t_k - t_{k+1} > 0$。

写成 $+\tfrac12 g^2 s$ 是把时间方向弄反了。

</details>

<details>

<summary>Q12. 为什么 Flow-GRPO 的 KL 有闭式？写出来。</summary>

policy 与 ref 的单步转移是**同协方差** $g^2 h I$ 的 Gaussian，KL 只剩均值差：$\lVert\mu_\theta - \mu_\text{ref}\rVert^2/(2g^2h)$。代入 drift 得 $\frac h2\big(\frac1g + \frac{g(1-t)}{2t}\big)^2\lVert v_\theta - v_\text{ref}\rVert^2$。是条件单步 KL，不是终态分布的 KL。

</details>

<details>

<summary>Q13. DGPO 为什么能消掉 $\log Z(c)$？</summary>

隐式 reward $r_\theta = \beta\log(p_\theta/p_\text{ref}) + \beta\log Z(c)$。组级 reward 是加权和，正负组相减后 $\log Z$ 的系数是 $\sum_{\mathcal G^+}w - \sum_{\mathcal G^-}w$；因为 $w = \lvert A\rvert$ 且 $\sum A = 0$，这个差恰为零。前提是平衡权重总量，而不是按人数平均。

</details>

<details>

<summary>Q14. DGPO 和 Diffusion-DPO 是什么关系？</summary>

同一套 reference-relative DSM surrogate。$G=2$ 时 DGPO 退化为 Diffusion-DPO（尺度吸进 $\beta$）。一般 DGPO 把单 pair 换成带 $\lvert A\rvert$ 权重的两个子组，组差进一个 sigmoid，不是逐对枚举。

</details>

<details>

<summary>Q15. DGPO 的 timestep clipping 解决什么？和 PPO clip 有关吗？</summary>

无关。少步采样的图有伪影，低噪声 $t$ 的回归会把伪影学走，所以只在 $[t_\min, 1]$ 采 $t$。这是训练 timestep 的截断；PPO clip 是 importance ratio 的截断。

</details>

<details>

<summary>Q16. 写出 Flow-GRPO 单步的 Gaussian log-density，说明"对维度求和"和"按维度平均"的区别。</summary>

$\log p = -\lVert x_{k+1} - \mu_\theta\rVert^2/(2g^2h) - \tfrac d2\log(2\pi g^2h)$，对 $d$ 维求和才是密度。官方代码按维度平均 log-prob，得到的是 $\rho^{1/d}$；论文公式和实现约定要分开说。

</details>

<details>

<summary>Q17. DiffusionNFT 的 $v_\theta^+$、$v_\theta^-$ 怎么定义？</summary>

关于 old 对称反射：$v_\theta^\pm = v_\text{old} \pm \beta(v_\theta - v_\text{old})$。展开负支即 $(1+\beta)v_\text{old} - \beta v_\theta$。$\beta$ 是 mixing 参数，$1/\beta$ 控制 guidance 强度。

</details>

<details>

<summary>Q18. 展开 NFT 的逐样本 loss，说明 $v_\theta = v_\text{old}$ 时梯度方向。</summary>

$d = v_\theta - v_\text{old}$，$e = v_\text{old} - v$：$\ell = \lVert e\rVert^2 + \beta^2\lVert d\rVert^2 + 2\beta(2r-1)e^\top d$。$d = 0$ 时梯度 $2\beta(2r-1)(v_\text{old} - v)$：正例朝 target 走，负例反向，$r = \tfrac12$ 只剩对 old 的二次约束。

</details>

<details>

<summary>Q19. Flow-GRPO 训练时哪些量固定、哪些重算？</summary>

固定：rollout 的状态序列、old log-prob、reward、advantage。重算：当前策略在这些状态上的 log-prob。ref 输出 stop-grad。存轨迹存的是状态，不是反传图。

</details>

<details>

<summary>Q20. 三种方法的 CFG 依赖分别是什么？</summary>

三种目标都不把 CFG 当数学前提。Flow-GRPO 的原 SD3 配置带 CFG，后续实现也支持 CFG-free；DGPO 论文未写，官方实现默认 rollout 带 CFG、DSM 更新用条件预测；NFT 主配方采集、训练、推理全程 CFG-free。

</details>

### L3 顶级 lab 题（5 题）

<details>

<summary>Q21. 证明 NFT 的最优解是 $v_\text{old} + \frac2\beta\Delta$，解释"2"从哪来。</summary>

逐样本 $d^* = (2r-1)(v - v_\text{old})/\beta$。取条件期望：$\mathbb E[(2r-1)(v - v_\text{old})\mid x_t] = 2\mathbb E[r(v - v_\text{old})\mid x_t]$（因 $\mathbb E[v\mid x_t] = v_\text{old}$），而 $\mathbb E[rv\mid x_t] = \alpha v^+$、$\mathbb E[r\mid x_t] = \alpha$，故为 $2\alpha(v^+ - v_\text{old}) = 2\Delta$。"2"来自正负两支各贡献一份同方向的位移。

</details>

<details>

<summary>Q22. 25× 能证明 NFT 的梯度估计本身快 25× 吗？</summary>

不能。它是特定 reward（GenEval）上、同为 10 步 rollout 的 **wall-clock 训练曲线**比较，包含收敛速度、CFG 有无、采样与实现差异。其它 reward 在 3–25× 之间。多 reward 模型的数字另算。

</details>

<details>

<summary>Q23. DGPO 里为什么"按人数平均"会破坏抵消？给个例子。</summary>

4 张图 reward $[0,1,2,7]$，正负 1:3。$w = \lvert A\rvert$ 时两边总量都是 $S$，$\log Z$ 的系数 $S - S = 0$。两种破坏方式：(a) 保留 $\lvert A\rvert$ 但把每一边按人数取平均，系数变成 $S - S/3 \ne 0$——抵消没了；(b) 权重全改成 1 直接求和，系数是 $1 - 3 = -2 \ne 0$，给所有隐式 reward 加同一个常数就会改变 logit。`code/diffusion_online_rl.py` 实验 C 验证的是 (b)。

</details>

<details>

<summary>Q24. Flow-GRPO 的"保边缘"在有限步、有学习误差、带 CFG 时还成立吗？</summary>

不严格成立。保边缘是连续时间性质，要求 $v$ 与 $s$ 对应同一族边缘。学习误差使 $s_\theta$ 与 $v_\theta$ 不再匹配；CFG 后的 velocity 与代入的 score **不保证对应同一前向加噪边缘族**，保边缘证明不能直接继承；有限步离散化引入额外偏差。所以 denoising reduction 只能是经验观察。

</details>

<details>

<summary>Q25. 如果让你选一个方法上线，怎么选？</summary>

看三件事：采样器（有 SDE 成本预算才考虑 Flow-GRPO）、存储（轨迹 $O(PGTd)$ vs 干净图 $O(PGd)$）、是否要 CFG-free（NFT 天然满足）。DGPO 和 NFT 都用 ODE、只存干净图；DGPO 保留 ref 约束，NFT 用 EMA old + 反射。三者的报告效率都是特定 reward 曲线上的比较，自己的 reward 要自己跑。

</details>

---

## §A 附录

### A.1　论文

| 方法 | arXiv | 会议 | 代码 |
| --- | --- | --- | --- |
| Flow-GRPO | [2505.05470](https://arxiv.org/abs/2505.05470) | — | github.com/yifan123/flow_grpo |
| DGPO | [2510.08425](https://arxiv.org/abs/2510.08425) | ICLR 2026 | github.com/Luo-Yihong/DGPO |
| DiffusionNFT | [2509.16117](https://arxiv.org/abs/2509.16117) | ICLR 2026 Oral | github.com/NVlabs/DiffusionNFT |
| NFT (LLM) | [2505.18116](https://arxiv.org/abs/2505.18116) | — | — |
| GRPO | [2402.03300](https://arxiv.org/abs/2402.03300) | — | — |

### A.2　Runnable toy

脚本：[`code/diffusion_online_rl.py`](code/diffusion_online_rl.py)，纯 PyTorch，CPU 几秒。

- **A** 保边缘：1-D $x_0\sim\mathcal N(1, 0.25)$，解析 $v^*$、$s^*$；从 $t=0.9$ 走到 $0.1$，比较 ODE / 正确 SDE / 只加噪不修正三组的均值方差。
- **B** NFT 梯度：固定配对 $t=0.5, x_0=0, \epsilon=1$，$\beta=0.5$；断言 $r=1,0,\tfrac12$ 的初始梯度为 $-1,+1,0$，正负例最优解 $\pm2$。
- **C** DGPO 抵消：4 样本组 $[0,1,2,7]$；断言正负权重总量相等、reward 整体平移不变、隐式 score 整体平移不变；单位权重的反例。

### A.3　工程踩坑

| 症状 | 原因 | 处理 |
| --- | --- | --- |
| Flow-GRPO 从 $t=1$ 起步 NaN | $g(t) = a\sqrt{t/(1-t)}$ 分母为零 | 首步用相邻时间点 |
| Flow-GRPO clip 在某些 $t$ 失效 | ratio 分布随 timestep 漂移 | RatioNorm / 梯度重加权（GRPO-Guard） |
| DGPO 图变糊 | 低噪声 $t$ 学走了少步伪影 | timestep clipping $[t_\min, 1]$ |
| DGPO loss 不动 | 组内 reward 全同，标准差为零，advantage 按约定取零 | 换 prompt 或加大 $G$ |
| NFT 不稳 | $\beta$ 太小或 old EMA 追得太紧 | 调 $\beta$、放慢 EMA |
| 三者都「涨分不涨质」 | reward 和视觉质量脱钩（原始 reward 也会被 hack）；组归一化分数另有「不反映跨轮进步」的问题 | 看原始 reward + 图 + 多样性 |
