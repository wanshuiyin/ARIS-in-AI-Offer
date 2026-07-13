## §0 TL;DR Cheat Sheet

> 💡 **8 sentences to nail Image Generation systems** — one page covering the core of a production text-to-image stack (see §1–§10 for derivations).

1. **LDM essentials**: VAE encode compresses $H\times W\times 3$ to $h\times w\times c$ (SD 1.x: $8\times$ downsample, $c=4$); diffusion runs in latent space, dropping token count to $1/8^2=1/64$ (an **order-of-magnitude estimate** assuming constant per-token compute; the actual FLOPs saving also depends on latent-space channels being wider and on self-attention FLOPs scaling quadratically with token count, so it is not an exact $64\times$ equivalence), then VAE decode reconstructs pixels (Rombach et al. 2022 CVPR).

2. **SD 1.x → SDXL → SD3 → FLUX lineage**: 1.x uses CLIP-L text encoder + U-Net; SDXL has dual encoders (OpenCLIP-G + CLIP-L) + 2.6B U-Net + size/crop conditioning + Refiner (Podell et al. 2024 ICLR); SD3 switches to **MM-DiT** + Rectified Flow (Esser et al. 2024 ICML); FLUX.1 is a 12B MM-DiT with parallel attention (Black Forest Labs 2024).

3. **CFG (must-know)**: at training, with probability $p_\text{drop}\approx 0.1$ replace condition $c$ with $\emptyset$; at inference, output $\hat\epsilon_\text{cfg} = \hat\epsilon_\emptyset + s\,(\hat\epsilon_c - \hat\epsilon_\emptyset)$, $s \in [1.5, 12]$ (Ho & Salimans 2022).

4. **ControlNet zero-conv** (Zhang et al. 2023 ICCV): deep-copy the entire U-Net encoder as a **trainable copy**; the $1\times 1$ conv connecting each branch to the backbone is initialized $W=0,b=0$ — forward pass is identity but **gradient is nonzero** ($\partial L/\partial W = \delta \cdot x \neq 0$). Training starts from a clean identity map and progressively injects condition signal, preserving pretrained capability.

5. **IP-Adapter** (Ye et al. 2023): **Decoupled Cross-Attention** — for image conditioning, **add a new set** of $W_K', W_V'$ in parallel with the text cross-attn; outputs are summed: $\text{out} = \text{Attn}(Q, K_\text{txt}, V_\text{txt}) + \lambda\,\text{Attn}(Q, K_\text{img}, V_\text{img})$. Only the new K/V + projector are trained, ~22M parameters.

6. **LoRA** (Hu et al. 2022 ICLR): $\Delta W = B A,\ B \in \mathbb{R}^{d\times r},\ A \in \mathbb{R}^{r\times k},\ r \ll \min(d,k)$; only $A, B$ are trained, $W$ is frozen. On SD, typical $r \in \{4,8,16,32\}$, 50–200× smaller than full fine-tune; at inference merge $W' = W + \alpha B A$.

7. **DreamBooth** (Ruiz et al. 2023 CVPR): rare-token (e.g., `sks dog`) + **prior preservation loss** $L = \|\epsilon - \hat\epsilon(x_t, t, \text{"a sks dog"})\|^2 + \lambda \|\epsilon' - \hat\epsilon(x_t', t, \text{"a dog"})\|^2$; the second term prevents language drift / overfitting.

8. **DiT vs MM-DiT**: DiT uses **AdaLN-Zero** (condition $\to$ MLP $\to$ scale/shift/gate, last layer's $W_\text{gate}=0$ for identity warm start, Peebles & Xie 2023 ICCV); MM-DiT **concatenates text and image tokens into a single sequence for joint self-attention** (each modality has independent QKV projections, but attention is global), enabling bidirectional information flow (Esser et al. 2024).

## §1 Intuition and Big Picture

Why LDM? **Pixel-space diffusion is too expensive**: in the SD era, a 1024² image = 3M pixels, and a U-Net forward at full resolution per timestep means single-card training is restricted to toy 64×64 sizes. LDM's idea: use a pretrained **VQ-VAE / KL-VAE** to compress images to latents (SD 1.x: $512\times 512\times 3 \to 64\times 64\times 4$, $64\times$ fewer tokens), run diffusion only in latent space, then VAE decode back to pixels — decoupling "semantics / structure / texture":

```
    pixels x        latent z (perceptually similar)        latent z_T (noise)
   [H,W,3]    →     [h,w,c]     →     diffuse     →     [h,w,c]
                       ▲                                    │
                       │  VAE decoder                       │  reverse SDE / ODE
                       │                                    ▼
                    pixels x̂          ←      latent z_0      ←    [h,w,c]
```

The production stack decomposes into 5 **orthogonal, swappable** modules:

| Module | Role | SD 1.x | SDXL | SD3 / FLUX |
|---|---|---|---|---|
| **VAE** | pixel ↔ latent | KL-VAE ($f=8$, $c=4$) | KL-VAE ($f=8$, $c=4$) | $f=8$, $c=16$ (wider latent) |
| **Text encoder** | text → token embedding | CLIP-L | OpenCLIP-G + CLIP-L | CLIP-L + OpenCLIP-G + **T5-XXL** (SD3) / T5-XXL + CLIP-L (FLUX) |
| **Denoiser** | $\epsilon$ / $v$ / $u$ prediction | U-Net 860M | U-Net 2.6B | MM-DiT 2B / 8B / 12B |
| **Objective** | training target | $\epsilon$-pred (DDPM) | $\epsilon$-pred | **Rectified Flow** (v-pred family) |
| **Sampler** | reverse process | DDIM / PLMS / DPM++ | DDIM / DPM++ | Euler / Heun (RF ODE) |

> 💡 **Three flavors of conditioning** — proactively disambiguate which "conditioning" you mean in interviews.

- **Semantic conditioning (text)**: text encoder embedding → cross-attention K/V; primarily drives "what to draw"

- **Structural conditioning (edges / depth / pose)**: ControlNet / T2I-Adapter; drives "what structure to follow"

- **Identity / style conditioning (face / style)**: IP-Adapter / InstantID / PuLID / DreamBooth / LoRA; drives "whose look / whose style"

## §2 LDM Core: VAE Compression + Latent Diffusion

### 2.1　LDM loss (Rombach et al. 2022 CVPR)

A pretrained KL-VAE gives encoder $\mathcal{E}: \mathbb{R}^{H\times W\times 3} \to \mathbb{R}^{h\times w\times c}$ and decoder $\mathcal{D}$ satisfying $\mathcal{D}(\mathcal{E}(x)) \approx x$ (perceptual reconstruction), with $h = H/f$ and downsample factor $f \in \{4, 8, 16\}$; the SD family uses $f=8$.

Training diffusion in latent space, the objective matches pixel-space DDPM:

$$\boxed{\;\mathcal{L}_\text{LDM} = \mathbb{E}_{z_0, \epsilon, t, c}\left[\,\big\|\epsilon - \epsilon_\theta\!\left(z_t,\, t,\, \tau_\theta(c)\right)\big\|^2\,\right]\;}$$

with $z_0 = \mathcal{E}(x)$, $z_t = \sqrt{\bar\alpha_t}\, z_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$, and $\tau_\theta(c)$ the text encoder output.

### 2.2　Why $f=8$ is the sweet spot

Rombach 2022 Table 8 ablation:

| Downsample $f$ | Token count reduction ($f^2$) | Reconstruction quality (FID↓) | Generation quality (FID↓) |
|---|---|---|---|
| $f=4$ | $16\times$ | Best (latent close to pixel) | Mediocre (diffusion still expensive) |
| $f=8$ | $64\times$ | Slight loss (small PSNR drop) | **Best** |
| $f=16$ | $256\times$ | Notable degradation (VAE reconstruction worsens) | Reconstruction bottleneck drags generation |
| $f=32$ | $1024\times$ | VAE essentially cannot reconstruct detail | Severe drop in generation quality |

> ⚠️ **Token-count reduction ≠ exact proportional FLOPs reduction** — the $f^2$ figures above are exact token-count ratios, but the actual compute saving also depends on: (i) latent-space networks typically have much wider channels than pixel-space's 3 channels, which offsets some of the token-count saving; (ii) self-attention FLOPs scale **quadratically** with token count, so attention-layer savings can exceed $f^2$ by a lot (e.g. at $f=8$, theoretically up to $\sim f^4=4096\times$). So these numbers should be read as order-of-magnitude savings, not an exact FLOPs equivalence.

**Key insight**: VAE reconstruction quality is an upper bound — no matter how strong the latent diffusion model is, it cannot produce images the VAE cannot decode. So larger $f$ is not always better; balance "compression ratio" vs "reconstruction ceiling".

### 2.3　VAE's "small KL" detail

The SD VAE is a **KL-VAE, not a VQ-VAE** — its latent is a **continuous Gaussian** with a tiny KL term (~$10^{-6}$ magnitude), effectively an AE with mild regularization. Rombach 2022 Appendix explains: too strong a KL would collapse the latent to a pure Gaussian, losing structural information.

> ⚠️ **SD VAE scaling factor** — using $\mathcal{E}(x)$ directly for diffusion training, the raw latent's standard deviation is far from 1 (SD 1.x raw std ≈ 5.5). SD's fix is to multiply the latent by scalar `0.18215` ("scaling factor", ≈ $1/5.5$) so std is close to 1. SDXL/SD3 recalibrate this constant (SDXL `0.13025`; SD3 `scaling_factor=1.5305` + `shift_factor=0.0609`; diffusers' SD3 pipeline computes `z = (z_raw - shift) * scaling`). **Mismatch causes over/under noising** — this is a classic source of pipeline bugs.

## §3 SD 1.x → SDXL → SD3 → FLUX Lineage

### 3.1　SD 1.x / 2.x (Stability AI 2022)

- **U-Net 860M parameters**: cross-attention is placed at three downsample levels of latent resolution 64 / 32 / 16 + corresponding upsample blocks + middle block (SD v1 config `attention_resolutions = [4, 2, 1]`, DS=1/2/4 i.e. 64/32/16; the deepest DS=8 i.e. 8×8 downsample/upsample block only has ResBlocks without transformer, but the middle block at 8×8 does have transformer); text comes from CLIP-L (`openai/clip-vit-large-patch14`, 768-d), providing 77 token embeddings

- **Objective**: $\epsilon$-prediction (DDPM Ho et al. 2020 NeurIPS)

- **Training resolution**: 512², 1B+ LAION images

- 2.x switches to OpenCLIP-H/14 (stronger and cleaner license) and fine-tunes at 768²

### 3.2　SDXL (Podell et al. 2024 ICLR)

Three key upgrades:

| Change | Detail |
|---|---|
| **2.6B U-Net** | Wider and deeper; 3× the 1.x parameters |
| **Dual text encoders** | OpenCLIP-G (1280-d) + CLIP-L (768-d); concat then do cross-attn |
| **Size & crop conditioning** | At training, $(h_\text{orig}, w_\text{orig})$, $(h_\text{crop}, w_\text{crop})$, and **target size** $(h_\text{tgt}, w_\text{tgt})$ (used for multi-aspect-ratio training) — 3 pairs, 6 scalars total — are each Fourier-embedded and **added to the timestep embedding**, so the model explicitly knows "this image's original size, its crop region, and the target resolution", preventing low-resolution / crop artifacts from leaking into inference |
| **Refiner** | A separate latent diffusion model dedicated to the last ~20% of the noise level ($t < 0.2$) for detail refinement; optional |
| **Training resolution** | 1024² (final), with bucketing across aspect ratios |

> ✅ **Size conditioning training effect** — SDXL Table 1: without size conditioning, the model sees 512² LAION images and treats "low-res feel" as a data prior, producing blurry outputs at 1024² inference; with size conditioning, filling `(1024, 1024)` at inference tells the model "I want 1024 quality", **significantly reducing blur / blocky artifacts**. Crop conditioning analogously fixes LAION's center-crop bias.

### 3.3　SD3 (Esser et al. 2024 ICML)

Two core changes:

**1) Training objective switches to Rectified Flow (RF)**

$$x_t = (1-t)\, x_0 + t\, x_1,\quad u_t = x_1 - x_0$$

where $x_0$ is the data end ($t=0$), $x_1 \sim \mathcal{N}(0, I)$ is the noise end ($t=1$). The model learns $v_\theta(x_t, t, c) \approx x_1 - x_0$. Loss uses logit-normal $t$ sampling (middle $t$ has higher density) + RF weighting. **The timestep convention runs in the same direction as DDPM** ($t=0$ data, increasing $t$ → more noise — not opposite); but exact variable naming / schedule details can differ across code bases (e.g., diffusers) — **proactively disambiguate in interviews**.

**2) Denoiser switches to MM-DiT (Multimodal Diffusion Transformer)**

Text and image tokens are **concatenated into a single sequence** for joint self-attention; each modality has **independent** QKV projection + AdaLN-Zero MLP parameters, but the attention matrix is global (bidirectional). This means:

```
[txt tokens, img tokens]  ──╮
       │                     ├─ joint self-attention  ──→  bidirectional flow
       │                     │                            (txt sees img, img sees txt)
       │                     │
   independent QKV (txt, img)│
   independent LN/MLP gate (txt, img)
```

Compared to SD 1.x/SDXL **cross-attention**: image queries unidirectionally read text K/V, **text is never updated**. MM-DiT lets text be updated by image (opening the "image → text" flow); empirically text alignment improves noticeably.

### 3.4　FLUX.1 (Black Forest Labs 2024)

12B-parameter MM-DiT v2, with the main differences:

| Dimension | FLUX.1 |
|---|---|
| Params | dev: 12B; schnell: same 12B but distilled |
| Architecture | MM-DiT + **parallel attention block** (attn and MLP run in parallel rather than sequentially, like PaLM / GPT-J) |
| Text encoding | T5-XXL (4096-d) + CLIP-L |
| Training objective | Rectified Flow (same family as SD3) |
| Sampling | dev: ~28-50 steps; schnell: 1-4 steps (adversarial diffusion distillation) |
| Position encoding | Multi-axis RoPE (encodes relative position by 2D coordinates for image tokens); text-token RoPE position ids are typically set to all zeros (`txt_ids` all 0, no explicit positional encoding) — ordering is instead carried implicitly by the T5 encoder's own relative-position mechanism |

> 💡 **Parallel attention** — Standard transformer block: `y = x + Attn(LN(x)); y = y + MLP(LN(y))`. Parallel block: `y = x + Attn(LN(x)) + MLP(LN(x))`, the two branches computed in parallel and summed. **Benefit**: on GPU, attn and MLP kernels can be overlapped, and weight fusion is cleaner; a slight expressivity loss is typically compensated by scale.

### 3.5　Parallel open-source lineage

- **PixArt-α / Σ** (Chen et al. 2024): DiT-XL/2 + T5 text, emphasizing "training cost only 12% of SDXL" — small but capable.

- **Hunyuan-DiT** (Tencent 2024 arXiv 2405.08748): Chinese-friendly bilingual DiT, 1.5B parameters, CLIP + mT5 dual encoders.

- **DiT** (Peebles & Xie 2023 ICCV): replaces the U-Net denoiser with a ViT-style transformer, class token + AdaLN-Zero conditioning; scaling laws smoother than U-Net — ancestor of SD3/FLUX.

- **U-ViT** (Bao et al. 2023): U-Net-style backbone but pure transformer blocks + long skip connections, an early transformer-based diffusion exploration.

- **Imagen** (Saharia et al. 2022 NeurIPS): Google's **pixel-space** (not latent) cascade — $64\times 64$ base + $256\times 256$ super-res + $1024\times 1024$ super-res; text uses a large T5-XXL, and results show **text encoder scale > U-Net scale** matters more for text alignment.

## §4 DiT Architecture and AdaLN-Zero (Must-Know)

### 4.1　DiT block (Peebles & Xie 2023 ICCV)

DiT adapts the ViT block for diffusion: each block is conditioned by $c = \text{embed}(t) + \text{embed}(\text{class})$.

```
Input tokens x_l (shape [B, N, D]),  condition c (shape [B, D])
                │
        ┌───────┴────────┐
        │                │
   MLP(c) → (α₁, β₁, γ₁) │  scale / shift / gate parameters
        │                │
        ▼                │
   LayerNorm(x_l)        │
        │                │
   scale·γ₁ + shift·β₁   │  ← AdaLN: normalize then conditioned affine
        │                │
   Multi-Head Attention  │
        │                │
   × α₁ (gate, 0-init)   │  ← gate × residual; α₁ starts at 0
        │                │
   +  x_l                │  residual
        │                │
        ▼                │
   ┌────────────┐        │
   │ second half│        │
   │ (LN + MLP) │       same (α₂, β₂, γ₂) ← MLP(c)
   └────────────┘
        │
        ▼
   Output x_{l+1}
```

### 4.2　AdaLN-Zero derivation ("why is the gate initialized to 0")

DiT's actual form (Peebles & Xie 2023, Eqn. (5)–(6)): condition $c$ goes through a single MLP that produces $(\beta_1, \gamma_1, \alpha_1, \beta_2, \gamma_2, \alpha_2)$, and the normalization uses **(1 + gamma)** rather than `gamma` directly:

$$\text{AdaLN}(x, c) = \big(1 + \gamma(c)\big) \odot \text{LN}(x) + \beta(c)$$

**AdaLN-Zero** initializes the final weight + bias of the MLP that produces $(\beta, \gamma, \alpha)$ **to 0**:

$$\text{Block}(x, c) = x + \alpha(c) \cdot f\!\left(\big(1 + \gamma(c)\big) \odot \text{LN}(x) + \beta(c)\right)$$

At training step 0: MLP is all-zero → $\gamma = 0, \beta = 0, \alpha = 0$ → AdaLN degenerates to $\text{LN}(x)$, gate $\alpha = 0$ → block output $= x$ (identity). **Note it's `1 + gamma`, so gamma=0 doesn't zero out the normalization path; the normalized branch just equals LN(x).**

> ✅ **Key property**: when $\alpha = 0$ the block is identity, but **gradients are nonzero**. Chain rule:

$$\frac{\partial L}{\partial \alpha} = \frac{\partial L}{\partial \text{out}} \cdot f\!\left((1+\gamma)\odot\text{LN}(x) + \beta\right)$$

at step 0, $\gamma = \beta = 0$, and $f((1+0)\cdot\text{LN}(x) + 0) = f(\text{LN}(x))$ is **nonzero** (LN(x) is generally nonzero, and attention/MLP do not map arbitrary inputs to 0); thus $\partial L/\partial \alpha \neq 0$, and chain-ruling back to $W^{\text{last}}_\text{MLP}$ yields nonzero gradient — $\alpha$ grows from 0 and the block gradually forks a non-trivial transformation from the identity map, keeping training stable.

Note that $\gamma, \beta$ themselves have zero gradient at step 0 (their downstream is gated by $\alpha = 0$: $\partial L/\partial \gamma = (\partial L/\partial \alpha\cdot$ ...) — this path must pass through $\alpha$, and when $\alpha = 0$, $\partial \text{Block}/\partial \gamma$ contains an $\alpha\cdot f'(\cdot)$ factor equal to 0). But once $\alpha$ has grown, $\gamma, \beta$ immediately receive nonzero gradients — so "$\alpha$ grows first, then $\gamma, \beta$ follow" is the two-stage dynamics of AdaLN-Zero.

Compare naive initialization (standard random $\alpha \neq 0$): early blocks already produce large-variance outputs; stacked over 24-32 layers, activations explode and training diverges. AdaLN-Zero is the key design enabling DiT to scale.

### 4.3　Time embedding

$$\text{TimeEmbed}(t) = \text{MLP}\!\left(\text{SinusoidalEmb}(t)\right),\quad \text{SinusoidalEmb}(t)_{2i} = \sin\!\left(t / 10000^{2i/D}\right)$$

Even/odd positions use sin / cos, similar to Transformer positional encoding. $t$ is discrete timestep $\in \{0, 1, ..., T-1\}$ in SD; continuous $\in [0, 1]$ in RF / FM.

## §5 SD Inference Loop + CFG (Core Code)

### 5.1　CFG formula

At training, with probability $p_\text{drop} \approx 0.1$, replace $c$ with null (empty text embedding or zero embedding) so a single network learns both conditional and unconditional branches. At inference:

$$\boxed{\;\hat\epsilon_\text{cfg}(z_t, t, c) = \hat\epsilon_\theta(z_t, t, \emptyset) + s\cdot\left[\hat\epsilon_\theta(z_t, t, c) - \hat\epsilon_\theta(z_t, t, \emptyset)\right]\;}$$

$s$ is the guidance scale; SD 1.x typically $s \in [5, 12]$; SDXL $s \approx 5$-$7$; FLUX dev $\approx 3.5$ (smaller, since RF models are more CFG-sensitive).

**Under v-prediction / RF** the form is identical, replacing $\hat\epsilon$ with $\hat v$.

### 5.2　SD inference loop (core 40 lines)

```python
import torch

@torch.no_grad()
def sd_sample(unet, vae, text_encoder, tokenizer, scheduler,
              prompt, neg_prompt="", num_steps=30, cfg_scale=7.0,
              height=512, width=512, device="cuda", dtype=torch.float16):
    # 1) text encoding: forward both prompt and negative prompt
    ids_pos = tokenizer(prompt, padding="max_length", max_length=77,
                        truncation=True, return_tensors="pt").input_ids.to(device)
    ids_neg = tokenizer(neg_prompt, padding="max_length", max_length=77,
                        truncation=True, return_tensors="pt").input_ids.to(device)
    emb_pos = text_encoder(ids_pos)[0]                  # [1, 77, D_text]
    emb_neg = text_encoder(ids_neg)[0]
    emb = torch.cat([emb_neg, emb_pos], dim=0)          # [2, 77, D_text]  (uncond, cond)

    # 2) latent initialization: pure noise [1, 4, h/8, w/8]
    lat_shape = (1, 4, height // 8, width // 8)
    z = torch.randn(lat_shape, device=device, dtype=dtype) * scheduler.init_noise_sigma

    # 3) scheduler sets timesteps
    scheduler.set_timesteps(num_steps, device=device)

    # 4) main loop
    for t in scheduler.timesteps:
        # batch trick: pair the two latents with (uncond, cond) and forward once,
        # saves one kernel launch and is more cuDNN-batch-friendly than two sequential forwards
        z_in = torch.cat([z, z], dim=0)                  # [2, 4, h, w]
        z_in = scheduler.scale_model_input(z_in, t)      # some samplers need sigma scaling

        eps = unet(z_in, t, encoder_hidden_states=emb).sample   # [2, 4, h, w]
        eps_neg, eps_pos = eps.chunk(2, dim=0)

        # 5) CFG combine
        eps_cfg = eps_neg + cfg_scale * (eps_pos - eps_neg)

        # 6) scheduler step: invert eps to z_{t-1}
        z = scheduler.step(eps_cfg, t, z).prev_sample

    # 7) VAE decode + denormalize back to pixels [0, 1]
    z = z / 0.18215                                       # SD 1.x scaling factor
    x = vae.decode(z).sample                              # [1, 3, H, W] in [-1, 1]
    x = ((x.clamp(-1, 1) + 1) / 2)                        # → [0, 1]
    return x
```

> ⚠️ **Don't forget to merge the CFG double forward** — beginners often run two unet forwards, doubling time; the correct approach is `torch.cat([z, z])` + `torch.cat([emb_neg, emb_pos])` in a single forward. **Going further**: CFG-distilled / Guidance-distilled models (e.g., SDXL-Turbo, FLUX schnell) don't even need the double forward.

> ⚠️ **Scaling factor must align** — SD 1.x: `0.18215`, SDXL: `0.13025`, SD3: scalar `scaling_factor=1.5305` + `shift_factor=0.0609` (diffusers' SD3 pipeline: `z = (z_raw - shift) * scaling`). Mismatches cause discolored / high-frequency-artifact outputs.

## §6 ControlNet and IP-Adapter: Conditioning Extensions

### 6.1　ControlNet architecture (Zhang et al. 2023 ICCV)

**Problem**: injecting edge / depth / pose and other structural conditions into a pretrained SD is too costly from scratch, and full fine-tuning destroys text-to-image capability.

**Solution**:

```
    Input latent z_t  ──┬───────────────────►  Original SD U-Net Encoder (frozen)
                        │                              │
                        │    Condition image c_img     │
                        │           │                  │
                        │           ▼                  │
                        │    Hint Encoder (conv stack) │
                        │           │                  │
                        │           ▼                  │
                        └────► Trainable Copy ←────────┤  encoder deep-copied,
                                    │                  │  starts training
                                    │                  │
                              Zero Conv (W=0, b=0)     │
                                    │                  │
                                    ▼                  │
                              add to skip connection ──┘  ─────►  Decoder (frozen)
```

- **Trainable copy**: full deep copy of SD U-Net's **encoder + middle block** as a sibling branch, initial weights = original SD encoder weights

- **Zero-conv**: each $1\times 1$ conv connecting back to the main skip path has **weight and bias both initialized to 0**

- **Hint encoder**: a 4-layer conv projecting the condition image (1 or 3 channels) into latent shape

- **At training**: original SD encoder/decoder are frozen; only trainable copy + zero-conv + hint encoder are trained

### 6.2　Zero-conv gradient derivation (L3 must-ask)

Zero-conv layer: $y = W \star x + b$, $W = 0, b = 0$, so $y = 0$; adding to the main skip equals "adding nothing", so forward is identity.

**Backward splits into two paths**:

**(a) zero-conv's own weights**:

$$\frac{\partial L}{\partial W_{ij}} = \frac{\partial L}{\partial y_i} \cdot x_j$$

$x_j$ (the zero-conv input, from the trainable copy output) is nonzero, $\partial L / \partial y_i$ is nonzero, so the **gradient is nonzero** — $W$ updates away from 0.

**(b) trainable copy's parameters $\theta_c$**: must pass through the $x \to y$ path; the chain rule's critical factor is $\partial y / \partial x = W$. At step 0 $W = 0$, **so the trainable copy's own parameter gradient is 0 in the first step**.

**Hence convergence has two stages**:

1. Step 0: $W = 0$ → ControlNet has zero effect on the backbone → output = original SD output → **cannot be worse than baseline**

2. Step 1: zero-conv breaks zero on its own (path (a)) → $W \neq 0$ → path (b) unlocks

3. Step 2+: trainable copy starts receiving gradients and learning

4. End of training: $W$ and the trainable copy together reach appropriate magnitudes; structural conditioning is integrated

> ✅ **Why this is elegant** — naive random initialization of the trainable copy lets an "undertrained copy" contaminate backbone signal early, destroying SD capability (catastrophic forgetting). The zero-conv guarantees a **clean warm start** — gradient flows immediately through zero-conv's own path (one step to break zero), then the trainable copy follows; **decouple first, integrate later**.

### 6.3　Hint encoding + zero-conv (core 50 lines)

```python
import torch
import torch.nn as nn

def zero_module(m: nn.Module) -> nn.Module:
    """ Zero out all parameters of a module (used for ControlNet zero-conv and IP-Adapter projector). """
    for p in m.parameters():
        nn.init.zeros_(p)
    return m

class HintEncoder(nn.Module):
    """ Encode the condition image (e.g. canny edge, depth) down to latent resolution. """
    def __init__(self, in_ch=3, out_ch=320):  # 320 = SD U-Net first hidden dim
        super().__init__()
        # progressively downsample to 1/8 (same factor as VAE); last layer is a zero-conv
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1),       nn.SiLU(),
            nn.Conv2d(16, 16, 3, padding=1),          nn.SiLU(),
            nn.Conv2d(16, 32, 3, padding=1, stride=2), nn.SiLU(),   # /2
            nn.Conv2d(32, 32, 3, padding=1),          nn.SiLU(),
            nn.Conv2d(32, 96, 3, padding=1, stride=2), nn.SiLU(),   # /4
            nn.Conv2d(96, 96, 3, padding=1),          nn.SiLU(),
            nn.Conv2d(96, 256, 3, padding=1, stride=2), nn.SiLU(),  # /8
            zero_module(nn.Conv2d(256, out_ch, 3, padding=1)),
        )

    def forward(self, hint):     # hint: [B, 3, H, W]
        return self.net(hint)    # [B, out_ch, H/8, W/8]

class ControlNetBlock(nn.Module):
    """ trainable copy output → zero-conv → add to backbone skip """
    def __init__(self, ch):
        super().__init__()
        # this is the "output zero-conv" connecting to the backbone skip
        self.zero_conv = zero_module(nn.Conv2d(ch, ch, 1))   # 1×1, init 0

    def forward(self, x_copy, x_main_skip):
        # x_copy: current layer's output from the trainable copy
        # x_main_skip: original SD U-Net's skip activation at the same level
        return x_main_skip + self.zero_conv(x_copy)
```

> ⚠️ **Common misconception** — zero-conv is not dropout, not LoRA, not BatchNorm. It's an **initialization strategy**: weight=0 simultaneously achieves "identity forward + nonzero gradient".

### 6.4　T2I-Adapter (Mou et al. 2024) comparison

ControlNet's trainable copy is heavy (~half of SD's parameters). T2I-Adapter's idea is a **pure adapter**:

| Dimension | ControlNet | T2I-Adapter |
|---|---|---|
| Backbone intervention | Copy entire encoder | 4 lightweight conv blocks feeding skip directly |
| Parameters | ~360M (SD 1.5) | ~77M |
| Quality | Higher (strong structure follow) | Slightly weaker (but adequate) |
| Inference speed | Slow (dual encoders) | Nearly free |

### 6.5　IP-Adapter (Ye et al. 2023)

**IP-Adapter uses a reference image for conditioning, preserving identity / style across generations**. The core is **Decoupled Cross-Attention**:

```
    image  ──► CLIP image encoder ──► [N_img, D_clip]
                                            │
                                            ▼
                                    Projector (Linear, ~22M params)
                                            │
                                            ▼
                                    image embeddings [N_img, D_text]
                                            │
                                            │  used in parallel with text embedding
                                            │
    text   ──► text encoder ──► [N_txt, D_text]
                │                                                          ▲
                │                                                          │
                ▼                                                          │
   ┌───────────────────────────────────────────────────────────────┐
   │   At each U-Net cross-attention layer:                         │
   │                                                                │
   │   Q = z W_Q                                                    │
   │                                                                │
   │   Original text path:   K_txt = c_txt W_K^txt,  V_txt = c_txt W_V^txt   │
   │   New image path:       K_img = c_img W_K^img, V_img = c_img W_V^img   │
   │                                                                │
   │   out = Attn(Q, K_txt, V_txt) + λ · Attn(Q, K_img, V_img)      │
   │                                                                │
   │   Only W_K^img, W_V^img, and projector are trained             │
   └───────────────────────────────────────────────────────────────┘
```

Total parameters ~22M (projector ~10M + new K/V projections per layer ~12M total), ~1% of full SD.

### 6.6　Decoupled Cross-Attention (core 45 lines)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoupledCrossAttention(nn.Module):
    """ Parallel text + image cross-attention with summed outputs. """
    def __init__(self, d_model, num_heads, d_text, d_image, lam=1.0):
        super().__init__()
        self.h = num_heads
        self.d = d_model
        self.d_head = d_model // num_heads
        self.lam = lam

        # text path: same as original SD; load pretrained weights and **freeze**
        self.W_Q     = nn.Linear(d_model, d_model, bias=False)   # query from latent
        self.W_K_txt = nn.Linear(d_text,  d_model, bias=False)
        self.W_V_txt = nn.Linear(d_text,  d_model, bias=False)
        self.W_O     = nn.Linear(d_model, d_model, bias=False)
        for p in (*self.W_Q.parameters(),
                  *self.W_K_txt.parameters(),
                  *self.W_V_txt.parameters(),
                  *self.W_O.parameters()):
            p.requires_grad_(False)        # ← IP-Adapter only trains the new K/V

        # image path: new K/V; only these two + the projector are trainable
        self.W_K_img = nn.Linear(d_image, d_model, bias=False)
        self.W_V_img = nn.Linear(d_image, d_model, bias=False)

    def _split_heads(self, x):     # [B, L, D] → [B, H, L, d_head]
        B, L, _ = x.shape
        return x.view(B, L, self.h, self.d_head).transpose(1, 2)

    def _attn(self, Q, K, V):       # standard scaled dot-product
        return F.scaled_dot_product_attention(Q, K, V)   # [B, H, L_q, d_head]

    def forward(self, z, c_text, c_image):
        # z: [B, L_z, D]   c_text: [B, L_t, d_text]   c_image: [B, L_i, d_image]
        Q     = self._split_heads(self.W_Q(z))
        K_txt = self._split_heads(self.W_K_txt(c_text))
        V_txt = self._split_heads(self.W_V_txt(c_text))
        K_img = self._split_heads(self.W_K_img(c_image))
        V_img = self._split_heads(self.W_V_img(c_image))

        out_txt = self._attn(Q, K_txt, V_txt)
        out_img = self._attn(Q, K_img, V_img)
        out = out_txt + self.lam * out_img            # ← decoupled sum

        # [B, H, L_q, d_head] → [B, L_q, D]
        B, _, L_q, _ = out.shape
        out = out.transpose(1, 2).contiguous().view(B, L_q, self.d)
        return self.W_O(out)
```

> 💡 **Why decoupled beats concat** — a naive idea is to **concat** image embeddings to text embeddings (variable-length sequence with a single cross-attn). But IP-Adapter Table 4 shows concat significantly degrades text alignment (CLIP-Score drop) — because Q shares the **same softmax** over K_txt and K_img; whichever has more tokens (typically image) hogs attention. Decoupled uses **two independent softmax outputs summed linearly**, with no crowding — the better engineering choice.

### 6.7　InstantID / PuLID / PhotoMaker

**The goal** for all is single-reference-image identity-preserving text-to-image. Mainstream approaches:

| Method | Core mechanism |
|---|---|
| **InstantID** (Wang, Bai et al. 2024) | IP-Adapter-style decoupled cross-attention, injecting the face embedding directly (extracted by ArcFace/insightface; this embedding itself is the identity embedding) + **IdentityNet** (a ControlNet-like branch conditioned on 5-point facial landmarks, handling spatial pose control) |
| **PuLID** (Guo, Wu et al. 2024 NeurIPS) | Dual branch + contrastive alignment, preventing ID signal from contaminating prompt-following |
| **PhotoMaker** (Li, Cao et al. 2024 CVPR) | **Stacked ID Embedding**: for each reference photo, fuse that photo's CLIP image embedding with the class-word text embedding into a per-photo ID embedding, then **stack** the per-photo ID embeddings along the token dimension (rather than first averaging CLIP embeddings across photos and then fusing once with the class embedding), and inject into cross-attn |

Common thread: **ID-relevant signal** goes through dedicated adapters; **ID-irrelevant signal** (pose / expression / lighting) stays prompt-controlled, avoiding direct identity paste.

## §7 Personalization: DreamBooth / Textual Inversion / LoRA / Custom Diffusion

### 7.1　Textual Inversion (Gal et al. 2023 ICLR)

**Train only one token embedding, leave the model untouched**:

1. Introduce a new token `S*` (e.g., `<my-cat>`); its embedding $e_{S^*} \in \mathbb{R}^{d_\text{text}}$ is the **only trainable parameter**

2. Training objective:

$$e_{S^*}^* = \arg\min_{e} \mathbb{E}_{z, \epsilon, t}\left[\|\epsilon - \epsilon_\theta(z_t, t, c(\text{"a photo of } S^*\text{"}; e))\|^2\right]$$

3. ~3-5K steps to converge; **embedding is only 768-1024 dimensions, file size < 10KB**

**Pros**: extremely lightweight, no catastrophic forgetting; **Cons**: limited expressivity (one embedding can't capture complex concepts).

### 7.2　DreamBooth (Ruiz et al. 2023 CVPR)

Two core ingredients:

**1) Rare token + class word**: use a rare token (e.g., `sks`, `zwx`) + class word (`dog`, `person`), with prompts shaped like `"a photo of sks dog"`. Rare tokens have "semantic dead zones" in pretrained embeddings, free from interference by existing concepts.

**2) Prior Preservation Loss**:

$$\boxed{\;\mathcal{L} = \mathbb{E}\!\left[\|\epsilon - \hat\epsilon_\theta(z_t, t, c_\text{sks})\|^2\right] + \lambda\,\mathbb{E}\!\left[\|\epsilon' - \hat\epsilon_\theta(z_t', t, c_\text{class})\|^2\right]\;}$$

The second term is "class-prior preservation" — use the model's **own generated** class images (e.g., 200 images of `"a photo of dog"`) as anchors, telling the model "sks dog is special, but a generic dog must still be drawn correctly". $\lambda$ is usually 1.0.

> ⚠️ **Failure modes without prior preservation** — (i) **Language drift**: the model shifts the "dog" concept entirely toward the sks-specific shape; (ii) **Concept bleed**: every `dog` prompt produces an sks dog; (iii) **Overfitting**: ~5 training images are memorized, and different prompts produce near-identical outputs. **Production practice**: DreamBooth must use prior preservation or LoRA-DreamBooth (more stable).

### 7.3　LoRA (Hu et al. 2022 ICLR)

**Core math**: instead of fine-tuning $W \in \mathbb{R}^{d \times k}$ directly, learn a **low-rank delta**:

$$\boxed{\;W' = W + \Delta W,\quad \Delta W = B A,\quad B \in \mathbb{R}^{d \times r},\ A \in \mathbb{R}^{r \times k},\ r \ll \min(d, k)\;}$$

Typically $A$ is initialized $\mathcal{N}(0, \sigma^2)$, $B$ is initialized to 0 → $\Delta W = 0$ (preserving pretrained behavior) → only $A, B$ are updated. At inference: $W' = W + \alpha B A$ ($\alpha$ is a scaling factor).

**Parameter savings**: $W$ has $d \cdot k$ params; LoRA has $r(d + k)$. On SD's U-Net cross-attn, one $W_K$ is $D \times d_\text{text}$ (e.g., $1280 \times 2048 = 2.6M$); with $r = 8$, LoRA is $8(1280 + 2048) = 26K$ — **100× fewer**.

### 7.4　LoRA injection into nn.Linear (core 40 lines)

```python
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    """ Wrap nn.Linear with low-rank delta. Original weight is frozen. """
    def __init__(self, base: nn.Linear, rank=8, alpha=8.0, dropout=0.0):
        super().__init__()
        self.base = base                          # freeze original Linear
        for p in self.base.parameters():
            p.requires_grad_(False)

        d_in, d_out = base.in_features, base.out_features
        self.rank, self.alpha = rank, alpha
        self.scale = alpha / rank                 # unified inference scaling

        # ΔW = B A,  A: [r, d_in],  B: [d_out, r]
        self.A = nn.Parameter(torch.empty(rank, d_in))
        self.B = nn.Parameter(torch.zeros(d_out, rank))   # B = 0  → ΔW = 0
        nn.init.kaiming_uniform_(self.A, a=5**0.5)        # like nn.Linear default

        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        # original path: use frozen base
        out = self.base(x)
        # LoRA delta: x @ A^T @ B^T  (order matters — avoid the large [d_out, d_in] matrix)
        out = out + self.drop(x) @ self.A.t() @ self.B.t() * self.scale
        return out

def inject_lora(model, target_module_names=("to_q", "to_k", "to_v"),
                rank=8, alpha=8.0):
    """ Replace cross-attn / self-attn to_q/to_k/to_v Linears in SD U-Net with LoRALinear;
        to_out in Diffusers is nn.Sequential([Linear, Dropout]); handle to_out.0 separately. """
    replaced = 0
    for name, mod in model.named_modules():
        for child_name, child in list(mod.named_children()):
            # 1) to_q / to_k / to_v: single Linear, replace directly
            if (child_name in target_module_names
                    and isinstance(child, nn.Linear)):
                setattr(mod, child_name, LoRALinear(child, rank=rank, alpha=alpha))
                replaced += 1
            # 2) to_out: in Diffusers Attention, to_out is nn.Sequential(Linear, Dropout);
            #    SDXL LoRA conventionally wraps only the 0-th (Linear)
            if child_name == "to_out" and isinstance(child, nn.Sequential):
                lin = child[0]
                if isinstance(lin, nn.Linear):
                    child[0] = LoRALinear(lin, rank=rank, alpha=alpha)
                    replaced += 1
    return replaced
```

> 💡 **LoRA on attention QKV vs MLP** — empirically on SD-class models, **placing LoRA on Q/K/V/O of attention yields more gains than on MLP** (attention is the bottleneck of text-image cross-modal interaction; tuning these directly changes conditioning behavior). For LLMs it's the opposite: MLP carries more knowledge → MoE LoRA also often goes in the FFN. SDXL LoRA default coverage: `to_q`, `to_k`, `to_v`, `to_out.0` + some convs (e.g., `conv1`, `conv2` in ResBlocks).

### 7.5　DreamBooth + LoRA = LoRA-DreamBooth

In practice **pure DreamBooth is rare** (full-train is too heavy); the mainstream is LoRA-DreamBooth: inject LoRA only into the Q/K/V/O of attention, with prior preservation. Files ~50-200MB, single-card 30 minutes, much more reproducible than pure DreamBooth.

### 7.6　DreamBooth training step (core 35 lines)

```python
import torch
import torch.nn.functional as F

def dreambooth_train_step(unet, vae, text_encoder, scheduler,
                          x_instance, c_instance,    # training images + "a sks dog" embedding
                          x_class, c_class,          # self-generated class images + "a dog" embedding
                          lam_prior=1.0, dtype=torch.float16, device="cuda"):
    """ One DreamBooth + prior preservation training step """
    bs = x_instance.shape[0]

    # 1) concat instance & class into a 2× batch for a single forward
    x = torch.cat([x_instance, x_class], dim=0)
    c = torch.cat([c_instance, c_class], dim=0)        # text embeddings

    # 2) encode to latent + scale
    with torch.no_grad():
        z = vae.encode(x).latent_dist.sample() * 0.18215    # [2bs, 4, h, w]

    # 3) sample random timestep + noise
    t = torch.randint(0, scheduler.num_train_timesteps, (z.shape[0],), device=device)
    eps = torch.randn_like(z)
    z_t = scheduler.add_noise(z, eps, t)

    # 4) predict ε
    eps_pred = unet(z_t, t, encoder_hidden_states=c).sample

    # 5) split loss into instance / class halves
    eps_pred_inst, eps_pred_cls = eps_pred.chunk(2, dim=0)
    eps_inst, eps_cls = eps.chunk(2, dim=0)

    loss_inst = F.mse_loss(eps_pred_inst.float(), eps_inst.float(),
                           reduction="mean")
    loss_cls  = F.mse_loss(eps_pred_cls.float(),  eps_cls.float(),
                           reduction="mean")
    loss = loss_inst + lam_prior * loss_cls

    return loss
```

### 7.7　HyperDreamBooth / Custom Diffusion comparison

- **HyperDreamBooth** (Ruiz et al. 2024): use a hypernetwork to directly predict per-reference-image LoRA weights, enabling **inference-time personalization** (~5 seconds vs DreamBooth's ~10-minute training).

- **Custom Diffusion** (Kumari et al. 2023 CVPR): directly fine-tunes (no low-rank decomposition) the full cross-attention $W_K, W_V$ matrices (leaving $W_Q$ alone), paired with real same-class regularization images to prevent concept forgetting / overfitting. It is a **selective full-parameter fine-tuning** method independent of LoRA, not a narrower variant of LoRA-DreamBooth.

| Method | Trained params | Inference cost | Expressivity | File size |
|---|---|---|---|---|
| Textual Inversion | embedding (1 token) | 0 | weak | < 10 KB |
| DreamBooth (full) | entire U-Net | 0 | strong | ~5 GB |
| LoRA-DreamBooth | LoRA on Q/K/V/O | 0 after merge | strong-ish | 50-200 MB |
| Custom Diffusion | W_K, W_V only | 0 | medium | ~70 MB |
| HyperDreamBooth | one hypernet outputs LoRA | one extra hypernet forward | medium | ~120 MB main net |

## §8 Image Editing: SDEdit / InstructPix2Pix / Prompt-to-Prompt

### 8.1　SDEdit (Meng et al. 2022 ICLR)

**Idea**: image editing = "add partial noise to the input image → reverse back guided by the prompt".

```
   input image x (e.g. sketch)
         │
         │   noise_strength = 0.6 (example)
         ▼
   z_0 = VAE_enc(x)
         │
   z_τ = √(ᾱ_τ) z_0 + √(1 - ᾱ_τ) ε,   τ = noise_strength × T
         │
         ▼
   reverse SDE / ODE from t=τ to t=0    (guided by prompt c)
         │
         ▼
   z_0' →  VAE_dec → edited image x'
```

**Key parameter strength $\in [0, 1]$**:

- $\text{strength} \to 0$: very little noise; output $\approx$ input (no editing)

- $\text{strength} \to 1$: fully noised; output = pure text-to-image (input signal lost)

- Common range 0.3–0.8 for the right balance

### 8.2　SDEdit core code

```python
import torch

@torch.no_grad()
def sdedit_sample(unet, vae, text_encoder, tokenizer, scheduler,
                  init_image, prompt, neg_prompt="",
                  strength=0.7, num_steps=30, cfg_scale=7.0,
                  device="cuda", dtype=torch.float16):
    assert 0.0 < strength <= 1.0, "strength=0 == no editing (return input directly); strength>1 is invalid"

    # 1) text encoding (same as §5)
    ids_pos = tokenizer(prompt, padding="max_length", max_length=77,
                        truncation=True, return_tensors="pt").input_ids.to(device)
    ids_neg = tokenizer(neg_prompt, padding="max_length", max_length=77,
                        truncation=True, return_tensors="pt").input_ids.to(device)
    emb = torch.cat([text_encoder(ids_neg)[0],
                     text_encoder(ids_pos)[0]], dim=0)

    # 2) encode the input image to latent
    z0 = vae.encode(init_image).latent_dist.sample() * 0.18215   # [1, 4, h, w]

    # 3) set timestep subset; only traverse the last `strength` portion
    scheduler.set_timesteps(num_steps, device=device)
    import math
    # use ceil + max(1, .) to ensure at least 1 step runs when strength>0
    n_edit = max(1, math.ceil(num_steps * strength))
    t_start = num_steps - n_edit               # 0 ≤ t_start < num_steps
    timesteps = scheduler.timesteps[t_start:]  # at least 1 timestep

    # 4) add noise to z_0 at the timesteps[0] level
    eps = torch.randn_like(z0)
    z = scheduler.add_noise(z0, eps, timesteps[:1])

    # 5) main loop (identical to §5)
    for t in timesteps:
        z_in = torch.cat([z, z], dim=0)
        z_in = scheduler.scale_model_input(z_in, t)
        eps_pred = unet(z_in, t, encoder_hidden_states=emb).sample
        eps_neg, eps_pos = eps_pred.chunk(2, dim=0)
        eps_cfg = eps_neg + cfg_scale * (eps_pos - eps_neg)
        z = scheduler.step(eps_cfg, t, z).prev_sample

    # 6) decode
    z = z / 0.18215
    return ((vae.decode(z).sample.clamp(-1, 1) + 1) / 2)
```

### 8.3　InstructPix2Pix (Brooks et al. 2023 CVPR)

**Instruction-style editing**: "add a hat to the dog". Training data is synthesized via GPT-3 + Prompt-to-Prompt (pairwise (source, instruction, target) tuples), used to fine-tune SD 1.x. Architecture:

- U-Net input channels 4 → 8 (original latent 4 + source image latent 4)

- Two CFG scales: text guidance $s_T$ and image guidance $s_I$, tuned independently

$$\hat\epsilon = \hat\epsilon(\emptyset, \emptyset) + s_I [\hat\epsilon(c_I, \emptyset) - \hat\epsilon(\emptyset, \emptyset)] + s_T [\hat\epsilon(c_I, c_T) - \hat\epsilon(c_I, \emptyset)]$$

### 8.4　Prompt-to-Prompt (Hertz et al. 2023 ICLR)

**Training-free editing**: achieves "change words but keep structure" by **manipulating cross-attention maps**.

- Run the original prompt P and record the per-layer per-step cross-attn maps $M_t$ (shape $[H_q, L_t]$ — weight each image token assigns to each text token)

- Run new prompt P* (changing only one word, e.g., `cat` → `dog`) but **force-replace the attention map for the corresponding text token positions with the original P's**

- Thus structure (which image patches attend to which text positions) is preserved; only content (the values aggregated after softmax) changes

Suitable for "swap one word", "add adjective", and "reweight emphasis" types of edits.

## §9 Distillation: Few-Step Sampling

### 9.1　LCM / LCM-LoRA (Luo et al. 2023)

**Latent Consistency Model**: distill latent diffusion into a **Consistency Model** (Song et al. 2023), letting a single-step prediction $f_\theta(z_t, t)$ directly yield $z_0$ — compressing 50 steps to 4-8.

**LCM-LoRA**: package this distillation as a **LoRA**. Base SDXL/SDXL-1.0 + a ~200MB LCM-LoRA = 4-step generation. Zero-shot plug-in, **stacks with personalization LoRA**.

### 9.2　SDXL-Turbo: ADD (Sauer et al. 2024 ECCV) / SD3-Turbo: LADD (Sauer et al. 2024)

**SDXL-Turbo = ADD (Adversarial Diffusion Distillation, arXiv 2311.17042 / ECCV 2024)**; **SD3-Turbo = LADD (Latent Adversarial Diffusion Distillation, arXiv 2403.12015)** — two distinct methods. ADD uses a pixel-space vision encoder as discriminator; LADD moves to a latent-space discriminator that scales to the SD3 model.

**Adversarial Diffusion Distillation (ADD)**:

- **Teacher**: original SDXL (multi-step diffusion model)

- **Student**: same architecture as teacher, targeting 1-4 step generation

- **Two losses** (not three — the distillation term is itself a score-distillation-style loss, not a separate third term):

  1. Adversarial loss: DINOv2-based discriminator judges the student's single-step output

  2. Distillation loss (score-distillation style): re-noise the student's single-step denoised output to a randomly sampled higher noise level, denoise it with the **frozen teacher** at that level, and regress the student's output toward that teacher reconstruction — there is **no** separate third MSE/LPIPS term matching the teacher's full multi-step sampling trajectory

ADD is one of the SOTA few-step distillation routes; FLUX schnell is based on similar ideas ("timestep distillation" + adversarial).

### 9.3　DMD / DMD2 / Hyper-SD short table

| Method | Core |
|---|---|
| DMD (Yin et al. 2024) | Indirectly align student and teacher distributions via KL divergence, 1-step generation |
| DMD2 (Yin et al. 2024) | DMD + training stability tricks |
| Hyper-SD (Ren et al. 2024) | trajectory-segmented consistency distillation |
| Lightning SDXL | 4-8 step SDXL distillation; the popular open-source variant |

## §10 Evaluation: FID / CLIP-Score / ImageReward / HPSv2 / PickScore

| Metric | Computation | What it measures |
|---|---|---|
| **FID** (Heusel et al. 2017 NeurIPS) | Fréchet distance over Inception-V3 pool3 features (real vs gen) | overall distribution similarity (diversity + realism) |
| **CLIP-Score** | mean (text, image) cosine similarity from CLIP | text alignment |
| **ImageReward** (Xu et al. 2023 NeurIPS) | reward model trained on human preference data (ViT + CLIP backbone) | holistic human preference (aesthetics / text alignment / realism) |
| **HPSv2** (Wu et al. 2023) | Human Preference Score V2, similar to ImageReward but more data | human preference (finer categories) |
| **PickScore** (Kirstain et al. 2023 NeurIPS) | CLIP-based, trained on Pick-a-Pic dataset | user preference |

> ⚠️ **FID limitations** — (i) insensitive to mode collapse (low-variance generation can paradoxically lower FID); (ii) Inception-V3 was trained on ImageNet — biased on faces / art / non-natural images; (iii) **need ≥ 10K generations**; with <5K variance is huge and inter-paper comparison is unsafe; (iv) FID-30K vs FID-10K are not directly comparable. **In interviews, proactively note FID is not the final word**; pair it with human preference metrics (IR / HPSv2 / PickScore).

## §11 Complexity / Resources

> ⚠️ **Numbers in this section are rough estimates** — training A100-hours, inference seconds, and peak memory come from community estimates and individual public reports. Accuracy depends on batch size / sequence implementation / optimizer / memory strategy; in interviews, state up front "these are order-of-magnitude estimates, not official figures".

### 11.1　Training side (order-of-magnitude estimates)

| Model | Params | Training data | Training compute (order-of-magnitude) |
|---|---|---|---|
| SD 1.5 | 860M U-Net + 84M VAE + 123M CLIP-L | LAION-5B → LAION-aesthetics 2B | ~150K A100-hr scale |
| SDXL | 2.6B U-Net + Refiner | internal + LAION | ~250-300K A100-hr scale |
| SD3 | 2B / 8B MM-DiT (largest 8B) | internal ~1B | undisclosed (≫ SDXL) |
| FLUX.1 | 12B MM-DiT | internal | undisclosed (massive) |

### 11.2　Inference side (rough, implementation-dependent)

**SD 1.5 512×512 (native resolution, latent 64×64×4)**:

- Per-step U-Net FLOPs: ~0.5T

- Per-step cross-attn QKV: (L_z=4096) × (L_t=77) → 0.3M score matrix × multi-layer

- Per-step memory: ~2-3 GB (FP16, no cross-attn KV cache)

- 30 steps ≈ 1.5-2 s / image (A100, FP16)

(SD 1.5 at 1024² is ~4× slower than 512²; the community does not recommend non-native-resolution direct inference, typically pairing it with SDXL or super-resolution.)

**SDXL 1024×1024**:

- Per-step U-Net FLOPs: ~1.2T

- 30 steps ≈ 4-5 s / image (A100, FP16)

**FLUX.1-dev 1024×1024**:

- MM-DiT FLOPs ≈ 2.5T / step

- 28 steps ≈ 12-15 s (A100); H100 ~5-7 s

### 11.3　Memory footprint cheat sheet

| Component | Estimate |
|---|---|
| Latent ($1024^2 / f=8$) | $128 \times 128 \times 4 \times 2$ bytes (FP16) ≈ 130 KB |
| U-Net activations (SDXL, batch 1) | ~7 GB (FP16; needs gradient checkpointing for training) |
| Cross-attn scores (peak) | $16384 \times 77 \times 2 \times \text{heads} \approx$ a few MB / layer |
| KV cache for text | text is a fixed 77 tokens, CFG double batch → 154 tokens equivalent; can be fully cached batch-wide |

## §12 Comparisons with Related Methods

### 12.1　Diffusion family vs other generative models

| Family | Speed | Quality | Training stability |
|---|---|---|---|
| **Diffusion / Flow** | slow (multi-step) | high | high (MSE regression) |
| GAN (StyleGAN, BigGAN) | fast (1 step) | high (specific domains) | low (mode collapse) |
| VAE | fast | low (blurry) | high |
| Autoregressive (DALL-E, Parti) | medium (token by token) | medium (DALL-E 1) / high (Parti) | high |
| Hybrid (Muse) | medium (few-step parallel token) | medium-high | medium |

### 12.2　Internal routes within diffusion

| Route | Representation | Training objective | Representative |
|---|---|---|---|
| Pixel-space DDPM | pixel | $\epsilon$-pred | Imagen, GLIDE |
| Latent diffusion | VAE latent | $\epsilon$-pred / $v$-pred | SD 1.x/2.x, SDXL |
| Latent rectified flow | VAE latent | $u_t = x_1 - x_0$ | SD3, FLUX |
| Score-based EDM | pixel / latent | preconditioned $\epsilon$ | EDM, EDM2 |

## §13 25 Frequently-Asked Interview Questions (codex 5.5 xhigh top-lab interviewer perspective)

Expand each entry to see key answer points + common pitfalls.

### L1 must-know (asked at every vision / multimodal engineering interview)

<details>

<summary>Q1. Why is LDM more compute-efficient than pixel-space diffusion?</summary>

- VAE downsamples $H\times W$ by $f=8$: $(H/8)\times(W/8)$ tokens

- Token count drops by $f^2 = 64$, and FLOPs drop by a similar order of magnitude (not an exact ratio — wider latent-space channels and the quadratic scaling of self-attention FLOPs with token count make actual savings deviate from $64\times$)

- Same compute budget enables higher resolution (512² → 1024²)

Common pitfall: just saying "compression" without the factor derivation; treating the token-count reduction as an exact FLOPs saving; forgetting that VAE reconstruction quality is the upper bound.

</details>

<details>

<summary>Q2. What is SD 1.x's latent shape?</summary>

- Input $512\times 512\times 3$; after VAE encode $64\times 64\times 4$

- $f = 8$ downsample, $c = 4$ channels

- **After VAE encode**, **multiply** the latent by scaling factor `0.18215` (SD 1.x) so variance approaches 1 before feeding diffusion; **before VAE decode**, **divide** it back. Training / inference must agree — encode-multiply, decode-divide

Common pitfall: wrong scaling factor (SDXL is 0.13025, SD3 uses scalar `scaling_factor=1.5305` + `shift_factor=0.0609`); or inverted direction (e.g., applying only at inference, not training — latent variance is off).

</details>

<details>

<summary>Q3. What is the CFG formula? How is the uncond branch trained?</summary>

- Inference: $\hat\epsilon_\text{cfg} = \hat\epsilon_\emptyset + s(\hat\epsilon_c - \hat\epsilon_\emptyset)$, $s \in [1.5, 12]$

- Training: with probability $p_\text{drop} \approx 0.1$, replace the condition with null embedding

- The same model learns both conditional and unconditional branches

Common pitfall: treating $s$ as a temperature knob; forgetting the dropout step at training time.

</details>

<details>

<summary>Q4. How is cross-attention used in the SD U-Net?</summary>

- Image latent tokens are Q

- Text token embeddings are K, V

- Appears in **U-Net levels equipped with transformer blocks** (SD 1.5 config `attention_resolutions=[4,2,1]`: DS=1/2/4, i.e. latent 64/32/16 downsample levels + corresponding upsample + middle block 8×8 (at 512² input with f=8 VAE → 64² latent, DS=8 means 8×8 middle); the SD/SDXL middle block is a standard SpatialTransformer / `UNetMidBlock2DCrossAttn`: ResBlock → (self-attention → cross-attention → FFN transformer block) → ResBlock — **it does perform cross-attn** (consistent with §3.1), not "pure ResBlock + self-attention with no cross-attn")

- Inside each transformer block the order is: self-attention → cross-attention → FFN

Common pitfall: swapping Q and K/V sources; assuming the 8×8 down/up blocks also have cross-attn (only middle does); or assuming cross-attn only at the bottleneck.

</details>

<details>

<summary>Q5. Main changes from SD 1.5 to SDXL?</summary>

- U-Net 860M → 2.6B

- CLIP-L → OpenCLIP-G + CLIP-L dual encoders

- Size + crop conditioning: training resolution signal explicitly injected

- Refiner (optional) does the last ~20% noise level for detail refinement

- Training resolution 512² → 1024², bucketing across aspect ratios

Common pitfall: only saying "bigger parameters", missing size conditioning / dual encoders.

</details>

<details>

<summary>Q6. What does ControlNet's zero-conv do?</summary>

- On the 1×1 conv that connects the trainable copy back to the backbone, initialize weight and bias to 0

- Forward: $y = 0 \cdot x + 0 = 0$; added to backbone = no change (identity warm start)

- Backward: $\partial L / \partial W = \partial L / \partial y \cdot x \neq 0$, still updates

- Both protects pretrained SD capability and learns conditional signal

Common pitfall: thinking zero-conv "can't be trained"; or conflating with BN / dropout.

</details>

<details>

<summary>Q7. What is the LoRA formula? How much parameter saving?</summary>

- $W' = W + B A$, $A \in \mathbb{R}^{r \times k}$, $B \in \mathbb{R}^{d \times r}$, $r \ll \min(d, k)$

- $W$ frozen, only $A, B$ trained

- Parameters from $dk$ to $r(d+k)$, exact ratio $r(d+k)/dk = r(1/d+1/k)$; when $d=k=n$ (square matrices, e.g. common attention Q/K/V/O projections) this is $2r/n$, **not** $r/n=r/\min(d,k)$ (off by a factor of 2); on SD, $W_K$ ($d=1280, k=2048$) with $r=8$ gives ~100× fewer parameters (the exact factor depends on the shape of $d,k$)

Common pitfall: quoting the ratio as $r/\min(d,k)$ (missing the factor of 2); swapping shapes of $A, B$; forgetting $W$ is frozen.

</details>

<details>

<summary>Q8. What problem does DreamBooth's prior preservation loss solve?</summary>

- Prevents language drift: model shifting the class concept entirely toward the sks-personalized form

- Prevents concept bleed: every `dog` prompt producing sks dog

- Uses model-generated class images as anchors; $L = L_\text{instance} + \lambda L_\text{class}$

Common pitfall: only remembering the sks rare token, missing the prior loss; or thinking prior loss is parameter regularization.

</details>

<details>

<summary>Q9. How does IP-Adapter differ from directly concatenating image tokens with text?</summary>

- IP-Adapter uses **decoupled cross-attn**: new $W_K', W_V'$ in parallel with text; outputs are **linearly summed**

- Concat would force Q to share one softmax over (text, image); image's longer length hogs text alignment

- Decoupled = two independent softmaxes, no mutual crowding

Common pitfall: just saying "added image embedding" without explaining the softmax exclusion problem.

</details>

<details>

<summary>Q10. What are the limitations of FID?</summary>

- Inception-V3 is ImageNet-trained — feature bias (faces / art styles inaccurate)

- Insensitive to mode collapse (low variance can give low FID)

- High variance when generation count <10K; cross-paper comparison unsafe

- Doesn't evaluate per-image quality, only distribution; pair with CLIP-Score / ImageReward / HPSv2

Common pitfall: treating FID as a universal gold standard; comparing FID-10K vs FID-30K directly.

</details>

### L2 advanced (research / production positions)

<details>

<summary>Q11. How is SDXL's size conditioning trained / used at inference?</summary>

- At training, record each image's original $(h_\text{orig}, w_\text{orig})$, crop origin $(h_\text{crop}, w_\text{crop})$, and **target size** $(h_\text{tgt}, w_\text{tgt})$ (used for multi-aspect-ratio training, SDXL paper §2.3)

- These 3 pairs — 6 scalars total — are each Fourier-embedded, passed through an MLP, then **added to the timestep embedding** before entering the U-Net

- At inference, fill $(1024, 1024)$ and $(0, 0)$ to tell the model "I want 1024 full-image quality", avoiding leakage of low-res / crop artifacts

- Can intentionally fill smaller sizes / nonzero crops to control style

Common pitfall: just saying "added resolution labels" while missing the target-size pair; not explaining the injection point (time embedding) and the Fourier-embed detail.

</details>

<details>

<summary>Q12. What is the information-flow difference between MM-DiT and SDXL cross-attn?</summary>

- SDXL: in cross-attn, image latent is Q and text is K/V, **unidirectional** (text not updated)

- MM-DiT: text tokens and image tokens are **concatenated into a single sequence for joint self-attn**, with independent QKV projections but a global attention matrix, **bidirectional** (text is also updated by image)

- Empirically: MM-DiT significantly improves text alignment on complex prompts (SD3 paper Table 1)

Common pitfall: reversed direction; thinking MM-DiT is just "bigger cross-attn".

</details>

<details>

<summary>Q13. Why is the AdaLN-Zero gate initialized to 0? How do gradients flow in early training?</summary>

- $\text{Block}(x, c) = x + \alpha(c) \cdot f\big((1+\gamma(c)) \odot \text{LN}(x) + \beta(c)\big)$ (DiT uses **1+γ**, not γ multiplied directly)

- $\alpha = 0$ → block output = $x$ (identity), guaranteeing deep-stack warm start

- Gradient through $\partial L / \partial \alpha = \partial L / \partial \text{out} \cdot f((1+0)\text{LN}(x)+0) \neq 0$; $\alpha$ slowly grows from 0

- Note $\gamma, \beta$ themselves have zero gradient at step 0 (gated by $\alpha = 0$); they only follow after $\alpha$ grows — two-stage dynamics

- Equivalent to a "learnable identity shortcut" — the key design that enables DiT to scale

Common pitfall: writing AdaLN as $\gamma\odot\text{LN}(x)+\beta$ (missing the 1+γ bias); thinking $\alpha = 0$ blocks training; conflating AdaLN-Zero with vanilla dropout.

</details>

<details>

<summary>Q14. Is LoRA on attention QKV or on MLP better? Why?</summary>

- SD-style visual generation: **attention QKV** wins (the cross-modal conditioning bottleneck is in cross-attn)

- LLM: MLP carries more task knowledge → placing in MLP / FFN wins

- SDXL LoRA defaults to covering `to_q, to_k, to_v, to_out.0` + some convs

- This is an empirical SD vs LLM fine-tuning gap, tied to "which parameters carry the conditional interaction"

Common pitfall: mixing SD's and LLM's LoRA configs; assuming "anywhere works".

</details>

<details>

<summary>Q15. Differences between Rectified Flow and DDPM in training objective / sampling?</summary>

- **DDPM** ε-pred: $\mathcal{L} = \|\epsilon - \hat\epsilon_\theta(x_t, t)\|^2$, $x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\epsilon$

- **RF**: $x_t = (1-t)x_0 + tx_1$ ($x_0$ data, $x_1$ noise), $\mathcal{L} = \|u_t - v_\theta(x_t, t)\|^2$, $u_t = x_1 - x_0$ is constant in $t$

- **Sampling**: DDPM commonly uses DDIM / DPM++ (high-order solvers from SDE / ODE); RF uses direct Euler / Heun

- RF has straight trajectories, better few-step sampling quality (the key reason SD3 / FLUX adopt RF)

Common pitfall: treating RF as "another noise schedule"; forgetting RF's target is $x_1 - x_0$ not $\epsilon$.

</details>

<details>

<summary>Q16. What is the SD VAE scaling factor? Why is it needed?</summary>

- VAE encode outputs raw latent with std far from 1 (SD 1.x raw std ≈ 5.5)

- Using it directly causes the diffusion noise schedule to mismatch ($x_t$ signal/noise ratio off)

- SD 1.x multiplies by scalar `0.18215` so latent variance ≈ 1; SDXL uses `0.13025`; SD3 uses scalar `scaling_factor=1.5305` + scalar `shift_factor=0.0609` (not per-channel mean/std)

- After VAE encode, × scaling (SD3 also subtracts shift first); before VAE decode, ÷ scaling (SD3 also adds shift back); must align strictly

Common pitfall: wrong value; thinking SD3 uses per-channel arrays (it's actually scalar+shift); forgetting to do it at both encode and decode.

</details>

<details>

<summary>Q17. How does SDEdit's strength parameter affect output?</summary>

- strength $\in [0, 1]$ determines the noised-to timestep $\tau = \text{strength} \cdot T$

- strength → 0: almost no noise; output ≈ input (no editing)

- strength → 1: fully noised; output = pure text-to-image (input completely lost)

- Common 0.3-0.8 to find the "preserve structure + follow prompt" sweet spot

Common pitfall: treating strength as CFG scale; forgetting 0 means "no editing".

</details>

<details>

<summary>Q18. Why can Prompt-to-Prompt preserve structure?</summary>

- When running the original prompt, save per-layer per-step cross-attn maps $M_t$ (image patch → text token weights)

- When running the new prompt, force-use the original $M_t$ for **unchanged tokens**

- Structure (which patches attend to which positions) is preserved; content (values aggregated after softmax) follows the new prompt

- Suitable for single-word swap / token re-weighting; not for large prompt rewrites

Common pitfall: thinking P2P modifies the latent; not knowing it manipulates attention maps.

</details>

<details>

<summary>Q19. Difference between LCM / Consistency Model and ADD distillation?</summary>

- **LCM** (Luo 2023): trains the student to directly predict $z_0$; target is "self-consistency" along the ODE trajectory (consistency loss)

- **ADD** (Sauer et al. 2024 ECCV, preprint arXiv 2311.17042 / 2023): adversarial loss (DINOv2 discriminator) + distillation loss (score-distillation style) — two terms, not three-in-one

- LCM-LoRA is in LoRA form, plug-and-play on any base; ADD is full distillation, slightly stronger but requires retraining

- Both compress NFE from 25-50 to 1-4

Common pitfall: treating LCM as a GAN; thinking ADD only has the adversarial loss.

</details>

<details>

<summary>Q20. Is the SDXL Refiner required? When to use?</summary>

- Not required; the Refiner is a separate latent diffusion model dedicated to the last ~20% noise level

- Mainly refines details (skin / hair / texture)

- When the base U-Net hits $t < 0.2$, switch to the Refiner to continue

- In practice the Refiner's gain is small but adds time; the community often **skips** it

Common pitfall: assuming Refiner is mandatory; saying it's "trained jointly" (actually two-stage independent training).

</details>

### L3 top-lab / deep questions (research-lead perspective)

<details>

<summary>Q21. Derive in detail why ControlNet's zero-conv has "forward 0, gradient nonzero".</summary>

Zero-conv layer: $y = W \star x + b$, $W \in \mathbb{R}^{c_\text{out} \times c_\text{in} \times 1 \times 1}$, initialized $W = 0, b = 0$, input $x$ from trainable copy output (nonzero):

- **Forward**: $y = 0 \star x + 0 = 0$. Adding to backbone skip = "no change", so step-0 output = baseline SD output.

- **Backward (zero-conv's own weights)**: $\partial L / \partial W_{ij} = \partial L / \partial y_i \cdot x_j$. $x_j$ comes from the trainable copy (initialized as pretrained SD encoder weights, gives nonzero activations on any nonzero input); $\partial L / \partial y_i$ from downstream loss; the **product is nonzero**, so $W$ updates — this is the key to zero-conv "breaking zero" on its own.

- **Backward (trainable copy's parameters) — the subtle bit**: trainable copy's parameters $\theta_c$ receive gradient only through the $x \to y$ path. Chain rule gives $\partial L / \partial \theta_c = (\partial L / \partial y) \cdot (\partial y / \partial x) \cdot (\partial x / \partial \theta_c)$. **Note $\partial y / \partial x = W$** — at step 0, $W = 0$, so **the trainable copy's parameter gradient is indeed 0 in the first step**! But once $W$ updates to nonzero at step 1 (per the above), at step 2 $\partial y / \partial x = W \neq 0$ and the trainable copy starts learning.

  Hence ControlNet's "warm start yet learnable" two-stage mechanism: **first zero-conv breaks zero on its own** (step 1), **then drives the trainable copy to learn** (from step 2).

This is the key engineering trick that makes ControlNet stronger than naive "freeze + adapter" — zero-interference start AND the whole branch eventually learns. **Similar ideas appear in**: AdaLN-Zero (DiT — $\alpha = 0$ only blocks the first step, not $\alpha$'s own gradient), LoRA's $B = 0$ initialization ($B = 0$ → $\partial L / \partial A$ contains a $B$ factor = 0, but $\partial L / \partial B$ contains an $A \neq 0$ factor, so $B$ moves first).

Common pitfall: can't write the chain rule; or misreading "forward 0" as "gradient all zero" (ignoring the $\partial y / \partial W = x$ path); or conversely claiming "trainable copy has gradient on the first step" (actually has to wait for zero-conv to break zero).

</details>

<details>

<summary>Q22. SDXL's size / crop conditioning: what training data distribution properties drive it, and how does conditioning fix them?</summary>

LAION training hits 3 distribution problems:

1. **Diverse resolutions**: from 256² to 4K, most < 1024². Naive training is dragged toward "blurry bias" because low-res dominates.

2. **Center-crop bias**: many pipelines center-crop images to square, throwing away edges. The model learns a "subject is in the center" prior, often cutting off heads / feet / edges at generation time.

3. **Aspect ratio is single-mode**: directly resizing to $1024^2$ wastes horizontal/vertical information; bucketing across ratios resolves this (SDXL uses ~30 buckets).

SDXL's conditioning fixes:

- $(h_\text{orig}, w_\text{orig})$ is sinusoidal-Fourier embedded → MLP → added to the timestep embedding. The model can "know" the original resolution; **at inference, fill $(1024, 1024)$ to trigger "generate at 1024 quality" mode**.

- $(h_\text{crop}, w_\text{crop})$ is embedded and injected the same way. At inference, fill $(0, 0)$ means "I started cropping from the top-left of the original". **Intentionally non-zero crops can control viewpoint** (e.g., shift subject to lower-right).

- At training, size/crop signals truthfully reflect the sample; at inference they're used as control knobs.

Ablation conclusion (Podell 2024 ICLR Table 1): removing size conditioning significantly raises FID, especially with blurriness on large-resolution prompts.

Common pitfall: just saying "added size labels" without explaining LAION's distribution problems; forgetting the Fourier embed + timestep injection location.

</details>

<details>

<summary>Q23. MM-DiT vs SDXL cross-attn information flow, and impact on text alignment?</summary>

**SDXL cross-attention per-layer flow**:

```
image latent Q  ─►───┐
                     │   softmax(QK^T/√d) V
                     │
text K/V (frozen)  ──┘──► attended_out → added to image latent residual
```

- text token representations are **not updated**; after every cross-attn, text is still the input embedding (just being "read" multiple times)

- Information flow is unidirectional: text → image; image **cannot enter** text's "context" update

- On complex prompts (e.g., "a red cube to the LEFT of a blue cube, with a small green ball between them"), the model's understanding of **position / relations** can only be extracted from the static text embedding

**MM-DiT per-layer flow**:

```
                 [txt | img] concatenated into one sequence
                       │
        ┌──────────────┼──────────────┐
        │              │              │
  independent QKV (txt)  independent QKV (img)
        │              │              │
        └──────────────┼──────────────┘
                       ▼
              joint self-attention
                  (global attn matrix)
                       │
        ┌──────────────┼──────────────┐
        │              │              │
   text output updated    image output updated
```

- **Every token (text or image) is simultaneously Q and K/V**, with a global attention matrix ($L_\text{txt} + L_\text{img}$ tokens look at each other)

- **Text is also updated by image**: at the current layer, text representation fuses image signal; at the next layer, text is already "context-aware" conditioning

- Spatial relations / multi-object binding on complex prompts improve markedly (SD3 paper Figure 5 shows ~10-20% gains across GenEval items for SD3)

Cost: parameters and compute increase (every token does full MHA).

Common pitfall: treating MM-DiT as just "bigger cross-attn"; failing to distinguish unidirectional vs bidirectional; unable to name concrete improvements on multi-object prompts.

</details>

<details>

<summary>Q24. Which of Q/K/V in SD attention is most critical for LoRA? Theoretical explanation?</summary>

Empirically SD attention LoRA covers $W_Q, W_K, W_V, W_O$ as a 4-piece set (**full set** is typically most stable). If forced to choose one, community experience says:

- **$W_V$ affects content**: in cross-attn, $V$ enters the residual directly; tuning $W_V$ tunes "what content text tokens deliver to image latent". **Style LoRA** tends to move $W_V$.

- **$W_K$ affects selection**: tuning $W_K$ changes "which text tokens get selected by Q". **Identity / concept LoRA** weights $W_K$ heavily (making specific tokens precisely selected).

- **$W_Q$ affects image side**: image latent's retrieval pattern; sensitive to image latent distribution itself.

- **$W_O$ affects mixing**: the post-head-concat linear; tuning $W_O$ tunes "how multi-heads are blended".

Theory: $\Delta(QK^T) = \Delta Q \cdot K + Q \cdot \Delta K + \Delta Q \cdot \Delta K$. The first-order terms are contributed by LoRA-Q and LoRA-K respectively — so K and Q affect the **attention map**; V and O affect the **post-map value pathway**. Modifying the attention map changes "semantic alignment"; modifying the value pathway changes "style / content". This is one explanation for the community's experience that "style LoRA leans on V/O, identity LoRA leans on Q/K".

**Comparison with LLM**: in LLM inference, Q comes from the current token, K/V from the cache; modifying Q directly changes "how the current query retrieves", but K/V are cache-bound — partly why LoRA in LLMs leans toward MLP / FFN (cross-attn is not LLM's main interaction channel).

Common pitfall: just saying "add LoRA everywhere"; not distinguishing the semantics of V / K / Q; can't explain why SD favors attn while LLM favors MLP.

</details>

<details>

<summary>Q25. From a production deployment perspective, enabling SDXL + LCM-LoRA + ControlNet + IP-Adapter together — how do you manage sampler / CFG / memory?</summary>

**Sampler choice**:

- With LCM-LoRA, you must use the **LCM scheduler** (4-8 steps); the DPM++ / DDIM multi-step setups no longer apply

- Typical 4-step config: scheduler `LCMScheduler`, num_steps=4, CFG **low** (LCM has done distilled guidance; stacking ordinary CFG over-saturates — LCM-Distill models commonly use $s = 1.0$, i.e., CFG off; or use explicit W-CFG)

**CFG stacking with ControlNet / IP-Adapter**:

- CFG is "differential amplification" of text

- ControlNet's condition embedding is added at the U-Net side branch and **does not participate** in the math of the CFG double-forward (both batches share the same ControlNet input)

- IP-Adapter's image condition **should** be CFG-dropped together with text (training has drop_rate for image condition); at inference, the unconditional batch's IP condition is also set to empty

**Memory estimation** (FP16, 1024², batch 1):

- SDXL U-Net forward activations: ~6-7 GB (gradient checkpointing only for training)

- ControlNet (full): + ~3-4 GB

- IP-Adapter: +~0.2 GB (small)

- LoRA merged: 0 (no extra cost after merge)

- VAE decode peak: ~1.5 GB

- **Total 1024² inference peak ~12-14 GB**; A10 (24GB) single-card works; T4 (16GB) is tight but works with attention slicing

**Optimizations**:

1. **LoRA merge**: merge LCM-LoRA and style LoRAs into the base, avoiding per-forward $\Delta W$ computation

2. **xformers / FlashAttention**: fuse cross-attn and self-attn, ~30% time and ~20% memory savings

3. **ControlNet quantize / pruning**: production often quantizes ControlNet to INT8, ~1.5GB

4. **Schedule sequential ControlNet calls**: don't run multiple ControlNets in parallel (OOM); sequentially aggregate

5. **Cache text embedding**: when the same prompt is used for many images, run the text encoder only once

**Pitfalls**:

- LCM-LoRA + ControlNet often suffers "weakened structure follow" — LCM distillation never sees ControlNet signal; you need to fine-tune ControlNet on the LCM path (or use the community `ControlNet-LCM` variant)

- IP-Adapter "Plus" (ViT-G + image patches) consumes more memory; for typical ID scenarios, the ViT-L version is enough

Common pitfall: just listing tool names; can't estimate memory; doesn't know the LCM + ControlNet compatibility caveat; can't distinguish injection points of CFG vs ControlNet / IP-Adapter.

</details>

## §A Appendix: References

### Main papers (chronological)

- DDPM — Ho, Jain, Abbeel 2020 NeurIPS

- LDM / Stable Diffusion — Rombach, Blattmann et al. 2022 CVPR

- Classifier-Free Guidance — Ho & Salimans 2022 (workshop / arXiv)

- DiT — Peebles & Xie 2023 ICCV

- U-ViT — Bao, Nie et al. 2023 CVPR

- LoRA — Hu, Shen et al. 2022 ICLR

- DreamBooth — Ruiz, Li et al. 2023 CVPR

- Textual Inversion — Gal, Alaluf et al. 2023 ICLR

- Custom Diffusion — Kumari, Zhang et al. 2023 CVPR

- HyperDreamBooth — Ruiz, Li et al. 2024 CVPR

- ControlNet — Zhang, Rao, Agrawala 2023 ICCV

- T2I-Adapter — Mou, Wang et al. 2024 AAAI

- IP-Adapter — Ye, Zhang et al. 2023 (arXiv 2308.06721)

- InstantID — Wang, Bai et al. 2024 (arXiv 2401.07519)

- PuLID — Guo, Wu et al. 2024 NeurIPS

- PhotoMaker — Li, Cao et al. 2024 CVPR

- SDEdit — Meng, He et al. 2022 ICLR

- InstructPix2Pix — Brooks, Holynski, Efros 2023 CVPR

- Prompt-to-Prompt — Hertz, Mokady et al. 2023 ICLR

- SDXL — Podell, English et al. 2024 ICLR

- SD3 — Esser, Kulal et al. 2024 ICML

- FLUX.1 — Black Forest Labs 2024 (technical report)

- PixArt-α / Σ — Chen, Yu et al. 2024 ICLR / ECCV

- Hunyuan-DiT — Zhimin Li, Jianwei Zhang et al. 2024 (arXiv 2405.08748)

- Imagen — Saharia, Chan et al. 2022 NeurIPS

- ADD / SDXL-Turbo — Sauer, Lorenz et al. 2024 ECCV (arXiv 2311.17042)

- LCM — Luo, Tan et al. 2023 (arXiv 2310.04378)

- LCM-LoRA — Luo, Tan et al. 2023 (arXiv 2311.05556)

- DMD — Yin, Gharbi et al. 2024 CVPR

- ImageReward — Xu, Liu et al. 2023 NeurIPS

- HPSv2 — Wu, Hao et al. 2023 (arXiv 2306.09341)

- PickScore / Pick-a-Pic — Kirstain, Polyak et al. 2023 NeurIPS

- FID — Heusel, Ramsauer et al. 2017 NeurIPS

### One-sentence summary

This cheat sheet covers latent diffusion mathematics (VAE + DDPM/RF + CFG) through mainstream architecture evolution (SD 1.x → SDXL → SD3 → FLUX), conditioning systems (ControlNet / T2I-Adapter / IP-Adapter / InstantID), personalization (DreamBooth / Textual Inversion / LoRA / Custom Diffusion), editing (SDEdit / InstructPix2Pix / Prompt-to-Prompt), distillation (LCM / ADD / DMD), and evaluation (FID / CLIP-Score / ImageReward / HPSv2). 25 questions are split L1/L2/L3; L3 emphasizes the production-lab viewpoint (zero-conv chain-rule derivation, size conditioning training effect, MM-DiT information flow, LoRA Q/K/V choice, SDXL + LCM-LoRA + ControlNet + IP-Adapter co-deployment trade-offs).

---

## 📜 Runnable Code

Minimal runnable PyTorch implementations of the MM-DiT / latent diffusion components in this tutorial live in [`docs/tutorials/code/`](code/):

- [`mmdit_block.py`](code/mmdit_block.py) — double-stream MMDiT block: text + image independent Q/K/V, concat along the seq dim and run one joint attention, per-stream FFN, AdaLN-Zero gated modulation
- [`toy_mmdit_t2i_pipeline.py`](code/toy_mmdit_t2i_pipeline.py) — end-to-end toy T2I pipeline: toy text encoder + 8× 16-channel VAE + 4-layer MMDiT + FlowMatchEuler scheduler + norm-preserving true CFG

Each script runs on CPU in seconds with built-in `assert` sanity checks (shape parity, AdaLN-Zero identity verification, CFG scale=1 equivalence test). Full overview in [`code/README.md`](code/README.md).
