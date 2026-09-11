## §0 TL;DR Cheat Sheet

> 💡 **One sentence** — world-model interview questions are never "whose world model is stronger"; they are three mechanism questions: **what information the model keeps, how it predicts the consequence of an action, and how those predictions enter a decision**. The three technical lineages give three sets of answers, and they overlap each other.

1. **Three lineages**: (1) latent dynamics / model-based RL — learn a compact state space you can roll out inside; (2) predict-in-representation-space (JEPA) — predict only the encoder's output, never reconstruct pixels; (3) generative video and interactive simulators — generate the interactive future observation itself.
2. **Lineages are not camps**: Dreamer 4 is both a generative dynamics model and imagination RL; Cosmos 3 spans understanding, generation and action; video diffusion models also work inside a VAE latent — "latent" is not a word owned by RSSM or JEPA.
3. **State**: the environment's true state $x_t$ is unobservable; the model state $s_t=(h_t,z_t)$ is its stand-in. The RSSM recursion is written $h_t=f_\theta(h_{t-1},z_{t-1},a_{t-1})$ throughout: $h_t$ supplies memory across time, $z_t$ carries stochasticity and new information.
4. **prior vs posterior**: $p_\theta(z_t\mid h_t)$ is used for prediction, $q_\phi(z_t\mid h_t,o_t)$ absorbs the current observation. **Only the prior may be used inside imagination** — calling the posterior means peeking at an observation that has not happened yet.
5. **KL balancing**: $\mathcal L_{KL}=\beta_{dyn}D_{KL}(\mathrm{sg}\,q\Vert p)+\beta_{rep}D_{KL}(q\Vert\mathrm{sg}\,p)$. Both terms are in the $q\Vert p$ direction; what is split is **whose gradient is stopped**, not the KL direction, and certainly not "turn the coefficient down".
6. **Four decision interfaces**: PlaNet plans online in latent space; Dreamer trains an actor-critic inside imagination; MuZero uses MCTS and reconstructs no observation at all; TD-MPC2 learns dynamics, reward and value decoder-free for MPC to use.
7. **The shared JEPA structure**: $P_\theta(E_\phi(x),m)\to\mathrm{sg}(E_{\bar\phi}(y))$ — an asymmetric predictor + an EMA target branch + stop-gradient. This combination **is not a no-collapse theorem**: a constant encoder with a constant predictor is still a degenerate solution.
8. **What matters on the generative route is not image quality but the action interface**: a latent action model's small codebook, camera pose, text conditioning and actuator commands mean completely different things; action-free video supplies only a prior — to use it for action selection you still have to add the interface and the evidence that goes with it.
9. **Numbers come with provenance**: Genie 3's 720p / 24 FPS / few-minutes consistency all come from the official blog post of 2025-08-05 — no paper, no published parameter count (11B is Genie 1's); GameNGen's speed is reported per version — the v1 abstract says ">20 FPS", the v2 body gives **20 FPS on a single TPU-v5 with 4-step DDIM** — and 29.4 dB is the PSNR of **next-frame prediction**, not the fidelity of a long rollout, and still less evidence of physical correctness.
10. **Killer questions**: why KL balancing is not the same as shrinking the coefficient, what exactly is zero in V-JEPA 2-AC's zero-shot, and whether photorealistic video can prove an understanding of physics — Physics-IQ gives an experimental answer (across models $r=-0.46$, $p=0.249$, no significant correlation detected), not a verdict that the two are unrelated.

---

## §1 One question, three lineages

### 1.1　What a world model is

A world model is an agent's **internally learned model of the environment's structure and dynamics**: given the observations and actions so far, it can predict "if I do this, what happens next". The definition deliberately leaves the output form open, because **which output is necessary depends on the use** — to do RL inside the model you need a way to compute returns in imagination and to truncate episodes (the Dreamer family does this by learning a reward head and a continuation head; when the reward function or the termination rule is known, using the known one is equally fine); for visual goal-directed navigation you only need to compare distances between goal representations; to serve as a playable simulator you must generate observations a human can look at.

So "does it count as a world model" is mostly a fake question. The real ones are three: **what information is kept, which quantity is predicted, and which kind of decision that quantity can support**. This tutorial is organised around those three.

### 1.2　Three lineages (they overlap)

| Lineage | What it predicts | Typical decision interface | Representative work |
| --- | --- | --- | --- |
| latent dynamics / MBRL | transitions of a compact latent state, usually with reward and continuation | latent-space planning, actor-critic inside imagination, MCTS | PlaNet, the Dreamer family, MuZero, TD-MPC2 |
| predict-in-representation-space | the representation output by a frozen or EMA encoder | MPC in representation space (minimise the distance to the goal representation) | I-JEPA, the V-JEPA family, DINO-WM |
| generative video / interactive simulator | the future observation itself (pixels or video tokens) | action-conditioned generation + a human or policy interacting in the loop | the Genie family, GameNGen, Cosmos, Matrix-Game |

> ⚠️ **Classify by "which quantity is predicted", not by camp** — Dreamer 4 is both a transformer generative dynamics model and imagination RL; Cosmos 3 is a single model spanning language, image, video, audio and action; video diffusion models generate inside a VAE latent, which shares only its name with RSSM's latent. Classifying by camp contradicts itself immediately; classifying by "which quantity is predicted and how it enters a decision" does not.

### 1.3　Selection criterion for this tutorial

"This tutorial gives priority to work that establishes a key mechanism, represents a major technical route, or was released by a major research group and is closely tied to embodied decision-making; at equal representativeness, open weights come first. The newest releases are used to indicate direction, and their maturity is stated separately."

The work that was excluded and the reason for each entry are in §A.2 — when an interview asks about a name this tutorial does not cover, knowing why it was excluded is more useful than memorising one more name.

### 1.4　Term disambiguation (where interviews most often talk past each other)

- **The environment model of RL**: $p(s_{t+1}, r_{t+1} \mid s_t, a_t)$, in service of planning and imagination. Lineage one here.
- **The world model in LeCun's sense**: prediction in an abstract representation space, explicitly refusing pixel-level reconstruction. His position paper was published on OpenReview (v0.9.2, 2022-06-27) and has **no arXiv number** — don't invent one when citing it. Lineage two here.
- **"World simulator"**: a generative model that can produce interactive video conditioned on actions. Lineage three here.
- **The "world model" inside an LLM**: whether a language model's activations encode external state such as a chess board or a map. That asks "is it in the representation", not "can it predict the consequence of an action". A different question, not covered here.
- **"World model" in the sense of code execution**: a model that predicts program execution state. This is a **terminology collision** with the present topic, not the same line of work.

Two more pairs of symbols and scopes that must be kept apart:

- **$x_t$ vs $s_t$**: $x_t$ is the environment's true state, unobservable, and the model makes no promise to recover it; $s_t$ is the model's own state. Saying "the latent state is the true state" is wrong.
- **Static representation vs interactive dynamics**: I-JEPA is a static image representation method and **does not by itself constitute an interactive dynamics model**; calling it "the image version of a world model" collapses under the first follow-up.

---

## §2 Convention (used throughout)

### 2.1　Time convention

One timeline throughout: **receive $o_t, r_t, c_t$ → form $s_t$ → execute $a_t$ → receive $o_{t+1}, r_{t+1}, c_{t+1}$**.

So $r_{t+1}$ is the immediate reward of action $a_t$, and $c_{t+1}\in\{0,1\}$ flags "the episode has not ended at $t+1$". Index shifts of one between papers almost always come from a different version of this convention — **align the convention first, then compare formula coefficients**.

### 2.2　Symbol table

| Symbol | Meaning | Note |
| --- | --- | --- |
| $x_t$ | the environment's true state | unobservable; the model makes no promise to recover it |
| $o_t$ | observation (image, proprioception, …) | partial observability is the norm |
| $a_t$ | the action executed after $s_t$ | its semantics change with the interface (see §5.1) |
| $r_{t+1}, c_{t+1}$ | reward and continuation | indexed as in §2.1 |
| $s_t=(h_t,z_t)$ | model state | RSSM's deterministic + stochastic halves |
| $p_\theta(z_t\mid h_t)$ | prior | no observation; used for prediction and imagination |
| $q_\phi(z_t\mid h_t,o_t)$ | posterior | absorbs the observation; **must not** be called inside imagination |
| $E_\phi, E_{\bar\phi}, P_\theta$ | online encoder, EMA target encoder, predictor | the JEPA trio |
| $\mathrm{sg}[\cdot]$ | stop-gradient | applies to a computation path, not to freezing a network |

### 2.3　Prediction heads: which ones are necessary

A Dreamer-style world model usually carries three heads: **observation** (reconstruction or token prediction), **reward** and **continuation**. The latter two exist so that "RL inside the model" can run — **in the Dreamer implementation described here**, returns in imagination come from the learned reward head and episode truncation from the learned continuation head; remove either one and the computation no longer closes.

That necessity holds only for this implementation; don't promote it into a general law. When the reward function or the termination rule is known (which is true of many simulators and games), calling the known one computes returns and truncates episodes in imagination just as well. Observation reconstruction is even further from necessary for "RL inside the model" — MuZero and TD-MPC2 both plan without reconstructing observations (§3.8).

A pure video generation model **does not necessarily need reward and continuation**: its purpose is to generate a future you can look at. Treating "no reward head" as a defect, or "has a reward head" as the definition of a world model, both give wrong answers.

### 2.4　Number provenance

Every external number in this tutorial states, at its first appearance in the body, the **task, metric, model version and setting**; speed numbers name the hardware; improvements distinguish **percentage points** from **relative percentages**. Openness is reported per artefact — paper, code, weights and data are four different things, and **a planned release is not written as already open**. Claims supported only by an official blog post or press release are labelled "blog-only" throughout.

---

## §3 Lineage one: latent dynamics and model-based RL

### 3.1　The starting point: training a policy in a dream

Ha & Schmidhuber 2018 (*World Models*, arXiv 1803.10122) split the structure into three pieces: a VAE compresses each frame into a low-dimensional $z$; an MDN-RNN predicts the next-step mixture-of-Gaussians distribution over the $z$ sequence; and a **linear** controller reads an action out of $(z,h)$. The key step is the last one — in the paper's **VizDoom** experiment the controller is trained entirely inside rollouts generated by the RNN (the paper calls them "dream"), then put back into the real environment; **the CarRacing experiment is not like this**, its controller is trained directly in the real environment. "The policy is trained entirely in a dream" is true only of the former.

Those three pieces set the division of labour that followed: **representation** (compress the observation into something predictable), **dynamics** (extrapolate on that quantity), and **decision** (read an action out of the state). Later work either changes one of them or merges them.

### 3.2　RSSM: why the state is half deterministic, half stochastic

PlaNet (Hafner et al., arXiv 1811.04551) introduced the Recurrent State-Space Model:

$$h_t=f_\theta(h_{t-1},z_{t-1},a_{t-1}),\qquad z_t\sim p_\theta(z_t\mid h_t)\ \text{(prior)},\qquad z_t\sim q_\phi(z_t\mid h_t,o_t)\ \text{(posterior)}$$

$h_t$ is **deterministic**: it provides a recurrent memory channel that does not pass through random resampling. A purely stochastic state has to be resampled at every step, and over a long horizon the information is washed out by sampling noise. Don't read "deterministic" as "lossless" — a fixed-width GRU state still compresses and forgets; it is merely no longer scoured by per-step sampling noise. Nor does this say other architectures cannot have memory — a transformer looks back at history directly through attention, so memory exists there too, it is just not squeezed into one recurrent vector.

$z_t$ is **stochastic**: the environment itself is stochastic and observations are incomplete, so $z_t$ carries "the genuinely new information at this step". Don't take the opposite extreme either — deterministic dynamics can work very well in a deterministic environment; the value of $z_t$ rises with environment stochasticity and with the degree of partial observability.

The division of labour between prior and posterior is the core of the whole lineage: **the posterior sees the observation, the prior does not**. Both exist during training; at rollout time there is no future observation, so only the prior is left.

### 3.3　The objective: start from the negative ELBO with no coefficients

$$\mathcal L_t=\underbrace{-\log p_\theta(o_t\mid h_t,z_t)}_{\text{reconstruction NLL}}+\underbrace{D_{KL}\big(q_\phi(z_t\mid h_t,o_t)\,\Vert\,p_\theta(z_t\mid h_t)\big)}_{\text{consistency term}},\qquad z_t\sim q_\phi(z_t\mid h_t,o_t)$$

This is the **single-step, single-sample estimate** of the standard sequential-VAE negative ELBO: the reconstruction term should be $\mathbb E_{z_t\sim q_\phi}[\cdot]$, and one $z_t$ sample stands in for the expectation; the full objective also sums over $t$ along the whole sequence. Written out it is $\sum_t \mathbb E_{q}\left[\mathcal L_t\right]$, **with no weights whatsoever**. It reads as two sentences: the decoder must be able to restore the observation from $(h_t,z_t)$; and the prior must be able to guess what the posterior guessed, without seeing the observation.

Every KL weight, KL balancing and free-bits trick that follows is an **engineering change** to the relative influence of two objectives during optimisation — not a new probabilistic derivation.

> ⚠️ **The KL term presses on two parties** — the consistency term acts on $q$ and $p$ at once. With a single coefficient attached, the optimiser gets a cheap way out: let the posterior degenerate into the prior, the KL goes to zero, and reconstruction rides entirely on $h_t$ — posterior collapse. Toy experiment 1 in §9 reproduces this collapse in an analytically solvable setting.

### 3.4　KL balancing and free bits

DreamerV2 (Hafner et al., arXiv 2010.02193) splits the KL in two:

$$\mathcal L_{KL}=\beta_{dyn}D_{KL}(\mathrm{sg}\,q\Vert p)+\beta_{rep}D_{KL}(q\Vert\mathrm{sg}\,p)$$

**Both terms are in the $q\Vert p$ direction.** Balancing does not touch the KL direction; it decides where the gradient goes: the first term stops $q$'s gradient so only the prior chases the posterior (dynamics learning); the second stops $p$'s gradient so only the posterior concedes to the prior (representation regularisation). Taking $\beta_{dyn}\gg\beta_{rep}$ says: **let the dynamics learn to predict first, then let the representation give ground for the sake of predictability**.

`sg` targets the **computation path**. $q$ and $p$ share $h_t$, and $h_t$'s gradient still flows back along the other path — in code this is a new distribution built with `probs.detach()`, not some block of the network set to `requires_grad_(False)`.

**free bits** is a different thing: $\max(\tau, D_{KL})$ flattens the KL at a floor $\tau$, so once the KL is small enough it is not pushed further and the spare capacity goes to reconstruction. It applies to the **joint KL** — sum over all latent groups first, then truncate; it is not a truncation per categorical probability.

DreamerV3 paper v1 (Hafner et al. 2023, arXiv 2301.04104, Eq. 4 and Table W.1) takes $\beta_{dyn}=0.5$, $\beta_{rep}=0.1$; the v2 body of the same paper writes $\beta_{dyn}=1$, $\beta_{rep}=0.1$. This tutorial uses v1 throughout — **when quoting coefficients, always say which version**.

### 3.5　Where actions come from: online planning vs an actor-critic inside imagination

PlaNet trains no policy network. At every step it plans online **in latent space** using **CEM**: sample a batch of action sequences from a Gaussian, roll them forward with the prior, score them with the reward head, take the elite sequences with the highest returns and **refit** the distribution to them, and iterate for several rounds; finally execute the first action of the **final round's distribution mean**, then start over at the next step. Note it is **not** "sample once and execute the best sample" — iterative refitting is the substance of CEM. Planning never decodes an image, so the sample count can be set very large.

PlaNet also has **latent overshooting**: instead of only pulling the one-step prior close to the posterior, it pulls **multi-step** unrolled priors close to the posteriors at the corresponding times. It is a **multi-step consistency signal**, not "make the training horizon longer" — a model that is accurate one step at a time can still drift into absurdity after a few dozen compositions, and overshooting penalises that drift directly.

Dreamer (Hafner et al., arXiv 1912.01603) changed the decision interface: **train an actor-critic inside imagination**. Take a real segment from the replay buffer, use the posterior to get a starting point $s_t$, then unroll $H$ steps forward with the prior only, and update policy and value on those imagined states. Policy updates no longer consume real interaction, so sample efficiency rises sharply; the price is that **the policy has only ever seen what the model believes will happen**.

The λ-return, written under the time convention of §2.1:

$$G_t^\lambda=\hat r_{t+1}+\gamma\hat c_{t+1}\big[(1-\lambda)v(s_{t+1})+\lambda G_{t+1}^\lambda\big],\qquad G_H^\lambda=v(s_H)$$

The critic's regression target takes a stop-gradient. DreamerV3 v1 Table W.1 takes $H=15$ while the body's prediction sequence length is $T=16$: **this tutorial always counts action transitions by $H$, so the state sequence including the start point has $H+1$ states**. That difference of 1 is not a typo, it is "counting states versus counting transitions".

> ⚠️ **Write the actor gradient per version, don't fuse them into one formula** — Dreamer uses **pathwise gradients**: the return is differentiable with respect to the action, and the gradient travels back to the actor through the learned dynamics. DreamerV3's actor loss is of the **score-function (REINFORCE-style)** form, with return normalization and an entropy regulariser. Also, "freeze the world-model parameters when updating actor/critic" means not changing the dynamics model's parameters; it is **not** the same as putting the whole imagination under `no_grad()` — a pathwise gradient only exists if it can pass through the dynamics graph.

### 3.6　Discrete latents and DreamerV3's three pieces of engineering

DreamerV2 changed $z_t$ from a Gaussian to **a set of categorical distributions** (several groups, each a one-hot over several classes), sampled in the forward pass and approximated with straight-through in the backward pass. That is a **biased** gradient estimate — it passes gradients through a non-differentiable sampling operation as if it were the identity.

Mixing **1% of a uniform distribution** into the categorical probabilities is a **DreamerV3** change (v1 body and Appendix C), not a DreamerV2 one — don't cross the versions when quoting; its purpose is to stop some class probability being squeezed to 0 and blowing up the $\log$. **Uniform mixing and free bits are two independent mechanisms**: one prevents probability degeneration, the other prevents the KL from being pressed too hard.

Besides that uniform mixing, DreamerV3 adds three more mutually independent pieces of engineering:

1. **symlog**: $\mathrm{symlog}(x)=\mathrm{sign}(x)\log(1+\lvert x\rvert)$, inverted by symexp. It compresses signals spanning orders of magnitude into a trainable range, symmetrically and approximately as the identity near 0.
2. **twohot**: encode a continuous scalar onto a fixed set of bins — find which two neighbouring bins it falls between and split a weight of 1 between them by distance. Regression thus becomes classification, and the expected value is recovered exactly.
3. **return normalization**: take the **difference between the 95th and the 5th percentile** $S$ of a batch of returns, smooth it with an EMA, and divide by $\max(1,S)$. The floor of 1 is the crux: in sparse-reward tasks almost all returns are 0, so $S$ is tiny, and without the floor the noise of near-zero returns would be amplified into enormous gradients. **This is not ordinary advantage normalisation** (subtract the mean, divide by the standard deviation) — it only rescales, it has a floor, and it is smoothed across batches.

In v1 (the body after Eq. 10 and Appendix C), **both reward and critic use symlog twohot**: symlog compresses first, then twohot encodes into a categorical regression. **Do not generalise this to every prediction head** — the observation and continuation heads are another matter.

The most frequently misquoted thing about DreamerV3 is its headline sentence: **one set of hyperparameters** covering more than 150 tasks. That says there is no need to tune per environment; it does **not** say one set of weights handles every environment (each task is still trained separately). Collecting a diamond from scratch in Minecraft was achieved **through interaction with the environment**, not from offline data.

### 3.7　Dreamer 4: after the backbone is swapped

Dreamer 4 (arXiv 2509.24527) is no longer an RSSM: **tokenizer + transformer dynamics + shortcut forcing**. Don't drape V3's RSSM/KL/categorical-latent description over it; that is a different backbone.

Its most eye-catching conclusion must be memorised together with its qualifiers: the diamond is obtained under the paper's **offline Minecraft setting**, and the training pipeline includes behaviour cloning, reward modelling and RL inside the model; the "real time on a single GPU" the paper reports refers to **interactive inference**, not training.

### 3.8　Planning without reconstruction: MuZero and TD-MPC2

MuZero (Schrittwieser et al., arXiv 1911.08265) learns only **the quantities planning needs**: reward, value and policy prior, with MCTS search. It reconstructs no observation at all and does not require the latent state to recover $x_t$. The core is **predictive sufficiency for planning** — as long as those quantities are predicted accurately inside the search tree, the search can pick the right action. This is a design principle plus an empirical result, **not** a strict value-equivalence theorem.

TD-MPC2 (Hansen et al., arXiv 2310.16828) carries the same idea into continuous control: learn latent dynamics, reward and value **decoder-free** and use them for MPC, with no observation reconstruction. In the paper's scaling experiments, a single 317M-parameter agent covers 80 tasks.

> 💡 **The price of not reconstructing** — with no decoder there is no direct way to visualise "what the model is thinking", and debugging must rely on indirect signals (reward prediction error, planning returns). The generative route is exactly the reverse: you can look, but looking right is not the same as working right (§7.1).

### 3.9　What to do about model error: PETS and MBPO

These two are often mentioned together, yet they do not solve the same problem:

- **PETS** (Chua et al. 2018, arXiv 1805.12114) handles **uncertainty representation**: an ensemble of probabilistic models expresses both environment noise and the uncertainty caused by insufficient data, and trajectory sampling propagates it during planning.
- **MBPO** (Janner et al. 2019, arXiv 1906.08253) handles **error accumulation**: instead of rolling out from scratch, it **branches** very short model rollouts from states in the real data and then does off-policy updates, keeping accumulated model bias within an acceptable range.

In one line: PETS says "I'm not sure", MBPO says "I only dare look a few steps ahead".

### 3.10　What imagination generates: tokens or pixels

IRIS (Micheli et al., arXiv 2209.00588) imagines over token sequences with a discrete tokenizer + transformer; DIAMOND (Alonso et al. 2024, arXiv 2405.12399) uses a **diffusion** world model to generate image-level imagined trajectories, and its experiments support one concrete conclusion: **visual detail affects control returns** — the small objects and fine edges that get compressed away are often exactly what triggers a reward or a collision.

DayDreamer (Wu et al., arXiv 2206.14176) is this lineage's representative on real hardware: in that paper's experiments, a quadruped learned to walk with **about one hour** of real interaction. That number belongs to that experiment; it cannot be read as "model-based RL learns any skill in an hour".

---

## §4 Lineage two: predicting in representation space (JEPA)

### 4.1　The shared structure

LeCun's position paper (OpenReview v0.9.2, 2022-06-27) argues that prediction should not happen on pixels, because most of the pixel detail of the future is **neither predictable nor worth predicting**. The JEPA family turns that argument into a trainable structure:

$$P_\theta\big(E_\phi(x),\,m\big)\ \longrightarrow\ \mathrm{sg}\big[E_{\bar\phi}(y)\big]$$

$x$ is the context, $y$ the target, $m$ the condition describing "which part to predict" (mask tokens, positions, etc.), $E_{\bar\phi}$ is an EMA copy of the online encoder, and the target branch takes a stop-gradient. Papers differ in three places — **how the distance is computed, how the mask is built, how the target is constructed** — discussed separately below; don't generalise any one paper's details into "that's how all JEPAs work".

### 4.2　I-JEPA: representation prediction on static images

I-JEPA (Assran et al. 2023, arXiv 2301.08243) works on a single image: take several **target blocks** from the image plus a context block, and use an asymmetric predictor to predict each target block's representation from the context representation. The loss is the **squared $L_2$** between patch representations; the target branch comes from the EMA encoder and is not back-propagated; masking is a **multi-block** strategy (target blocks large enough, context block not overlapping the targets).

> ⚠️ **Don't bolt a VICReg term onto I-JEPA** — it relies on an asymmetric predictor + an EMA target, and contains no variance/covariance regularisers. Attaching VICReg's three-term loss to I-JEPA is a common crossed wire. Also, I-JEPA is a **static image representation method**: it takes no action input, predicts no temporal evolution, and **does not by itself constitute an interactive dynamics model**.

### 4.3　V-JEPA: moving to video

V-JEPA (Bardes et al. 2024, arXiv 2404.08471) moves the same structure to video: masked feature prediction, with the distance switched to **$L_1$**, still with stop-gradient + EMA target + predictor. It **needs no negative samples, does no pixel reconstruction, and requires no pre-trained image encoder** — those three "no"s are its position relative to contrastive learning and reconstruction-based self-supervision.

Note the prediction here is still **action-free**: the model learns to **predict masked spatio-temporal representations** (mask prediction, where masked blocks may lie before, inside or after the context in time), not to "predict the future causally given the past"; and there is certainly no "what I did" in it.

### 4.4　What actually prevents collapse

The standard answer is a set of mechanisms: the asymmetric structure (a predictor on one side only), the slow EMA target branch, stop-gradient, and the difficulty of the masking task itself. But the point is the next sentence:

> ⚠️ **Stop-gradient and EMA are not a no-collapse theorem** — a constant encoder with a constant predictor is still a degenerate solution with very low loss. These mechanisms make the training dynamics **tend to** avoid collapse, and the effect depends on initialisation, the EMA coefficient, mask difficulty and optimiser settings. Calling it "theoretically guaranteed not to collapse" in an interview is wrong; "in practice it relies on these few things plus monitoring representation variance" is right. The toy in §9.2 shows it: with shared weights and no stop-gradient, representation variance decays geometrically to 0; with the stop-gradient EMA target, the gradient vanishes at the symmetric point, so collapse is no longer an attractor — but starting from a collapsed initialisation it **also stays on the collapsed solution**.

### 4.5　V-JEPA 2 and 2-AC: action enters in the second stage

V-JEPA 2 (Assran et al. 2025, arXiv 2506.09985) is still pre-trained with **action-free masked prediction** ($L_1$, EMA target, stop-gradient). The **77.3** it is so often cited for is the **top-1 accuracy of an attentive probe** on Something-Something v2 — a representation-quality metric, **not a robot success rate**.

Action enters in the second stage: **V-JEPA 2-AC** freezes the pre-trained encoder and trains only an **action-conditioned predictor** whose input includes robot actions and end-effector state. The paper's "<62 h unlabeled robot video" means **no task-semantic labels**, not no action records — an action-conditioned predictor cannot be trained without actions.

Its planning interface is **MPC in representation space**: given a goal image, encode it into a goal representation; predict the representation reached after executing each candidate action sequence and minimise the **$L_1$ distance** to the goal representation; search action sequences with CEM, execute the first action and replan. **The goal representation is not a reward label** — it is one end of a distance being minimised, with no return regression anywhere.

> ⚠️ **What exactly is zero in zero-shot** — it refers to the **new deployment environment**: on a Franka arm, doing reaching / grasp / pick-and-place from a visual goal, **without re-collecting data in the deployment lab**. It does not prove "any robot, any task, any long-horizon decomposition needs no data collection". Say all three qualifiers when you answer.

### 4.6　V-JEPA 2.1: what changed, and how to break down the gains

V-JEPA 2.1 (arXiv 2603.14482) has three increments: apply prediction supervision to **both masked tokens and visible-context tokens** (the context term carries its own weight); use **multi-layer outputs for deep self-supervision**; and extend the image/video tokenizer and the model scale (the smaller sizes involve distillation).

The robotics gains must be reported broken apart: in the paper's Table 6, under the **same planning setting**, grasp goes from 60% to 70% (**+10 percentage points**); after changing the planning setting and lengthening the horizon it reaches 80% (**+20 percentage points**, at which point it is **no longer a pure representation comparison**). 10 tasks per skill. Blending the two tiers into "a 20% improvement" is the classic provenance accident — it mixes percentage points with relative percentages and mixes two settings at once.

### 4.7　DINO-WM and one common misattribution

DINO-WM (Zhou et al. 2024, arXiv 2411.04983) learns dynamics on **frozen DINOv2 features**, and plans towards the features of a goal image. What it demonstrates is: **the representation can be off the shelf, and the dynamics is the part you have to learn** — there is no need to retrain an encoder just to build a world model.

Navigation World Models (Bar et al. 2024, arXiv 2412.03572) is often filed under JEPA, but it **does not belong here**: it is a **conditional diffusion transformer** over first-person video and actions, and what it predicts is the **future observation**, which puts it in lineage three (see §5.2). Both do "goal-conditioned navigation planning", but they predict different quantities — the most decisive of the three questions in §1.1.

### 4.8　Theory progress and its boundary

A 2026 paper on JEPA generalisation (arXiv 2606.27014) connects pre-training error to the regret of downstream planning, in a setting of **low-rank factorisation** of a conditional spectral graph / action-conditioned co-occurrence matrix. The conclusion holds only under those assumptions — it is **not an unconditional guarantee that "small representation loss implies good planning"**. Cite it together with its setting, or you are substituting a paper's title for an argument.

VL-JEPA (arXiv 2512.10942) brings language into the same framework; this tutorial mentions it directionally only and cites none of its numbers.

### 4.9　What this lineage is missing

Read against lineage one, the gap is clear — **confined to the planning instance described here** (2-AC's representation-space MPC): **no reward, no continuation, no explicit uncertainty**, with the decision resting entirely on one quantity, the distance to the goal representation. That is a property of that interface, not a law of the whole lineage: reward or uncertainty estimation can perfectly well be attached on top of the same representation.

It is therefore naturally suited to **goal-conditioned** tasks (move the gripper to this visual goal), and when long-horizon returns or risk aversion are needed you have to supply the missing quantities yourself. Don't count "sparse reward" among the exclusions either — sparse-reward tasks can often be written in goal-conditioned form, which makes goal-conditioned MPC the convenient choice.

It also inherits an evaluation gap: probe accuracy measures whether the information is in the representation, **not whether the consequences of actions are predicted correctly**. So papers on this line must give both kinds of evidence — representation metrics and closed-loop success rates; giving only the former shows only that the representation is good.

---

## §5 Lineage three: generative and interactive world models

### 5.1　The action interface: the real technical core of this lineage

Video generation models carry no action by nature. **How to plug "what I did" into them** is the most valuable stretch of this lineage.

**latent action model (LAM)**: **infer** an "action" backwards from adjacent observations. The example here is the Genie-style **discrete VQ** LAM (DreamGen uses the continuous pre-quantisation embedding as a pseudo-action, see §6). The VQ bottleneck does three things:

$$k=\arg\min_j\big\lVert E(o_t,o_{t+1})-e_j\big\rVert^2$$

The encoder sees $(o_t,o_{t+1})$ and emits a continuous vector, which is quantised to the nearest code $e_k$ in the codebook; the decoder predicts the future observation from **the current information and $e_k$**; the training loss has three terms — a prediction term, a codebook term and a commitment term (straight-through is the standard VQ formulation, code in §9.3).

Why must the codebook be small? A **capacity argument**: one code carries at most $\log_2 K$ bits, which is 3 bits at $K=8$. Narrow the bottleneck until it only fits "the few directions that best explain the change between frames", and the model has no room left to stuff a whole frame's appearance difference through it.

> ⚠️ **A small codebook does not guarantee that what is learned is an action** — small capacity only guarantees "little information survives", not "what survives is the actuator action", and not that the same code means the same thing across scenes. arXiv 2506.15691 gives the mechanism: **action-induced change and exogenous change compete for the same bottleneck capacity**, and high-variance irrelevant change (lighting, camera shake, background motion) may be encoded first; in the linear case this connects directly to PCA keeping the maximum-variance directions.

Genie (Bruce et al. 2024, arXiv 2402.15391) is the first paper to scale a LAM into a product: **11B** parameters, three components — a video tokenizer, a latent action model with **8 codes**, and a dynamics model; generation uses **MaskGIT-style parallel masked sampling**, not GPT-style token-by-token decoding. Keep those three components separate; fused into "Genie is a video model", you cannot explain why it is interactive.

Besides LAM there are three other interfaces, **semantically completely different and not interchangeable**:

| Interface | What goes in | Which decisions it can support |
| --- | --- | --- |
| text conditioning | scene/event description | content generation, scene construction; no correspondence to an actuator |
| camera motion / pose | viewpoint trajectory | viewpoint control, spatial consistency; not "what the body did" |
| actuator action | joint / end-effector / steering commands | directly corresponds to executable control |
| LAM code | a discrete code inferred backwards from video | interactive, but the semantics need separate verification |

> ⚠️ **Action-free video supplies only a prior** — it can learn "how the world usually evolves", which is valuable. But to use it to **select actions** you must add an action interface and provide decision evidence under that interface. Turning "the video prediction is good" directly into "it can control a robot" is this lineage's most common over-inference.

### 5.2　Real-time interaction and memory

**GameNGen** (Valevski et al. 2024, arXiv 2408.14837) shows a diffusion model can serve as an interactive engine, and both the speed and the fidelity number must be memorised together with their scope: the speed is written "**over 20 FPS**" in the v1 abstract, while the v2 body gives **20 FPS on a single TPU-v5 with 4-step DDIM sampling**; the **29.4 dB PSNR** is a **next-frame prediction** metric measuring single-step prediction, **not the fidelity of a long rollout**. The point is that these two numbers measure **the speed and fidelity of simulating the game DOOM**, not physical correctness — DOOM's "physics" is itself a set of game rules.

**Genie 3** (Google DeepMind official blog, 2025-08-05, **blog-only**): **720p, 24 FPS**, with consistency officially described as on the order of "**a few minutes**" and visual memory of about **one minute**. There is **no paper and no published parameter count** — 11B belongs to Genie 1, don't graft it onto Genie 3, and don't use Genie 1's three-component structure to infer Genie 3's internals either.

**Project Genie** (official blog, 2026-01-29, **blog-only**) is a consumer-facing prototype; the 60-second world length it announces is **a limit of the product prototype**, not a research ceiling for Genie 3.

**Matrix-Game 3.0** (arXiv 2604.08995) puts the weight on three pieces of engineering: action control, **memory retrieval**, and streaming generation. The speed scope must be complete: a **5B** model reaching at most **40 FPS @ 720p** under **asynchronous deployment**, where that deployment uses **8 GPUs running the DiT + 1 doing VAE decoding**. Writing it as "40 FPS on one GPU" is wrong.

**Navigation World Models** (arXiv 2412.03572) trains a conditional diffusion transformer on first-person video and actions, and uses it to evaluate candidate trajectories for goal-directed navigation — an example of a generative model acting directly as a navigation planner.

> 💡 **Real time and memory pull against each other** — interaction demands a fixed compute budget per frame, while remembering a room you walked through ten seconds ago demands that information cross a long time span. The most direct approach — keep the full history verbatim and process it every frame — has a compute cost that grows with time; a fixed-size state or an external memory does not need the per-frame context to grow without bound, but it hands the choice of "what to remember" to training or to a retrieval mechanism. The three engineering compromises are: truncate the context (fast, but when you turn back the room **may have** changed — truncation only raises the risk of that inconsistency, it does not guarantee it happens), compress the history into a fixed-size state (fast and cheap, but training decides what is compressed), or retrieve historical segments explicitly (Matrix-Game 3.0's memory-retrieval route, at the cost of computing the retrieval). When you see "consistency maintained for N minutes", first ask which of these it is, and whether anybody actually turned around and looked back within those N minutes.

### 5.3　Multimodality and spatial conditioning

**UniSim** (Yang et al., arXiv 2310.06114) proposes the "universal action-conditioned simulator" route: unify data from different sources (human activity video, robot data, navigation data and so on) into an "observation–action–observation" conditional generation problem, and train an interactive video simulator.

**Cosmos** is a family, and **cannot be compressed into one architecture**:

- **Cosmos WFM** (NVIDIA 2025, arXiv 2501.03575) gives both an **autoregressive** and a **diffusion** path, and treats **Predict / Transfer / Reason** as **different tasks** rather than three modes of one model.
- **Cosmos-Reason1** (arXiv 2503.15558) is a **vision-language model** for physical common sense and embodied reasoning; it **cannot** be used as evidence of a "dynamics model" — it answers questions, it does not predict the consequences of actions.
- **Cosmos-Predict2.5** (arXiv 2511.00062) announces 2B / 14B configurations; it **has a paper** and does not belong in the "blog-only" tier; these configurations are **not inherited** by Cosmos 3.
- **Cosmos 3** (arXiv 2606.02800) is omnimodal: language, image, video, audio, action. Its **mixture-of-transformers** means **every layer holds both a set of autoregressive reasoner parameters and a set of diffusion generator parameters** — it is **not** a sparse-FFN MoE, and **not** "one expert per modality". Per its paper §2.5 and the HF model cards, the sizes are **Edge 4B / Nano 16B / Super 64B**, **all three released** (Edge on 2026-07-20), under the **OpenMDW-1.1** licence.

**Atlas** (World Labs official blog, 2026-09-01, **blog-only**) has to be described at the level of its interfaces: autoregressive diffusion conditioned on **text, image, camera pose, depth and spatial context**; the blog shows results of **up to one minute at 1440p**. Demo length is **not** evidence of real-time speed, and "camera controllable", "sparse reconstruction" and "robot contact dynamics" are three different kinds of evidence that cannot substitute for one another.

### 5.4　Driving, and Sora as a contrast

For driving this tutorial keeps only the **task difference**: **GAIA-2** (Wayve 2025, arXiv 2503.20523) makes one concrete point — adding **action, road layout and driving conditions** as conditions gets much closer to the controllability that driving decisions need than unconditional video generation does. **GAIA-3** currently has information only at the press-release level (**blog-only**), so this tutorial mentions it only at the level of its release.

Sora / Sora 2 appear here as a **contrast**, not a protagonist: their public material is high-level reporting, and their goal is the quality and usability of general video generation, which is not the same problem as the "consequences of actions and decisions" this cheat sheet is about (reasons for exclusion in §A.2). Equating "video generator" with "world model" is exactly the starting point of the definition fight in §7.4.

---

## §6 Three roads in embodied AI

For a generative model to actually help a robot, it must cross the same gap: **what it generates is pixels, what the policy needs is actions**. There are three clear roads today.

**Road one: video as the plan, actions by inference.** UniPi (Du et al. 2023, arXiv 2302.00111) first generates the task as a video of "how it should be done", then converts the video into executable actions with something like inverse dynamics. The plan lives in pixel space, the landing happens in action space.

**Road two: generate data, then supply action labels.** DreamGen (NVIDIA 2025, arXiv 2505.12705) augments training data with generated video, but generated video has no action labels — it recovers **pseudo-actions with a LAM or an inverse dynamics model (IDM)** (the LAM route takes the continuous pre-quantisation embedding, not the discrete code), turning "video" into "policy training data". That step is the necessary bridge; without it, synthetic video is of no direct use for policy training.

**Road three: the model itself is the environment.** Genie Envisioner (arXiv 2508.05635) connects the world model to policy learning and simulated evaluation, so the model is both data source and evaluation arena. **SIMA 2** (Google DeepMind 2025, arXiv 2512.04797) is an **agent** acting inside Genie-generated worlds — it shows "you can play inside a generated world", which **does not mean** real-world transfer has been established.

| Road | What the model produces | Where actions come from | Main risk |
| --- | --- | --- | --- |
| video as the plan | a video of "how it should be done" | inverse dynamics from adjacent frames | the actions in the video are not physically executable |
| generate training data | large amounts of synthetic trajectory video | pseudo-actions recovered by a LAM or IDM | pseudo-actions are noisy and the error enters the policy |
| the model as environment | an interactive world | the policy explores inside the model itself | the policy exploits model error and it fails on real hardware |

> ⚠️ **None of the three roads gets around real data, but the "real data" they need is not the same kind** — road one's inverse dynamics needs **real observation–action pairs** to train. Road two's two sources of pseudo-actions fall on opposite sides (DreamGen §2.3 distinguishes them explicitly): **IDM depends on action supervision**, that is, real observation–action pairs; **a LAM can be trained from action-free real video alone**. Both need real data, but "needs real video" and "needs real observation–action pairs" are thresholds of substantially different magnitudes — don't merge them into one sentence. Whether the codes a LAM learns line up with actuator actions is **a separate question** (the semantic alignment of §5.1), and it cannot be used to deny that a LAM needs no action annotation. Road three's evaluation still has to return to real hardware in the end. "Replace real data with a world model" is a direction, not a conclusion, today.

---

## §7 Evaluation and the definition fight

### 7.1　Can photorealistic video prove an understanding of physics

**Physics-IQ**'s design (Motamed et al. 2025, arXiv 2501.09038) must be remembered precisely: **66 physical scenes, 396 videos** filmed for real; the model is given a **conditioning frame or a conditioning video** (depending on whether the model under test is image-to-video or video continuation) and asked to generate **five seconds** of continuation, which is then compared with the real continuation; the metrics are **spatial IoU, spatio-temporal IoU, weighted spatial IoU and MSE**. It measures **conditional video continuation**, **not** physics question answering.

It also measures **visual realism** by "can an MLLM tell real video from generated video". Across models, the correlation between realism and the physics metrics is $r=-0.46$, $p=0.249$ — **no significant correlation was detected with this set of models and this sample size**.

> ⚠️ **This is an experimental conclusion and cannot be promoted into "visual quality and physical understanding are unrelated"** — no significant correlation detected ≠ proof of independence; the number of models is small and the correlation coefficient is itself moderately negative. The correct statement is: **in this set of measurements from Physics-IQ, looking realistic did not buy an advantage on the physics metrics**. Saying that in an interview is far safer than reciting "the two are unrelated".

### 7.2　Two benchmarks, two questions

**WorldModelBench** (arXiv 2502.20694) measures **instruction following** plus detection of **physics/common-sense** violations — whether the generated content follows the instruction and whether it breaks basic common sense. **WorldScore** (arXiv 2504.00983) measures **controllability, quality and dynamics**. They cannot substitute for each other: the former asks "is it obedient, does it make common-sense errors", the latter asks "is it controllable, does it look good, does it move correctly".

### 7.3　What the two newer benchmarks each add

**WorldArena 2.0**'s (arXiv 2605.17912) increment is **visuo-tactile signals, policy optimisation inside the model, and a real robot platform** — pushing evaluation from "watch the video" to "optimise a policy inside the model and verify it on real hardware". **WorldRoamBench**'s (arXiv 2606.31672) increment is stability **during interaction**: whether action following, visual consistency, physical plausibility and memory retention can all be maintained over a long interaction.

Both papers reach conclusions of the form "no model meets the requirements across the board". **That conclusion is confined to the models and protocols each of them tested**, and is not a universal judgement about the whole field.

### 7.4　The definition fight: several proposals, not a consensus

- **GLP critique** (arXiv 2507.05169) criticises the route of treating generative video models directly as general-purpose world models, and argues that predictive capability and decision utility must be argued separately.
- **Definition & Roadmap** (arXiv 2607.06401) offers a definition and a roadmap proposal whose taxonomy uses a different set of roles (Renderer / Simulator / Planner, etc.). **It is not the Predictor / Simulator / Evolver one** — the easiest misattribution in this field.
- **L0–L7 position** (arXiv 2606.15032) proposes a grading over **several crossing axes**, not levels along a single line.
- **Agentic World Modeling** (arXiv 2604.22748) discusses world models inside an agent's decision loop, caring about "which step of the decision the model serves"; **"Predictor / Simulator / Evolver" comes from this paper**.

> ⚠️ **Don't mix the grading schemes, and don't treat them as consensus** — L0–L7 (2606.15032) is a multi-axis grading, Predictor/Simulator/Evolver comes from Agentic World Modeling (2604.22748), and Definition & Roadmap (2607.06401) is yet another division; **all of them are the authors' positions, not a consensus of the field**. The L1/L2/L3 in §10 of this tutorial denote only **interview question difficulty** and have nothing to do with any of those gradings.

### 7.5　Two things to say fairly

**The value of the generative route**: visible detail may directly determine contact, collision and reward — DIAMOND is the experimental support for that argument. Also, diffusion generation is **not** "predicting one average future with an MSE"; it models the conditional distribution, and a sample gives one concrete, sharp future rather than a blurred superposition of several.

**The value of the representation-prediction route**: ignoring unpredictable detail can save a great deal of compute and concentrates prediction on quantities that are useful downstream. But it has a hard boundary in the other direction — **a small representation loss does not mean the consequences of actions are predicted correctly**. Distance in representation space is defined by the model itself, and there is no free equivalence between it and "will this action knock the cup over".

---

## §8 The three lineages side by side

| Dimension | latent dynamics / MBRL | predict-in-representation-space | generative / interactive |
| --- | --- | --- | --- |
| Quantity predicted | latent state transitions (mostly with reward and continuation) | representations of an EMA/frozen encoder | future observations (pixels or video tokens) |
| Memory | recurrent $h_t$ (or a transformer context) | context representation + action-conditioned predictor | context window, explicit memory retrieval |
| Action interface | environment actions, present by nature | pre-training carries no action; **the V-JEPA 2-AC pipeline** puts action in a second stage (freeze the encoder, train an action-conditioned predictor), which is not a rule of the lineage | must be designed on purpose: LAM code / camera pose / actuator / text |
| Decision interface | latent-space planning, imagination actor-critic, MCTS | MPC in representation space (distance to the goal representation) | a human or a policy interacting in the loop; landed through action inference |
| Main loss | reconstruction NLL + KL (or regression on planning quantities with no reconstruction) | representation distance ($L_1$ / squared $L_2$) | generative loss (diffusion / token prediction) |
| Typical evidence | task return, sample efficiency | probe accuracy + closed-loop success rate | video quality, controllability, FPS, consistency duration |
| Main failure mode | the policy exploits model error; posterior collapse | representation collapse; representation distance decoupled from task utility | long-horizon drift, memory loss, realistic but physically wrong |
| Overlap example | Dreamer 4 is transformer generative dynamics + imagination RL | DINO-WM learns dynamics on frozen representations, which is a lineage-one move | Cosmos 3 is a generative model with reasoning ability, spanning language/image/video/audio/action |

---

## §9 Code Patterns

Three core patterns; the full experiments are in [`code/world_models_toy.py`](code/world_models_toy.py). All three share `import torch, torch.nn as nn, torch.nn.functional as F`.

### 9.1　One RSSM step + balanced KL

```python
def cat_kl(probs_q, probs_p, eps=1e-8):
    """KL(q||p) for grouped categorical distributions, probs: [..., G, C].
    Sum over classes first, then over groups = the joint KL -- free bits truncates this quantity."""
    return (probs_q * ((probs_q + eps).log() - (probs_p + eps).log())).sum(-1).sum(-1)

class RSSM(nn.Module):
    def __init__(self, deter=512, groups=32, classes=32, a_dim=6, emb_dim=1024, hid=256):
        super().__init__()
        self.groups, self.classes = groups, classes
        z_dim = groups * classes
        self.cell = nn.GRUCell(z_dim + a_dim, deter)          # h_t = f(h_{t-1}, z_{t-1}, a_{t-1})
        self.prior_head = nn.Sequential(nn.Linear(deter, hid), nn.SiLU(), nn.Linear(hid, z_dim))
        self.post_head = nn.Sequential(nn.Linear(deter + emb_dim, hid), nn.SiLU(), nn.Linear(hid, z_dim))

    def _probs(self, logits, uniform_mix=0.01):
        logits = logits.unflatten(-1, (self.groups, self.classes))
        return (1 - uniform_mix) * F.softmax(logits, -1) + uniform_mix / self.classes   # 1% uniform mixing

    @staticmethod
    def _sample(probs):
        idx = torch.multinomial(probs.reshape(-1, probs.shape[-1]), 1).view(probs.shape[:-1])
        z = F.one_hot(idx, probs.shape[-1]).float()
        return z + probs - probs.detach()                     # straight-through: biased but differentiable

    def step(self, h_prev, z_prev, a_prev, emb_t=None):
        """emb_t=None is imagination: no observation, so only the prior is available."""
        h = self.cell(torch.cat([z_prev.flatten(-2), a_prev], -1), h_prev)
        prior = self._probs(self.prior_head(h))
        if emb_t is None:
            return h, self._sample(prior), prior, None
        post = self._probs(self.post_head(torch.cat([h, emb_t], -1)))
        return h, self._sample(post), prior, post

def balanced_kl(post, prior, beta_dyn=0.5, beta_rep=0.1, free_nats=1.0):
    """betas from DreamerV3 paper v1 Eq. 4 / Table W.1; the 32x32 categorical latent and free_nats=1 also follow the paper (Table W.1, Eq. 5)."""
    kl_dyn = cat_kl(post.detach(), prior)                     # sg(q)||p: only the prior chases the posterior
    kl_rep = cat_kl(post, prior.detach())                     # q||sg(p): only the posterior concedes
    return beta_dyn * kl_dyn.clamp(min=free_nats) + beta_rep * kl_rep.clamp(min=free_nats)
```

`post.detach()` stops the gradient **on that path**; $h_t$ is shared by the two heads, and its gradient still comes back along the other one.

In this code $\beta_{dyn},\beta_{rep}$ (v1 Eq. 4 / Table W.1), the 32-group × 32-class categorical latent (Table W.1) and `free_nats=1.0` (the 1 nat of Eq. 5) all follow the DreamerV3 paper; **using `GRUCell` as the recurrent cell and the network widths are choices made here for readability**, not the only implementation the paper prescribes.

### 9.2　JEPA: asymmetric predictor + EMA target

```python
@torch.no_grad()
def ema_update(target, online, m=0.999):
    for p_t, p_o in zip(target.parameters(), online.parameters()):
        p_t.mul_(m).add_(p_o.detach(), alpha=1 - m)           # the target branch moves only by EMA

def jepa_loss(enc, enc_ema, pred, x_ctx, y_tgt, mask_tok, p=1):
    """p=2 -> I-JEPA's squared L2 on patch representations; p=1 -> V-JEPA / V-JEPA 2's L1."""
    ctx = enc(x_ctx)                                          # online branch, has gradients
    with torch.no_grad():
        tgt = enc_ema(y_tgt)                                  # stop-gradient: the target is not pulled toward the prediction
    out = pred(ctx, mask_tok)                                 # asymmetric: the predictor is on this side only
    return (out - tgt).abs().mean() if p == 1 else (out - tgt).pow(2).mean()

def latent_mpc_cost(pred_ac, s0, action_seq, z_goal):
    """V-JEPA 2-AC's planning interface: the encoder is frozen, only the action-conditioned predictor rolls."""
    s = s0
    for a in action_seq.unbind(1):                            # action_seq: [B, H, a_dim]
        s = pred_ac(s, a)
    return (s - z_goal).abs().flatten(1).mean(1)              # CEM ranks by this, replan after the first action
```

### 9.3　The VQ bottleneck of a latent action model

```python
class LatentActionModel(nn.Module):
    def __init__(self, obs_dim, d=32, K=8):                   # Genie 1 uses 8 codes
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(2 * obs_dim, 128), nn.SiLU(), nn.Linear(128, d))
        self.codebook = nn.Embedding(K, d)
        self.dec = nn.Sequential(nn.Linear(obs_dim + d, 128), nn.SiLU(), nn.Linear(128, obs_dim))

    def forward(self, o_t, o_next, beta_commit=0.25):
        e = self.enc(torch.cat([o_t, o_next], -1))            # E(o_t, o_{t+1}): the encoder can see the future
        d2 = (e.pow(2).sum(-1, keepdim=True) - 2 * e @ self.codebook.weight.t()
              + self.codebook.weight.pow(2).sum(-1))          # ||E(.) - e_j||^2
        k = d2.argmin(-1)
        e_k = self.codebook(k)
        e_st = e + (e_k - e).detach()                         # straight-through: the standard VQ formulation
        pred = self.dec(torch.cat([o_t, e_st], -1))           # the decoder gets only current information + the code
        loss = (F.mse_loss(pred, o_next)                      # prediction term
                + F.mse_loss(e_k, e.detach())                 # codebook term
                + beta_commit * F.mse_loss(e, e_k.detach()))  # commitment term
        return pred, k, loss
```

The bottleneck's entire job is to **limit the amount of information**: at $K=8$ one code is at most 3 bits. The decoder only gets $o_t$ and $e_k$, so $e_k$ has to carry "the bit of the inter-frame change most worth keeping" — but whether that is an **action**, the code does not guarantee (§5.1). The `beta_commit=0.25` here is the common empirical value from VQ-VAE, **copied here as a demonstration**; Genie prescribes no such number.

---

## §10 25 high-frequency interview questions

### L1 essentials (10)

<details>

<summary>Q1. What is a world model?</summary>

An agent's internally learned model of the environment's structure and dynamics: given the observations and actions so far, predict "what happens after I do this". **Which output is necessary depends on the use** — to do RL inside the model you must be able to compute returns in imagination and truncate episodes (Dreamer-style implementations learn a reward head and a continuation head for this; when the reward function or the termination rule is known, using the known one is fine); for visual goal-directed navigation you only need to compare distances between goal representations; to be a playable simulator you must generate observations a human can look at. Put the three mechanism questions on the table when you answer: what information is kept, which quantity is predicted, how that quantity enters a decision.

Saying only "a model that can predict the future", or equating it directly with a video generator, scores nothing.

</details>

<details>

<summary>Q2. What are RSSM's prior and posterior? Which is used in imagination?</summary>

The recursion is $h_t=f_\theta(h_{t-1},z_{t-1},a_{t-1})$. The prior $p_\theta(z_t\mid h_t)$ cannot see the current observation; the posterior $q_\phi(z_t\mid h_t,o_t)$ absorbs it. Both are present in training: reconstruction goes through the posterior, and the KL pulls the prior toward the posterior. **In imagination there is no future observation, so only the prior can be used** — calling the posterior smuggles in an observation that has not happened, and the imagined return becomes untrustworthy.

</details>

<details>

<summary>Q3. How do PlaNet and Dreamer select actions?</summary>

PlaNet trains no policy network: at every step it plans online in latent space with **CEM** — sample a batch of action sequences from a Gaussian, roll forward with the prior, score with the reward head, refit the distribution to the elite sequences, iterate a few rounds, then execute the first action of the **final round's distribution mean** and start over at the next step, never decoding an image. Saying "sample once and execute the highest-scoring sample" drops CEM's iterative refitting. Dreamer trains an actor-critic inside imagination: take a real starting point with the posterior, unroll $H$ steps with the prior only, update policy and value on the imagined states, and at deployment the policy emits an action in one forward pass.

The difference in one line: **PlaNet spends compute at decision time, Dreamer spends it at training time.**

</details>

<details>

<summary>Q4. Video looks photorealistic — does that prove the model understands physics?</summary>

Physics-IQ (arXiv 2501.09038) gives an experimental answer: 66 physical scenes and 396 videos filmed for real; the model is given a conditioning frame or a conditioning video (depending on whether it is image-to-video or video continuation) and asked to generate five seconds of continuation, with spatial IoU, spatio-temporal IoU, weighted spatial IoU and MSE as metrics. It also measures visual realism by "can an MLLM tell real from generated", and the across-model correlation is $r=-0.46$, $p=0.249$ — **no significant correlation was detected with this set of models and this sample size**.

Saying "research proves visual quality and physical understanding are unrelated" is wrong: no significant correlation detected is not proof of independence. The correct phrasing is "in this set of measurements from Physics-IQ, realism did not buy an advantage on the physics metrics".

</details>

<details>

<summary>Q5. What are $h_t$ and $z_t$ each responsible for?</summary>

$h_t$ is deterministic and provides **a recurrent memory channel that does not pass through random resampling**; $z_t$ is stochastic and expresses environment stochasticity and the new information at that step. Don't call deterministic "lossless" — a fixed-width GRU state also compresses and forgets, it is merely not scoured by per-step sampling noise. Don't over-generalise in either direction: **a recurrent vector is not the only way to model memory** (a transformer looks back at history directly through attention), and **it is not the case that no deterministic world model can work** (in a deterministic environment it can work very well).

</details>

<details>

<summary>Q6. Which prediction heads does a Dreamer-style model have? Does a pure video model need them?</summary>

Three: observation, reward, continuation. The latter two exist so that "RL inside the model" works: **in this Dreamer implementation**, returns in imagination come from the learned reward head and episode truncation from the learned continuation head, and the computation does not close without either. **A pure video generation model does not necessarily need them** — its purpose is to generate a watchable, interactive future.

But that necessity ends with this implementation. When the reward function or the termination rule is known (as in a great many simulators and games), using the known one does RL in imagination just as well; and observation reconstruction is even less necessary — MuZero and TD-MPC2 both plan without reconstructing observations. Treating "has a reward head or not" as the defining line of a world model collapses at the first follow-up.

</details>

<details>

<summary>Q7. How big is Genie 3? How does it relate to Genie 1's 11B?</summary>

Genie 3 has **no paper and no published parameter count**; 720p, 24 FPS, "few-minutes" consistency and roughly one minute of visual memory all come from the official blog post of 2025-08-05. **11B belongs to Genie 1** (arXiv 2402.15391), and its three components (video tokenizer, an 8-code latent action model, dynamics model) likewise cannot be used to infer Genie 3's internal structure. Project Genie's 60 seconds (2026-01-29 blog) is **a consumer prototype's limit**, not a research ceiling for Genie 3.

Quoting 11B for Genie 3 is a hard error.

</details>

<details>

<summary>Q8. Is the "latent" in the three lineages the same thing?</summary>

No. RSSM's latent is a state variable with a prior and a posterior, in service of rolling out inside it; JEPA's latent is the representation output by the encoder, with no generative prior; a video diffusion model's latent is the generation space after VAE compression. The only thing they share is that **the computation does not happen on pixels**. Separating the lineages by "who works in a latent" misfiles video models under JEPA.

</details>

<details>

<summary>Q9. Can a model pre-trained on action-free video be used directly to select actions?</summary>

Not directly. Action-free video supplies a prior on "how the world usually evolves", which is valuable. To use it for **action selection** you must add an action interface and provide decision evidence under that interface — text conditioning, camera motion, actuator actions and LAM codes mean completely different things and cannot substitute for one another. V-JEPA 2 does exactly this in two stages: action-free pre-training, then a second stage that trains an action-conditioned predictor on data with actions.

</details>

<details>

<summary>Q10. What does DreamerV3's "one set of hyperparameters" mean? Where did its diamond come from?</summary>

It means **one set of hyperparameters** covering more than 150 tasks, i.e. no need to tune per environment; it does **not** mean one set of weights handles every environment — each task is still trained separately. Collecting a diamond from scratch in Minecraft was done **through interaction with the environment**. Dreamer 4's (arXiv 2509.24527) "diamond offline" is another paper and another setting; the two must not be conflated.

</details>

### L2 advanced (10)

<details>

<summary>Q11. Is KL balancing just turning the KL coefficient down?</summary>

No. $\mathcal L_{KL}=\beta_{dyn}D_{KL}(\mathrm{sg}\,q\Vert p)+\beta_{rep}D_{KL}(q\Vert\mathrm{sg}\,p)$: **both terms are in the $q\Vert p$ direction**, and what is split is whose gradient is stopped. $\beta_{dyn}\gg\beta_{rep}$ means "let the prior chase the posterior first (learn the dynamics), then let the posterior concede for the sake of predictability (regularise the representation)". Shrinking a single coefficient merely loosens the consistency term as a whole and produces a different imbalance: either the posterior degenerates and reconstruction rides on $h_t$, or the prior never catches the posterior and imagination drifts as soon as it unrolls. Report coefficients per version: DreamerV3 v1 Eq. 4 / Table W.1 gives $\beta_{dyn}=0.5,\beta_{rep}=0.1$, while the v2 body gives $1$ and $0.1$.

Writing $D_{KL}(p\Vert q)$, or saying balancing changes the KL direction, scores nothing.

</details>

<details>

<summary>Q12. How does JEPA avoid collapse?</summary>

Through a combination of mechanisms: the asymmetric structure (a predictor on one side only), the slow EMA target branch, the stop-gradient on the target branch, and the difficulty of the masking task itself. The key is the next sentence — **this is not a no-collapse theorem**: a constant encoder with a constant predictor is still a degenerate solution with very low loss, the effect depends on initialisation, the EMA coefficient, mask difficulty and optimiser settings, and in practice you monitor representation variance or rank.

The toy matching §9.2 (experiment 2 in §A.3) reproduces this: with shared weights and no stop-gradient, the variance of the context representation decays geometrically; switching to a stop-gradient + EMA teacher makes the gradient vanish at the symmetric point, so collapse is no longer an attractor — but starting from the collapsed point it stays there all the same.

</details>

<details>

<summary>Q13. What exactly does V-JEPA 2-AC's zero-shot refer to?</summary>

It refers to **no data collection in the new deployment environment**: on a Franka arm, doing reaching / grasp / pick-and-place from a visual goal, without re-collecting data in the deployment lab. Structurally, pre-training is still action-free masked prediction ($L_1$, EMA, stop-gradient); 2-AC freezes the encoder and trains only an action-conditioned predictor whose input includes robot actions and end-effector state. The paper's "<62 h unlabeled robot video" means **no task-semantic labels**, not no action records. Planning is MPC in representation space: predict the representation reached after executing an action sequence, minimise the $L_1$ distance to the goal image's representation, search with CEM, and replan after executing the first action — **the goal representation is not a reward label**.

It does not prove that any robot, any task or any long-horizon decomposition is free of data collection.

</details>

<details>

<summary>Q14. If you make the VQ codebook very small, will it automatically discover the true actions?</summary>

No guarantee. The argument for a small codebook is **capacity**: one code carries at most $\log_2 K$ bits ($K=8$ means 3 bits), forcing the model to discard most of the inter-frame difference. But small capacity only guarantees "little information survives"; it **does not guarantee that what survives is the actuator action**, nor that the same code means the same thing across scenes. arXiv 2506.15691 gives the mechanism: action-induced change and exogenous change (lighting, camera shake, background motion) **compete for the same bottleneck capacity**, and high-variance irrelevant change may be encoded first; in the linear case this connects directly to PCA keeping the maximum-variance directions.

</details>

<details>

<summary>Q15. Write the λ-return used in imagination and state the start-point and length conventions.</summary>

Under this tutorial's time convention:

$$G_t^\lambda=\hat r_{t+1}+\gamma\hat c_{t+1}\big[(1-\lambda)v(s_{t+1})+\lambda G_{t+1}^\lambda\big],\qquad G_H^\lambda=v(s_H)$$

The critic's regression target takes a stop-gradient. DreamerV3 v1's Table W.1 takes $H=15$ and the body's prediction sequence length is $T=16$: **count action transitions by $H$, and the state sequence including the start point therefore has $H+1$ states**. When pressed on "is it 15 or 16", answer "15 counting transitions, 16 counting states".

</details>

<details>

<summary>Q16. Are free bits and the 1% uniform mixing of categorical latents the same thing?</summary>

No, two independent mechanisms. **free bits** is $\max(\tau, D_{KL})$, a floor applied to the **joint KL** (summed over all latent groups first), preventing the KL from being pressed so hard that representation capacity is exhausted. **1% uniform mixing** is a **DreamerV3** change (v1 body and Appendix C) that mixes a uniform distribution into the categorical probabilities, preventing some class probability from being squeezed to 0 and blowing up the $\log$ with pathological gradients; it is not a DreamerV2 change — what DreamerV2 brought was the categorical latent itself. One governs the shape of the objective, the other the numerical health of the distribution.

Also, the categorical latent + straight-through is itself a **biased** gradient approximation — it passes gradients through non-differentiable sampling as if it were the identity, which is a third thing.

</details>

<details>

<summary>Q17. What is the division of labour between symlog and twohot?</summary>

**symlog** is scale compression: $\mathrm{symlog}(x)=\mathrm{sign}(x)\log(1+\lvert x\rvert)$, inverted by symexp, symmetric and approximately the identity near 0, used to squeeze signals spanning orders of magnitude into a trainable range. **twohot** is an encoding: split a continuous scalar between the two neighbouring bins by distance, so regression becomes classification and the expected value is recovered exactly.

In DreamerV3 v1 (the body after Eq. 10 and Appendix C), **both reward and critic use symlog twohot** — **do not generalise this to every prediction head**; the observation and continuation heads are another matter.

</details>

<details>

<summary>Q18. How does DreamerV3's return normalization differ from ordinary advantage normalisation?</summary>

It takes the **difference between the 95th and the 5th percentile** $S$ of a batch of returns, smooths it with an EMA, and divides by $\max(1,S)$. Three differences: it only rescales and does not subtract the mean; it **has a floor of 1**; and it is smoothed across batches. The floor is the crux — in sparse-reward tasks almost all returns are 0 and $S$ is tiny, so the ordinary "subtract the mean, divide by the standard deviation" would amplify the noise of near-zero returns into enormous gradients, and $\max(1,S)$ blocks that path.

</details>

<details>

<summary>Q19. Do PETS and MBPO solve the same problem?</summary>

No. **PETS** (arXiv 1805.12114) solves **uncertainty representation**: an ensemble of probabilistic models expresses both environment noise and the uncertainty caused by insufficient data, propagated during planning by trajectory sampling. **MBPO** (arXiv 1906.08253) solves **error accumulation**: branch very short model rollouts from states in the real data and do off-policy updates, keeping the compounding of model bias under control.

In one line: PETS says "I'm not sure", MBPO says "I only dare look a few steps ahead".

</details>

<details>

<summary>Q20. Why can't DreamerV3's description be applied to Dreamer 4? What qualifies its offline conclusion?</summary>

The backbone is different: Dreamer 4 is **tokenizer + transformer dynamics + shortcut forcing**, with no RSSM, KL balancing or categorical latents, so forcing the old description onto it means getting the architecture wrong. Its diamond was obtained under the paper's **offline Minecraft setting**, with a training pipeline that includes behaviour cloning, reward modelling and RL inside the model; the "real time on a single GPU" the paper reports refers to **interactive inference**, not training. DreamerV3's diamond came from interaction with the environment — the two conclusions cannot substitute for each other.

</details>

### L3 top-lab (5)

<details>

<summary>Q21. When should you predict pixels, and when representations?</summary>

The criterion is **which information the decision depends on**, not which route is more advanced.

Reasons to predict pixels: detail may directly determine contact, collision and reward (DIAMOND's experiments support exactly this — the small objects and fine edges that get compressed away are often the triggers); you need a human to watch it, you need an interactive simulator, or you need an observation interface that is general across tasks. **Diffusion generation is not "predicting one average future with an MSE"** — it models the conditional distribution and samples one concrete, sharp future.

Reasons to predict representations: ignoring unpredictable detail saves compute and concentrates prediction on quantities useful downstream; and the downstream task is a distance or MPC in representation space to begin with (DINO-WM learns dynamics on frozen DINOv2 features, V-JEPA 2-AC uses the $L_1$ to a goal representation).

Each has a hard boundary: the pixel route's **looking right is not moving right**; the representation route's **a small representation loss does not mean the consequences of actions are correct** — distance in representation space is defined by the model itself, and there is no free equivalence between it and "will this action knock the cup over".

</details>

<details>

<summary>Q22. MuZero reconstructs no observation — why can it still plan?</summary>

Because planning only uses the few quantities inside the search tree: reward, value and policy prior. The latent state need not be able to recover the environment's true state $x_t$; it only needs to be **predictively sufficient** for those quantities. There are two costs: with no decoder there is no channel for visualising "what the model is thinking", so debugging relies on indirect signals such as reward prediction error and planning returns; and "predictive sufficiency" is a design principle plus an empirical result, **not a strict value-equivalence theorem**.

TD-MPC2 is the same idea in continuous control: learn latent dynamics, reward and value decoder-free for MPC to use; in its scaling experiments a single 317M agent covers 80 tasks.

</details>

<details>

<summary>Q23. Why can't a world model learned offline solve exploration through "unlimited imagination"?</summary>

Because the model is only trustworthy on the state–action distribution the data covers. Imagining more times will not conjure information the data does not contain; worse, policy optimisation will **actively find and exploit trajectories where the model overestimates the return** — the return is enormous inside the model and does not hold up back in the real environment. That is exactly what MBPO's short branched rollouts and PETS's uncertainty propagation are there to suppress.

Two things must be kept apart: **broadening coverage can only come from adding data or new environment interaction**; **pessimism / uncertainty penalties do not broaden coverage** — what they do is constrain the decision back into the region where the model has evidence, so the policy does not gamble on high-return hallucinations in uncovered areas. Calling the penalty term "a second way to broaden coverage" gets the direction backwards.

Dreamer 4's offline result is not a counterexample: it has offline data, behaviour cloning and reward modelling prepared for that setting, and the scope of its conclusion is that setting.

</details>

<details>

<summary>Q24. How do you judge whether a world model is genuinely useful for robots?</summary>

Ask four things. **One, what is the action interface** — actuator actions, camera pose, LAM codes or text conditioning? Only the first corresponds directly to executable control. **Two, what is the decision evidence** — closed-loop success rate, on which platform, on how many tasks, and was the planning setting changed? V-JEPA 2.1's Table 6 is the ready-made example: under the same planning setting, grasp going from 60% to 70% (+10 percentage points) is a representation comparison, while the 80% (+20 percentage points) after changing the planning setting and lengthening the horizon is no longer a pure representation comparison. **Three, where does the error accumulate** — long-horizon consistency, memory retention, contact dynamics. **Four, the cost scope** — speed on what hardware, in what deployment (Matrix-Game 3.0's peak 40 FPS @ 720p is an asynchronous deployment with 8 GPUs running the DiT + 1 doing VAE decoding, not a single GPU).

Video quality scores and representation probe accuracy cannot substitute for the first three questions — V-JEPA 2's frequently cited 77.3 is the attentive-probe top-1 on SSv2, not a robot success rate.

</details>

<details>

<summary>Q25. Given a freshly released world-model report, how do you place it quickly?</summary>

Read it by the three questions first: **which quantity it predicts** (pixels / representations / planning quantities), **what the action interface is**, and **how the prediction enters a decision**. Those three fix the lineage, and they are more reliable than what it calls itself — Dreamer 4 is both generative dynamics and imagination RL, and Cosmos 3 spans understanding, generation and action.

Then check the evidence tier: is there a paper or only a blog post; how far are code, weights and data each open (**planned release does not count as open**); does every number come with the task, metric, model version, setting and hardware.

Finally put it on the evaluation map: WorldModelBench measures instruction following and physics/common-sense violations, WorldScore measures controllability, quality and dynamics, WorldRoamBench measures action following, visual consistency, physical plausibility and memory retention during interaction, and WorldArena 2.0 brings in visuo-tactile signals, policy optimisation inside the model and a real robot platform. They do not ask the same thing, and doing well on one does not vouch for another; and the "no model meets the requirements across the board" conclusions in these benchmarks are confined to the models and protocols they tested.

Be careful with the gradings: L0–L7 (arXiv 2606.15032) is a proposal over **several crossing axes**; Predictor / Simulator / Evolver comes from **Agentic World Modeling (arXiv 2604.22748)**, so don't file it under Definition & Roadmap — the latter (arXiv 2607.06401) is yet another proposal with a different set of roles (Renderer / Simulator / Planner, etc.). **None of these is a consensus of the field**; the GLP critique (arXiv 2507.05169) represents yet another position. The L1/L2/L3 in §10 of this tutorial denote only interview question difficulty.

</details>

---

## §A Appendix

### A.1　Papers and provenance

Evidence tier: **✅ paper** (public on arXiv; the details cited here are taken from the paper) / **⚠️ blog-only** (official blog post or press release only, no paper). A third case is written out separately: **a paper exists but this tutorial mentions it directionally only** (its numbers were not individually verified), and it is not merged into "blog-only".

**Lineage one: latent dynamics / MBRL**

| Work | Provenance | In one line |
| --- | --- | --- |
| World Models | ✅ 1803.10122 | VAE + MDN-RNN + linear controller; in the **VizDoom experiment** the policy is trained entirely in the dream, while CarRacing's controller is trained in the real environment |
| PlaNet | ✅ 1811.04551 | RSSM; latent overshooting; online planning in latent space |
| Dreamer | ✅ 1912.01603 | actor-critic trained in imagination (pathwise gradients) |
| DreamerV2 | ✅ 2010.02193 | categorical latents; KL balancing; human-level Atari from imagination |
| DreamerV3 | ✅ 2301.04104 | symlog / twohot / return normalization; one set of hyperparameters over 150+ tasks; the Minecraft diamond came from environment interaction |
| Dreamer 4 | ✅ 2509.24527 | tokenizer + transformer dynamics + shortcut forcing; diamond under the offline Minecraft setting; single-GPU real-time **inference** |
| MuZero | ✅ 1911.08265 | learns only reward/value/policy quantities, no reconstruction; MCTS |
| TD-MPC2 | ✅ 2310.16828 | decoder-free latent dynamics + reward + value for MPC; 317M single agent / 80 tasks |
| DayDreamer | ✅ 2206.14176 | in that experiment a quadruped learned to walk with about 1 hour of real interaction |
| PETS | ✅ 1805.12114 | probabilistic ensemble + trajectory sampling = uncertainty representation |
| MBPO | ✅ 1906.08253 | short rollouts branched from real data = controlling model bias |
| IRIS | ✅ 2209.00588 | discrete tokenizer + transformer imagination |
| DIAMOND | ✅ 2405.12399 | diffusion world model; visual detail affects control returns |

**Lineage two: predict-in-representation-space**

| Work | Provenance | In one line |
| --- | --- | --- |
| LeCun position paper | OpenReview v0.9.2 (2022-06-27) | prediction in an abstract representation space; **no arXiv number** |
| I-JEPA | ✅ 2301.08243 | asymmetric predictor + EMA target; squared $L_2$ on patch representations; multi-block masking |
| V-JEPA | ✅ 2404.08471 | video masked feature prediction; $L_1$; no negatives, no pixel reconstruction |
| V-JEPA 2 / 2-AC | ✅ 2506.09985 | action-free pre-training; 2-AC freezes the encoder and trains an action-conditioned predictor + representation-space MPC |
| V-JEPA 2.1 | ✅ 2603.14482 | both masked and visible-context tokens supervised; multi-layer deep self-supervision; scaling |
| DINO-WM | ✅ 2411.04983 | learns dynamics on frozen DINOv2 features, plans with goal features |
| JEPA generalisation theory | ✅ 2606.27014 | low-rank factorisation of a conditional spectral graph / action-conditioned co-occurrence matrix → pre-training error and planning regret |
| VL-JEPA | ✅ 2512.10942 | brings language into the same framework; **paper 2512.10942; mentioned directionally only here**, none of its numbers cited |

**Lineage three, embodiment and evaluation**

| Work | Provenance | In one line |
| --- | --- | --- |
| Genie | ✅ 2402.15391 | 11B; video tokenizer + 8-code LAM + dynamics model; MaskGIT-style sampling |
| Genie 3 | ⚠️ blog 2025-08-05 | 720p / 24 FPS / few-minutes consistency / about one minute of visual memory; no paper, no published parameter count |
| Project Genie | ⚠️ blog 2026-01-29 | consumer prototype; the 60 seconds is a product limit |
| GameNGen | ✅ 2408.14837 | v1 abstract ">20 FPS", v2 body 20 FPS (single TPU-v5, 4-step DDIM); 29.4 dB is the **next-frame prediction** PSNR (a DOOM-simulation number) |
| UniSim | ✅ 2310.06114 | the unified action-conditioned interactive simulator route |
| Navigation World Models | ✅ 2412.03572 | conditional diffusion transformer on first-person video + actions |
| Cosmos WFM | ✅ 2501.03575 | autoregressive and diffusion paths; Predict / Transfer / Reason are different tasks |
| Cosmos-Reason1 | ✅ 2503.15558 | a VLM for physical common sense and embodied reasoning, **not** evidence of a dynamics model |
| Cosmos-Predict2.5 | ✅ 2511.00062 | 2B / 14B configurations; **not inherited** by Cosmos 3 |
| Cosmos 3 | ✅ 2606.02800 | omnimodal; per-layer AR reasoner + diffusion generator parameter sets; Edge 4B / Nano 16B / Super 64B |
| GAIA-2 | ✅ 2503.20523 | driving: action, road layout and driving conditions as conditions |
| GAIA-3 | ⚠️ press release only | mentioned at the level of its release only |
| Matrix-Game 3.0 | ✅ 2604.08995 | 5B; asynchronous deployment (8 GPUs DiT + 1 GPU VAE) at up to 40 FPS @ 720p; memory retrieval, streaming generation |
| Atlas | ⚠️ World Labs blog 2026-09-01 | AR diffusion on text/image/camera pose/depth/spatial context; demos up to one minute at 1440p |
| LAM bottleneck analysis | ✅ 2506.15691 | action change and exogenous change compete for bottleneck capacity; linear case related to PCA |
| UniPi | ✅ 2302.00111 | video as the plan, actions inferred by inverse dynamics |
| DreamGen | ✅ 2505.12705 | generated video → LAM/IDM recovers pseudo-actions → policy data |
| Genie Envisioner | ✅ 2508.05635 | connects the world model to policy learning and simulated evaluation |
| SIMA 2 | ✅ 2512.04797 | an agent acting inside Genie-generated worlds; not the same as real-world transfer |
| Physics-IQ | ✅ 2501.09038 | 66 scenes / 396 videos; five-second continuation; IoU and MSE metrics |
| WorldModelBench | ✅ 2502.20694 | instruction following + physics/common-sense violations |
| WorldScore | ✅ 2504.00983 | controllability / quality / dynamics |
| WorldArena 2.0 | ✅ 2605.17912 | visuo-tactile signals, policy optimisation inside the model, real robot platform |
| WorldRoamBench | ✅ 2606.31672 | action following, visual consistency, physical plausibility and memory stability during interaction |
| GLP critique | ✅ 2507.05169 | argues against treating generative video models directly as general-purpose world models |
| Definition & Roadmap | ✅ 2607.06401 | a definition and roadmap proposal; its taxonomy is a different set of roles (Renderer / Simulator / Planner, etc.), **not** Predictor / Simulator / Evolver |
| L0–L7 position | ✅ 2606.15032 | a grading proposal over several crossing axes |
| Agentic World Modeling | ✅ 2604.22748 | puts the world model back in the agent's decision loop; **Predictor / Simulator / Evolver comes from this paper** |

**Openness**: Cosmos 3's Edge, Nano and Super are all released (Edge released 2026-07-20, see its HF model card) under the OpenMDW-1.1 licence; Genie 3, Project Genie, Atlas and GAIA-3 have information only at the official blog level (Cosmos-Predict2.5 **has a paper**, arXiv 2511.00062). For the remaining entries this tutorial asserts only that "the paper is public"; whether code, weights and data are open **is governed by each project's own official page** — this tutorial makes no per-entry assertion, because those four things are frequently out of sync.

### A.2　Exclusion list and reasons

| Excluded | Reason |
| --- | --- |
| Sora / Sora 2 | high-level public reporting only; does not match this tutorial's focus on the consequences of actions and decisions — kept in §5.4 as a contrast |
| Oasis | the mechanism is already represented by GameNGen; researchability and length |
| Hunyuan-GameCraft | the same technical problem is already covered by a representative work |
| Runway GWM / Odyssey | at the level of a release announcement |
| HunyuanWorld | **has a paper** (HunyuanWorld 1.0, arXiv 2507.21809); excluded because its topic leans toward 3D scene generation and roamable world construction, which is not the same question as this tutorial's "how the consequences of actions enter a decision", plus limited length |
| STORM / TransDreamer | mechanisms overlap with Dreamer / IRIS |
| Meta CWM | terminology collision: it refers to "world model" in the code-execution sense |
| Othello-GPT-style world models inside LLMs | asks "is it in the representation", not "can it predict the consequence of an action" |
| Tesla | no paper |

### A.3　Runnable toy

Script: [`code/world_models_toy.py`](code/world_models_toy.py), pure PyTorch, seconds on CPU. All three experiments have analytic answers you can check item by item — **the conclusions cover only their own explicit settings and are not paper reproductions**.

**Experiment 1 | RSSM linear-Gaussian: posterior collapse and balanced KL.** The environment is $x_{t+1}=0.5x_t+a_t+\epsilon$ with $\epsilon$ equally likely to be $\pm1$. The deterministic part $h=0.5x_t+a_t$ is computable exactly, and the residual $\epsilon=o_{t+1}-h$ is the only new information. With the scalar parameterisation $q(z\mid h,o)=\mathcal N(w\epsilon,1)$, $p(z\mid h)=\mathcal N(v,1)$, $\hat o=h+dz$, we get $\mathcal L_{rec}=\tfrac12[(1-dw)^2+d^2]$ and $\mathbb E D_{KL}=\tfrac12(w^2+v^2)$. Comparing a single KL coefficient of 9 with balanced $\beta_{dyn}=9,\beta_{rep}=0.1$ (**toy values, not the paper's**): starting from $w=d=0.5$, $v=1$ and running SGD at lr 0.05 for 1,000 steps, the single-coefficient version converges to $\lvert w\rvert,\lvert d\rvert<10^{-6}$ — posterior collapse, with reconstruction MSE ≈ 1, i.e. the observation is not used at all; the balanced version converges to $v\to0$, $w^2=1/\sqrt{0.1}-1$, $d=w/(1+w^2)$, with MSE $=\sqrt{0.1}$.

**Experiment 2 | JEPA's shortcut and stop-gradient.** With $x=s$, $y=s+\epsilon$, an encoder $E_w(u)=wu$ and an identity predictor: sharing $w$ and **not** stopping the gradient gives $\mathcal L=\tfrac12w^2$, and gradient descent at lr 0.1 gives $w_n=0.9^n$ — starting from $w_0=1$ and running 100 steps, the context representation's variance falls to $0.9^{200}$; collapse is an attractor. Switching to stop-gradient + an EMA teacher $b\leftarrow0.9b+0.1w$: starting from $w=b=1$ the gradient is zero, $w$ stays at 1, and the loss of 0.5 comes entirely from unpredictable noise; but starting from $w=b=0$ it stays put just the same. **This is the executable version of "a mechanism is not a theorem".**

**Experiment 3 | A coordinate LAM on a grid world.** The encoder takes $\Delta=s'-s$, the VQ has 4 codes, and the decoder is additive. After farthest-first initialisation plus one round of nearest-neighbour assignment and centroid update, the four transition vectors are recovered exactly: validation reconstruction error 0, purity 1, $I(K;A)=2$ bits (up to a permutation of the codes). This shows that in **this coordinatised setting with no exogenous change** the bottleneck does capture the action — once high-variance change unrelated to the action is added, the competition described in §5.1 begins.
