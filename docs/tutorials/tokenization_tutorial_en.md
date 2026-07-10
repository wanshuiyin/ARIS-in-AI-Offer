## §0 Pipeline Mental Model + TL;DR Cheat Sheet

**A tokenizer is not "one algorithm" — it is a pipeline.** Most interview confusion ("SentencePiece is Unigram", "byte-level is a segmentation objective") comes from compressing the whole pipeline into a single noun. Set up this mental model first:

```text
(Upstream: the chat template renders messages into text first; registered specials — when the API enables/allows them — are recognized atomically / bypassed before the pipeline below)
raw text
  → ① normalization       (NFC/NFKC/identity; possibly LOSSY!)
  → ② pre-tokenization    (regex chunking / whitespace handling; merges never cross chunk boundaries)
  → ③ segmentation model  (BPE merges by rank / WordPiece longest-match / Unigram Viterbi)
  → ④ ID mapping          (token → integer ID → embedding row)
  → ⑤ post-processing     (auto-inserting template tokens like BOS/EOS/CLS/SEP)
decode: IDs → concatenate token strings → (byte restore / ▁→space) → text (the reverse, but strict invertibility is NOT guaranteed)
```

> 💡 **Tokenization in 9 sentences** — the interview core on one page (derivations in §1–§7).

1. **Pipeline mental model**: chat template and special-token recognition live upstream; the tokenizer proper = normalization → pre-tokenization → segmentation model → ID mapping → (BOS/EOS-style) post-processing; decode is the reverse. **SentencePiece is a framework** (Unigram/BPE are its selectable models), **byte-level is an alphabet choice** (not a standalone algorithm) — these two concept groups are not on the same layer.

2. **Why subword**: word-level suffers OOV + vocabulary explosion; char/byte-level sequences are too long (a BMP Chinese character = 3 bytes, supplementary-plane = 4); subword is the compromise — frequent words stay whole, rare words split into pieces, and with a byte fallback you get **zero OOV on any valid UTF-8 text**. Metrics must state their denominator: **fertility** (tokens/word? tokens/char? tokens/byte?); cross-tokenizer comparison uses **BPB** (bits-per-byte).

3. **BPE: training $\ne$ encoding**. Training = each round count the most frequent adjacent pair in the corpus, merge it, and **record the merge in an ORDERED merge table** (locally greedy, not globally optimal); encoding = repeatedly apply merges by **frozen merge rank** — it neither re-counts frequencies nor does longest-match. This is the single most frequent hand-coding trap.

4. **WordPiece**: merges are chosen by likelihood improvement (the common teaching-implementation score is $= \frac{f(xy)}{f(x)f(y)}$, whose log is PMI-like); encoding is **longest-match-first** (with `##` continuation forms), and a match failure at ANY position turns the **whole word** into `[UNK]` (not "keep the successful prefix").

5. **Unigram subtracts from the top**: start from an oversized seed vocabulary, EM-estimate piece probabilities (the E-step aggregates posterior counts over **all** segmentations via forward-backward, not just the Viterbi one) + prune by deletion loss; Viterbi picks the best segmentation only at inference; sampling segmentations comes natively (subword regularization).

6. **Byte-level and SentencePiece are two engineering routes**: GPT-2 first maps the 256 bytes into a printable proxy alphabet and then runs BPE (`Ġ` = the proxy character of space byte 0x20; it often shows up at token starts because the regex attaches the space to the following chunk); SentencePiece symbolizes whitespace with `▁` (U+2581), and `byte_fallback` only falls back to `<0xXX>` when normal pieces lack coverage — **byte fallback $\ne$ byte-level BPE**.

7. **Vocabulary size is a two-sided ledger**: embedding $\approx Vd$, plus another $Vd$ if the output head is untied; larger V → (usually, other things equal) lower fertility → shorter sequences → cheaper attention/KV cache; but softmax/embedding get more expensive. 32k→100k→128k→200k→256k is a set of **multi-objective design points**, not a monotone timeline; when reporting numbers distinguish mergeable count, special count, `len(tokenizer)`, max-ID+1, and embedding rows — the five quantities are often unequal.

8. **Four production traps to memorize**: **prefix instability** (`encode("ab")` need not be a token-prefix of `encode("abc")` → the shared root of token healing / streaming / prefix-cache issues); **digit and whitespace splitting** affects arithmetic and code; **special-token injection** (whether a literal `<|assistant|>` in user text becomes a special ID depends on the API allowlist); **glitch tokens** (in the vocabulary but under-exposed in training, e.g. SolidGoldMagikarp).

9. **Tokenizer-free frontier**: ByT5 (raw bytes in), MegaByte (fixed byte patches + global/local hierarchy), BLT (entropy-driven dynamic patches); hierarchy mitigates byte-sequence length, but training efficiency, inference ecosystem, and cache granularity remain real obstacles — "the tokenizer is dead" is premature.

## §1 Why Subword + How to Measure a Tokenizer

### 1.1 The word-level vs char-level dilemma

- **word-level**: the vocabulary balloons with the corpus (worse for morphologically rich languages); any new word / typo / rare entity → OOV `[UNK]`, information simply lost; and "run/running/runs" each occupy an embedding row — wasted parameters, no shared statistical strength.
- **char/byte-level**: byte-level is zero-OOV (char-level can still OOV when the character table is incomplete), but sequence length explodes — an English word $\approx$ 4–5 characters, a BMP Chinese character (incl. Extension A) = 3 UTF-8 bytes (supplementary plane, Extension B and later = 4 bytes) — and attention's $O(L^2)$ plus the KV cache's $O(L)$ pay directly; the model also burns precious depth learning "letters spell words", structure it could have gotten for free.
- **The subword compromise**: frequent words stay whole tokens ("the", "人工智能"), rare words split into statistically reusable substrings ("tokenization" → "token"+"ization" — sometimes these coincide with morphemes, but none of the three algorithms guarantees linguistic boundaries), and rare characters drop to bytes. Vocabulary stays bounded, coverage stays guaranteed.

**Coverage (the OOV guarantee) comes in three tiers**, a frequent follow-up:

1. Classic WordPiece (BERT-style): encoding failure → **whole-word `[UNK]`** — weakest;
2. Unigram/SentencePiece: incomplete character coverage (`character_coverage` $\lt 1$) still yields `<unk>`;
3. **A complete 256-byte alphabet** (GPT-2-style) or SentencePiece with **byte fallback correctly enabled**: any valid UTF-8 text is encodable — the **strong coverage guarantee**. Zero OOV is not a default perk of subword; it is bought by these two designs.

### 1.2 Metrics: fertility must state its denominator

**Fertility (segmentation granularity)**: how many tokens the average "unit" splits into. In the literature, narrow-sense fertility usually means tokens/**word** specifically (tokens/char and tokens/byte are more often called compression-ratio metrics); this tutorial uses the broad sense, but you **must declare the denominator** — tokens/**word**, tokens/**char**, or tokens/**byte**? Chinese has no natural whitespace word boundaries, so "tokens/word" depends on an external word segmenter (different standards give different counts). For cross-lingual comparison, prefer reporting all of these **on the same evaluation text**:

| Metric | Definition | Use |
| --- | --- | --- |
| UTF-8 **bytes/token** | byte count of eval text ÷ token count | the most stable cross-tokenizer measuring stick (byte count is tokenizer-independent; cross-lingual comparison still needs parallel corpora + multiple metrics) |
| Unicode **chars/token** | code-point count ÷ token count | intuitive, but emoji/combining characters deceive |
| **tokens/word** | token count ÷ external word count | comparable only once the segmenter is declared |
| UNK rate | share of `<unk>` | coverage check |

Engineering also looks **end-to-end**: total token count on the same corpus (directly drives training/inference cost), throughput, time-to-first-token — tokenizer quality is ultimately a systems question (§4.2).

### 1.3 BPB: a comparable loss metric across tokenizers (given a unified byte stream and scoring protocol)

**Loss/token and perplexity are not directly comparable across tokenizers** — the denominator (token count) itself differs: even with identical total NLL, a different token count alone changes loss/token severalfold; real comparisons additionally mix in genuine model-quality differences, so the two effects are entangled. The fair move is to spread the total negative log-likelihood over the **UTF-8 byte count $B$ of the same text**:

$$\mathrm{BPB} = \frac{-\sum_i \log p(t_i \mid t_{\lt i})}{B \cdot \ln 2}\qquad(\text{divide by}\ \ln 2\ \text{to convert nats to bits}).$$

Numeric example (executable as [D07] in §7): same text, total NLL = 4.0 nats, $B=8$ bytes; tokenizer A cuts 2 tokens, B cuts 4 → loss/token is **2.0 vs 1.0** (a 2× "gap"!), yet BPB is identically $\frac{4.0}{8\ln 2}\approx 0.7213$ — **same BPB; the loss/token "gap" is pure denominator illusion**.

> ⚠️ **BPB's precondition: both sides actually score the SAME post-normalization byte stream**
> If a tokenizer applies lossy normalization (NFKC rewrites characters), the two models are already scoring different text — "feeding the same raw text" is not enough; the normalized byte stream entering each model must be identical, and its byte count is what you use as $B$. BOS/EOS handling, context chunking, and the rest of the scoring protocol must be aligned too. BPB is the common, comparable metric — not an unconditional "only fair one".

## §2 The Three Core Algorithms: BPE / WordPiece / Unigram

**Iron rule of this section: teach the training objective and the encoding algorithm separately.** Each algorithm has a "what training learns" half and a "how encoding uses it" half; blending them is the most common way to lose points.

### 2.1 BPE: iteratively merge the most frequent pair

BPE was originally a 1994 byte-compression algorithm (Gage, 1994); Sennrich, Haddow & Birch (2015, arXiv 1508.07909, ACL 2016) brought it to NMT for subword segmentation, and it became the mainstream choice of the GPT, Llama, and Qwen families.

**Training algorithm** (bottom-up, additive):

1. Split the corpus into initial symbol sequences (characters or bytes) and count each "word"'s frequency;
2. Count the weighted frequency of every **adjacent symbol pair**;
3. Merge the **most frequent** pair into a new symbol and **append that merge, in order, to the merge table**;
4. Repeat 2–3 until the target number of merges / vocabulary size.

Three details to keep on the tip of your tongue:

- **Locally greedy**: each round picks the currently most frequent pair; the final sequence length is **not** guaranteed globally optimal — it is a compression heuristic, not an optimizer;
- **Tie-breaking must be deterministic**: equal counts need a fixed rule (e.g. lexicographic), or the same corpus trains different merge tables and reproducibility collapses;
- **Overlapping counts**: in `"aaaa"` the sliding-window count of pair `(a,a)` is 3, but non-overlapping replacement can only apply twice — keep the counting convention and the replacement semantics separate.

**Worked example**: corpus `{"abab"×3, "abac"×1}`, initial segmentation = characters. Round-1 pair counts:

| pair | occurrences | weighted count |
| --- | --- | --- |
| $(a,b)$ | 2 per "abab" ×3 + 1 per "abac" ×1 | $2\times 3+1=\mathbf{7}$ |
| $(b,a)$ | 1 per "abab" ×3 + 1 per "abac" ×1 | $3+1=4$ |
| $(a,c)$ | 1 per "abac" ×1 | $1$ |

→ **merge₁ $=(a,b)\to$ `ab`** (count 7). Round 2: words become `ab ab`×3, `ab a c`×1; pair counts $(ab,ab)=3,\ (ab,a)=1,\ (a,c)=1$ → merge₂ $=(ab,ab)\to$ `abab`. Merge table = `[(a,b), (ab,ab), ...]` — **order IS rank, and rank IS encoding priority**.

**Encoding algorithm** (a completely different program from training): given the frozen merge table, split the input into initial symbols, then **repeatedly merge the applicable pair with the LOWEST rank (earliest learned), leftmost first**, until no merge applies:

$$\text{encode}(x):\quad \text{while } \exists\ \text{pair} \in \text{merge table}:\ \text{merge the lowest-rank pair (leftmost first)}.$$

Example: merge table `[(a,b), (b,c)]` (ranks 0, 1), `encode("abc")`: both $(a,b)$ and $(b,c)$ apply, and **rank-0 $(a,b)$ wins** → `["ab","c"]` — even if you feel `(b,c)` is "also frequent". There is **no frequency counting at encoding time, period**.

> ⚠️ **The #1 hand-written-BPE trap: writing encode as "merge the currently most frequent pair in the input"**
> Training counts frequencies; encoding only looks up ranks. Get it wrong and the same string encodes differently in different contexts, drifts from the training-time segmentation distribution, and disagrees with official implementations. §7's [D02] pins this semantics with `assert bpe_encode("abc", [("a","b"),("b","c")]) == ["ab","c"]`.

**decode**: in toy semantics, concatenate the token strings (byte-level additionally restores bytes to UTF-8); real systems must also undo boundary markers like `▁`/`@@` and the byte proxy. BPE's decode is still the simplest of the three — merged tokens are naturally substring concatenations.

### 2.2 WordPiece: likelihood-driven merges + longest-match encoding

WordPiece (Schuster & Nakajima, ICASSP 2012; adopted by BERT, Devlin et al., 2018, arXiv 1810.04805) is also bottom-up merging like BPE — **the difference is the pair-selection score**.

**Training objective**: original WordPiece selects merges by **language-model likelihood improvement**; the score used by common teaching implementations (e.g. the HF tokenizers docs) is

$$\mathrm{score}(x,y)=\frac{f(xy)}{f(x)\,f(y)},$$

whose log is $\log f(xy)-\log f(x)-\log f(y)$ — differing from PMI only by a constant that depends on the total-count convention. Calling it **PMI-like** is accurate; saying "it IS PMI" or "this is THE exact definition of every historical WordPiece" is too strong (Google's historical trainer was never fully specified publicly).

**Worked example** (BPE and WordPiece pick different pairs):

| pair | $f(xy)$ | $f(x)$ | $f(y)$ | BPE criterion (count) | WordPiece score |
| --- | --- | --- | --- | --- | --- |
| $(a,b)$ | 6 | 10 | 10 | **6** (BPE picks it) | $6/100=0.06$ |
| $(q,u)$ | 2 | 2 | 10 | 2 | $2/20=\mathbf{0.1}$ (WordPiece picks it) |

Intuition: `q` almost only appears inside `qu` — high normalized "cohesion", so WordPiece picks it in this example; `a` and `b` appear independently in bulk, so `ab`'s high count is just a "large base". **BPE looks at absolute frequency; WordPiece looks at cohesion** (the score is a surrogate — do not read the ratio as total likelihood gain).

**Encoding algorithm**: unrelated to training — greedy **longest-match-first**:

1. From the word start, find the **longest** vocabulary match (word-initial pieces use the plain form; non-initial pieces use the `##` continuation form, e.g. `##ing`);
2. On a match, advance and continue from the remainder;
3. **If ANY position has no match → output `[UNK]` for the whole word** — not "keep the successful prefix and append `[UNK]`".

Example (executable as [D03] in §7): with vocabulary `{"a", "##b", "##bc", "[UNK]"}`, `wp("abc") = ["a", "##bc"]` (`##bc` beats `##b` on length); remove `##bc` without adding `##c`, and `wp("abc") = ["[UNK]"]` — **whole-word failure**.

> 💡 **`##` is the "I am not word-initial" marker**
> WordPiece relies on pre-tokenization to cut "words" at spaces/punctuation first; `##` marks a subword's continuation position within the word, and decode strips it accordingly. For CJK text, the common BERT-pipeline practice is that BasicTokenizer isolates Han ideographs character-by-character (kana/hangul are NOT split per character, and Korean does have spaces) — an implementation strategy, not an algorithmic constraint, but exactly the kind of language-specific preprocessing SentencePiece set out to eliminate (§3.2).

### 2.3 Unigram: a probabilistic model + top-down pruning

Unigram LM segmentation (Kudo, 2018, arXiv 1804.10959, ACL 2018) changes the paradigm: **instead of learning merge rules, learn a "piece probability table"** — segmentation becomes inference under a probabilistic model.

**Model**: every piece $z_i$ in vocabulary $V$ has probability $p(z_i)$, $\sum_{z\in V}p(z)=1$; a segmentation $z=(z_1,\dots,z_k)$ has independent-product probability $P(z)=\prod_i p(z_i)$; **a string's (marginal) probability sums over all valid segmentations**:

$$P(x)=\sum_{z\in S(x)}\prod_{i} p(z_i),\qquad \text{training objective}\ \max_\theta \sum_{x\in\text{corpus}}\log P(x).$$

**Training** (top-down, subtractive — the opposite direction from BPE):

1. Start from an **oversized seed vocabulary** (e.g. frequent substrings + all single characters);
2. **EM for the probabilities**: the E-step uses forward-backward over $S(x)$ (all segmentations) to aggregate each piece's **expected posterior counts** — integrating over ALL segmentations, **not just the Viterbi best one** (that would be hard-EM; getting this direction wrong is a frequent point deduction); the M-step renormalizes $p(z)$;
3. **Prune**: for each piece compute "the approximate corpus-likelihood loss if it were deleted"; delete the lowest-loss batch (e.g. 20%), go back to 2; single characters, specials, and byte pieces are **protected** from deletion;
4. Converge to the target vocabulary size.

**Inference (encoding)**: deterministic segmentation uses **Viterbi** for the best split:

$$z^{*}=\arg\max_{z\in S(x)}\sum_i \log p(z_i).$$

**Worked example** (executable as [D04] in §7): probability table

$$p(\text{a})=0.15,\quad p(\text{b})=0.15,\quad p(\text{ab})=0.40,\quad p(\text{aba})=0.20,\quad p(\text{bab})=0.10.$$

Run both DPs on `"abab"` ($f_i$ = the **all-segmentation** probability sum of prefix $x_{1:i}$; $v_i$ = the best single-segmentation probability):

| $i$ | prefix | forward $f_i=\sum_j f_j\,p(x_{j+1:i})$ | Viterbi $v_i$ | best path at $v_i$ |
| --- | --- | --- | --- | --- |
| 0 | $\varepsilon$ | 1 | 1 | — |
| 1 | `a` | $0.15$ | $0.15$ | `a` |
| 2 | `ab` | $0.40+0.15\times 0.15=0.4225$ | $\max(0.40,\ 0.0225)=0.40$ | `ab` |
| 3 | `aba` | $0.20+0.4225\times 0.15=0.263375$ | $\max(0.20,\ 0.06)=0.20$ | `aba` |
| 4 | `abab` | $0.015+0.169+0.0395\ldots=\mathbf{0.22350625}$ | $\max(0.015,\ \mathbf{0.16},\ 0.03)=0.16$ | `ab`·`ab` |

where $f_4 = f_1\,p(\text{bab}) + f_2\,p(\text{ab}) + f_3\,p(\text{b}) = 0.15\times 0.10 + 0.4225\times 0.40 + 0.263375\times 0.15$. Two readings to memorize: **the Viterbi segmentation is `["ab","ab"]` (probability 0.16)**; and **the marginal 0.2235 $\geq$ the best-path probability** (a sum over all segmentations can never be smaller than a single path — exactly why the E-step aggregates all-segmentation statistics via forward-backward instead of the single Viterbi path).

**Subword regularization vs BPE-dropout** (two randomization mechanisms — don't blend them):

- **Unigram sampling**: when training an LLM, **sample** segmentations from the segmentation posterior (or an n-best approximation) instead of fixing Viterbi, with sampling distribution $q(z)\propto P(z)^{\alpha}$; `alpha` is an **inverse temperature / smoothing coefficient** (larger $\alpha$ → closer to Viterbi, smaller → more random, $T=1/\alpha$) — the same word may segment differently every epoch, acting as data augmentation and exposing the model to fragmented spellings;
- **BPE-dropout** (Provilkov et al., 2019, arXiv 1910.13267, ACL 2020): BPE has no probabilistic model, so instead **randomly skip (block) merges with probability $p$ at encoding time**, producing more fragmented splits. Similar purpose (robustness), **completely different mechanism** (posterior sampling vs merge blocking).

### 2.4 Three-algorithm comparison + coverage

| Dimension | BPE | WordPiece | Unigram |
| --- | --- | --- | --- |
| Training direction | bottom-up **additive** (merge) | bottom-up additive (merge) | top-down **subtractive** (prune) |
| Training objective | most frequent pair (greedy compression) | likelihood gain / PMI-like score | corpus marginal likelihood (EM) |
| Training artifact | **ordered merge table** | vocabulary (with `##` forms) | piece **probability table** |
| Encoding algorithm | merge by frozen rank | longest-match-first | Viterbi (sampleable) |
| OOV behavior | byte-level variant zero-OOV; char-level can OOV | failure → **whole-word `[UNK]`** | incomplete coverage → `<unk>` (byte fallback optional) |
| Randomization | BPE-dropout (block merges) | — (none in standard impls) | native (sample segmentations, `alpha` temperature) |
| Representative models | GPT family / Llama / Qwen | BERT family | T5 / ALBERT (SentencePiece-Unigram) |

> ✅ **The skeleton for answering "how do the three differ"**
> Training first: BPE counts frequencies, WordPiece scores likelihood gain, Unigram EM-plus-prunes from a large vocabulary. Then encoding: rank merging / longest match / Viterbi — **training and encoding are two independent programs**. Close with coverage and randomization (`[UNK]` semantics; Unigram is natively sampleable). Say those three beats and you are already deeper than most candidates.

## §3 Byte-Level and SentencePiece Engineering

### 3.1 GPT-2 byte-level BPE: proxy alphabet + regex pre-tokenization

GPT-2's (Radford et al., 2019) actual order is: **raw text → regex pre-tokenization into chunks → UTF-8-encode each chunk → map through `bytes_to_unicode` into a 256-character printable proxy alphabet → run ordinary BPE within each chunk** (chunk first, THEN byte-map — reversing the order would change the chunking). The reason is pure engineering: running directly on raw bytes leaves you with piles of unprintable/control characters, painful to process and debug; `bytes_to_unicode` keeps printable bytes as themselves and shifts the other 68 bytes to `chr(256+n)`, forming a 256↔256 **bijection** (reversible).

- The mapping is an **implementation bridge** only — it does not mean "every final token holds one byte": after merging, one token can cover many bytes;
- Space byte `0x20` maps to **`Ġ`** (U+0120) and newline `0x0A` to `Ċ` — so the wall of `Ġ` you see when visualizing GPT-family tokens is the proxy spelling of the space byte. `vocab.json` genuinely stores masses of string entries like `Ġthe`, but that `Ġ` is the internal proxy character, not a U+0120 from the original text;
- **Regex pre-tokenization chunks first; BPE merges never cross chunk boundaries.** GPT-2's regex splits by category (key fragments only — the full regex evolves across versions, don't recite it):

```text
's|'t|'re|'ve|'m|'ll|'d    # English contractions get their own chunk
 ?\p{L}+                   # letter runs (optionally with one leading space) — the origin of Ġword
 ?\p{N}+                   # digit runs: unbounded in GPT-2 (the root of the digit trap, §5.1)
 ?[^\s\p{L}\p{N}]+         # punctuation/symbol chunks
\s+(?!\S)|\s+              # whitespace (the lookahead branch affects multi-space attribution; simplified here)
```

A key change in later encodings (cl100k_base / o200k_base / Llama-3-style) is switching the digit branch to **`\p{N}{1,3}` (at most 3 digits per chunk)** — note "a ≤3-digit pre-token" does not mean "final tokens are always 3 digits": BPE can still split within the chunk (`999` → `9`+`99`); it just **cannot merge across chunks**.

### 3.2 SentencePiece: a framework, not an algorithm

**SentencePiece (Kudo & Richardson, 2018, arXiv 1808.06226, EMNLP 2018 demo) is a tokenizer training/encoding framework** whose `model_type` can be **unigram / bpe / char / word** — it is not a synonym of Unigram (when someone says "SentencePiece tokenization" in an interview, proactively clarify which model). Key engineering properties:

- **What "no pre-tokenization required" really means**: it does not require a language-specific word tokenizer first and can train straight from raw sentences (crucial for CJK); but normalization, whitespace escaping, and configurable split rules **still exist** — "no preprocessing at all" is wrong;
- **`▁` (U+2581, LOWER ONE EIGHTH BLOCK)**: promotes the space to an ordinary symbol that participates in modeling; decode maps `▁`→space. Defaults also add a dummy prefix (a `▁` at sentence start) and may fold consecutive whitespace — **these defaults mean you cannot claim lossless invertibility from `▁` alone**;
- **Default normalizer `nmt_nfkc`**: an NFKC-derived rule set (not exactly any runtime's standard NFKC); many modern models explicitly switch to identity or custom rules to preserve the original text;
- **`byte_fallback=true`**: the vocabulary reserves 256 byte pieces of the form `<0xXX>`; spans that normal pieces cannot cover after normalization are encoded as their UTF-8 bytes, and decode reassembles consecutive byte pieces back into characters.

> ⚠️ **byte fallback $\ne$ byte-level BPE**
> SentencePiece's byte fallback is a **safety net**: it only retreats to `<0xXX>` when ordinary Unicode pieces lack coverage; GPT-2-style byte-level BPE is the **foundation**: it learns merges on the full 256-byte alphabet from step one. And byte fallback cannot recover text already rewritten by lossy normalization — coverage and fidelity are two independent properties (§7's [D05] separates them with an NFKC example).

> 💡 **`Ġ` vs `▁`: the two "space stand-ins" in one line**
> `Ġ` is the **proxy character** of byte `0x20` after GPT-2's proxy-alphabet mapping (the byte-level route); `▁` is SentencePiece's U+2581 that **explicitly symbolizes** the space (the framework route). Different mechanisms, same message: whitespace information must be stored explicitly, or decode cannot restore it.

### 3.3 Unicode traps and the precise conditions for "reversible"

**One user-perceived "character" may be several code points and even more bytes** — the ancestor of every offset/length bug:

- **Combining characters**: `é` can be the single code point U+00E9 or `e`+U+0301 (combining accent) — NFC canonically composes the latter into the former (the single-code-point form is unchanged); without normalization, the two spellings produce **different token sequences** under a fully covering, faithful tokenizer;
- **Compatibility characters**: NFKC maps fullwidth `Ａ`→`A`, `①`→`1` (search-friendly but **lossy** — the original spelling is gone forever);
- **Emoji ZWJ sequences**: 👨‍👩‍👧 is 5 code points (3 emoji + 2 U+200D zero-width joiners) = 18 UTF-8 bytes, yet 1 grapheme cluster (one user-perceived "character");
- **Variation selectors** (U+FE0F etc.), zero-width space U+200B, invisible control characters: invisible to the eye yet genuinely affecting the token sequence (provided the normalizer preserves them and the vocabulary covers them; with insufficient coverage they collapse into `<unk>`) — security audits and dedup must check BOTH the raw and normalized layers.

**Conditions for two-way invertibility** (a frequent follow-up):

- `decode(encode(x)) == x` **does not hold by default**. The essential requirement is that **the whole encode→decode pipeline preserves information on the target input domain**: the normalizer/pre-tokenizer loses nothing, no `<unk>` swallows characters (byte alphabet or byte fallback as the net), special/post-processing behavior is symmetric, and the decoder does no cleanup "beautification". The most common sufficient configuration in practice: identity normalization (one NFKC pass ruins it) + whitespace fidelity (dummy prefix / whitespace folding off) + full coverage + decoder cleanup off — note this is a common configuration, not a logical iff (a dummy prefix that the decoder removes symmetrically does not break the round-trip; conversely you additionally need the escaping of internal markers like `▁` / special surfaces to be injective, or a literal `▁` in the original text collides with the space escape);
- `encode(decode(ids)) == ids` is **generally not guaranteed**: multiple non-canonical token sequences decode to the same string (`["a","b"]` and `["ab"]` both restore `"ab"`), and re-encoding returns only the canonical segmentation. This ID-round-trip infidelity (encode∘decode is a canonicalization — usually idempotent under typical normalization configs) and §5.4's prefix instability are related but distinct properties — the latter is the direct root of token healing.

## §4 Training Corpus, Vocabulary Size, Multilingual, and Vocab Extension

### 4.1 The tokenizer's training-corpus design often matters more than "which algorithm"

The same BPE algorithm, fed different corpora, trains tokenizers whose fertility differs severalfold. The knobs you actually design:

- **Language mix**: a language with low share in the tokenizer corpus → fewer merges for it → higher fertility → the same content burns more tokens (double penalty: cost and context window);
- **Code/math share**: decides whether indentation, operators, and common identifiers get whole tokens (§5.2);
- **Dedup and cleaning**: repeated template text pushes meaningless long strings into the vocabulary (glitch-token breeding ground, §5.3);
- **Normalization consistency**: the tokenizer and the model must train with the same normalization, or online/offline segmentation diverges;
- **Trainer knobs differ per implementation**: `character_coverage`/reserved special slots are SentencePiece-style knobs; `min_frequency` is an HF-tokenizers-style BPE/WordPiece trainer knob; byte-level BPE's base coverage is fixed at 256 bytes — don't pin every knob on every trainer.

### 4.2 Vocabulary size: a two-sided ledger

With hidden size $d$ and vocabulary size $V$: **input embedding $\approx Vd$ parameters; an untied output projection adds another $Vd$, totaling $\approx 2Vd$**. Tying saves parameters, but the output-side full softmax **compute** still scales with $Vd$. For scale ($d=4096$): $V=32{,}000$ gives $Vd\approx 130$M parameters; $V=128{,}256$ gives $Vd\approx 530$M — double it if untied.

The other side: **other things equal and with a roughly incremental vocabulary extension, larger V → more merges → usually lower fertility → fewer tokens for the same text** → attention ($\propto L^2$), activations, KV cache ($\propto L$), and generation steps all win. The two sides pull in opposite directions, hence:

> 🎯 **Never judge vocabulary size by a single metric**
> Compare end-to-end: tokens/s, time-to-first-token, memory, and task quality — not just "compression ratio" or just "embedding parameters". 32k→100k→128k→200k→256k is **not a monotone industry timeline** but a family of design points jointly determined by language coverage, code share, model width ($d$ must be large enough to afford a large $V$), and deployment budget.

### 4.3 The mainstream vocab facts table: five numbers not to mix up

**One tokenizer has five easily confused quantities**: mergeable token count, special token count, `len(tokenizer)`, max token ID+1 (holes allowed), and the model's embedding rows (possibly padded larger for alignment). They are **not all equal** in the table below. Two conversions: **valid entries = mergeable count + special count**; tiktoken's `n_vocab` = **max ID + 1** (with ID holes it exceeds the valid-entry count), and HF's `len(tokenizer)` is further affected by added-token semantics:

| Tokenizer | mergeable tokens | special tokens (public) | valid tokenizer entries | model vocab_size (embedding rows) |
| --- | --- | --- | --- | --- |
| bert-base-uncased (WordPiece) | 30,517 (incl. ~1k `[unused]` placeholders) | 5 (`[PAD]` `[UNK]` `[CLS]` `[SEP]` `[MASK]`) | 30,522 | 30,522 |
| bert-base-cased | 28,991 (same convention) | 5 | 28,996 | 28,996 |
| GPT-2 (byte-level BPE) | 50,256 (256 byte symbols + ~50k merges) | 1 (the end-of-text marker) | 50,257 | 50,257 |
| cl100k_base (tiktoken) | 100,256 | 5 | 100,261; **max ID+1 = 100,277 (ID holes)** | — (API models undisclosed) |
| o200k_base (tiktoken) | 199,998 | 2 | 200,000; **max ID+1 = 200,019** | — |
| Llama 2 (SentencePiece BPE) | 31,997 (incl. 256 byte pieces) | 3 meta pieces (`<unk>` UNKNOWN; `<s>` `</s>` CONTROL) | 32,000 | 32,000 |
| Llama 3 (tiktoken-style byte-level BPE) | 128,000 | 256 (reserved special slots) | 128,256 | 128,256 |
| Qwen2 / Qwen2.5 | 151,643 | 3 core chat markers | 151,646 | **151,936** (0.5B–3B) / **152,064** (7B+) — padded per checkpoint size, same rule in both generations |
| Gemma 1 / Gemma 2 (SentencePiece) | — | — | 256,000 (config value) | 256,000 |
| Gemma 3 | — | — | 262,144 ($=2^{18}$, base tokenizer) | 262,208 (4B/12B/27B text config; check per checkpoint for 1B) |

Rows you must be able to expand on (the literal special spellings contain vertical bars, which clash with table delimiters, so they are omitted in cells: GPT-2's sole special is `<|endoftext|>`; Qwen's 3 core ones are `<|endoftext|>`, `<|im_start|>`, `<|im_end|>`):

- **cl100k/o200k's "vocab size" has two answers**: valid entries (100,261 / 200,000) vs max ID+1 (100,277 / 200,019) — the ID space has holes, so `n_vocab` is not "the number of usable tokens". "GPT-4o's vocab is 200,019" and "o200k has 200k valid tokens" describe **different quantities**; saying GPT-4o uses the o200k family of encodings is reasonable, but the public `o200k_base` need not include every hidden server-side chat-protocol detail — freeze and verify the model↔encoding mapping per version;
- **Llama 3 is not an enlarged Llama 2**: Llama 2 uses SentencePiece BPE (32,000); Llama 3 switched to a Meta-trained byte-level BPE loaded through a tiktoken-style implementation (128,000 ordinary + 256 reserved specials = 128,256). "tiktoken-based" describes the **implementation interface and the byte-level BPE form**, not reuse of OpenAI's merge vocabulary — it is not a copy of cl100k_base;
- **Qwen's three sets of numbers**: base vocabulary 151,643; `len(tokenizer)` 151,646 (+3 core specials); embedding rows **151,936 (0.5B–3B) / 152,064 (7B and up)** — embedding rows are padded model-side by checkpoint size, the same rule for Qwen2 and Qwen2.5; defer to each checkpoint's `config.vocab_size`. Multimodal variants (e.g. vision editions) add further special tokens — "what is Qwen's vocab size" must start by fixing the convention;
- **"BERT is 30k" is only approximate**: uncased 30,522 / cased 28,996; **"Gemma is always 256k" is inaccurate**: Gemma 3's base tokenizer is 262,144 ($2^{18}$) while the 4B/12B/27B text configs have 262,208 embedding rows — one more case of "tokenizer size $\ne$ embedding rows".

### 4.4 Multilingual: Chinese is not a "characters vs words" binary

In a modern BPE/Unigram vocabulary, Chinese tokens are simultaneously a mix of **single characters, multi-character words (frequent ones like "我们", "人工智能"), byte-fallback fragments (a rare character splits into 3–4 `<0xXX>` pieces — BMP characters 3 bytes, supplementary plane 4), and punctuation combos** — "Chinese is tokenized per character" is an outdated impression. To evaluate a tokenizer's Chinese efficiency, go back to §1.2: report bytes/token and chars/token on the same corpus, and declare the external word segmenter before quoting tokens/word (word counts vary considerably across segmentation standards). Cross-lingual fairness has one more layer: the tokenizer's corpus mix significantly shapes each language's fertility (§4.1) — on the same model, users of low-resource languages literally pay more for the same content under per-token billing and get less usable context.

### 4.5 Vocab extension: editing the tokenizer file is only step one

Extending an existing LLM's vocabulary (domain terms, a new language, agent control symbols): **editing the tokenizer alone will not make the model understand the new tokens**. The full checklist:

1. **Preserve the old world**: old token IDs, old merge ranks, and normalization all stay frozen — if a new **ordinary** token joins longest-match/merging, it can change how old text segments (backward compatibility collapses); control-symbol-style new tokens go through the **special channel** (atomic matching, explicit boundaries); domain words / new-language pieces are ordinary lexical tokens — they are supposed to participate in normal segmentation, at the cost that old-text segmentation may change: bound the compatibility domain with regression tests instead of marking everything special (specials get swallowed by `skip_special_tokens` and widen the injection surface);
2. **Resize both ends**: both the embedding and the LM head gain rows; with tied weights, confirm they still share one parameter after resizing; optimizer state (Adam's $m,v$) resizes in sync;
3. **Initialization**: an ordinary lexical token uses the mean of the embeddings of its surface's old segmentation; a special/control token (whose surface has no meaningful old segmentation) uses the mean over its description text's old segmentation — both are common heuristic baselines (the motivation: start the new row near the old-segmentation semantics); with untied weights the output row/bias needs separate initialization;
4. **Keep training**: new embedding rows need gradient feeding; mean-init without training = half a glitch token;
5. **Regression tests**: re-encoding the old test set must match ID-for-ID, special-token behavior must be unchanged, and round-trip behavior must match the pre-extension baseline (only samples that were reversible before assert equality with the original) — §7's [D08] is the minimal executable subset of this regression suite (covers IDs/specials/tied weights; no decoder round-trip).

## §5 Production Traps and Inference-Stack Interactions

### 5.1 Digits: three generations of splitting, and arithmetic

Three generations of digit pre-tokenization (the regex branch of §3.1):

| Scheme | Representative | Behavior | Consequence |
| --- | --- | --- | --- |
| unbounded digit runs | GPT-2 | `12345` freely merged by BPE | irregular groupings (`123`+`45`, `1`+`2345`... frequency-dependent), messy place alignment |
| per-digit | Llama 2 (SentencePiece `split_digits`) | one token per digit | clean place alignment, longer sequences |
| at most 3 per chunk | cl100k / o200k / Llama 3 (`\p{N}{1,3}`) | left-to-right greedy, up to 3 digits (`12345`→`123`+`45`, **not** right-aligned thousands grouping) | a compromise; **BPE can still split within a chunk** — it just cannot merge across chunks |

Digit splitting affects **place alignment** (units-digit over units-digit is how column addition becomes learnable) and **cross-token carrying** — but arithmetic ability cannot be single-causally pinned on the tokenizer: the share of arithmetic in training data, positional encoding, and CoT/tool use matter just as much. In an interview, blaming only the tokenizer for "why LLMs are bad at arithmetic" is half right; never mentioning the tokenizer is also half right.

### 5.2 Code: whitespace IS syntax

Tokenizers trained on code corpora typically learn merges for **multi-space runs, newline+indent combinations, and common symbol strings** — but "whether a standalone 4-space token exists" depends on the specific merge table, so don't assume; and tabs vs spaces are never equivalent by default (two indentation styles = two entirely different token sequences). For indentation-sensitive languages like Python, whitespace handling directly affects: a completion model's indentation accuracy, the token cost of the same code (how many files fit the context window), and boundary stability in diff/editing tasks.

### 5.3 Glitch tokens: a vocabulary–corpus mismatch

> ⚠️ **SolidGoldMagikarp (the classic glitch token, in one paragraph)**
> **Definition**: a token that exists in the vocabulary but almost never appeared under that ID during model training. **Cause (the mainstream explanation)**: the tokenizer's training corpus diverged from the model's (e.g. a Reddit username made it into the BPE vocabulary while its text got cleaned out) → that embedding/output row was never adequately learned. **Symptoms**: the model cannot repeat the token, emits gibberish or unpredictable behavior. **Diagnosis**: scan for embedding-norm outliers; ask the model to repeat candidate tokens. **Boundary**: it is a case study of vocabulary–corpus mismatch, not a universal failure of all GPT models or byte-level tokenizers.

### 5.4 Prefix instability and token healing

**The rule more general than the "trailing space problem"**: $\text{encode}(s_1)$ is **not necessarily** a token-prefix of $\text{encode}(s_1 + s_2)$ — appended characters can trigger merges across the old boundary. Minimal example (§7's [D06]): under merge table `[a+b, ab+c]`, `encode("ab")=["ab"]` but `encode("abc")=["abc"]` — the former is not a prefix of the latter. Three victims:

- **Completion-style prompts**: when user input ends with `"The qu"`, the trailing incomplete word may be cut at a "dangling" boundary rare in the training distribution (the exact split depends on vocabulary/leading space/version) — distribution mismatch, worse completions;
- **Token healing**: back up the last (or last few possibly-incomplete) prompt token(s), hand their string back to the decoder, and **constrain subsequent generation so its decoded prefix reproduces the backed-up suffix** (with single-token backup this is "the first token must start with that suffix"; multi-token backup needs trie/FSM-level sequence constraints). Note it is an **inference-side wrapper technique**, not a standard capability of every model API;
- **Streaming input / prefix cache**: see §5.7.

### 5.5 Special tokens: training-side details + the injection boundary

**Training side** (the pits run deeper than "forgot BOS"):

- BOS/EOS/PAD each have their role; **the attention mask and the loss mask are two different things** — PAD is usually masked in both, and chat data commonly uses assistant-only loss (loss computed only on assistant spans);
- **Setting PAD = EOS is not inherently wrong**; the genuine classic accident is then building masks indiscriminately from `input_ids == pad_id`, masking real EOS's loss (even attention) along the way → EOS supervision on that data is removed and stopping ability is markedly weakened (especially fatal when training from scratch) — masks should be built from true padding positions / sequence lengths;
- **Packing** (multiple documents in one sequence) must place EOS/BOS correctly at document boundaries and handle cross-document attention; when applying chat templates beware **double-added BOS** (one from the template, another from the tokenizer's `add_special_tokens`).

**The injection boundary**: when a literal `<|assistant|>` appears in user text, whether it encodes as a special ID or plain text **depends on the API** — provided the string is registered as a special in the encoding at all (`allowed_special` only gates admission; it cannot conjure special IDs); tiktoken errors by default on "registered but not allowed" specials (only relaxing `disallowed_special` makes it encode as plain text), while HF interfaces follow special-token config and template escaping. This is both a correctness issue (template assembly corrupts) and a **prompt-injection boundary**: before rendering user input, pin down "will the literal become a control symbol" — and write tests to freeze the behavior.

### 5.6 Offset mapping: multiple coordinate systems

Extractive QA, NER, code editors, and highlighters all must answer "which span of the original text does this token map to". The trap: there are **multiple coordinate systems** — character offsets in the normalized text, Unicode code-point offsets in the original text, UTF-8 byte offsets in the original text (some runtimes add UTF-16 code-unit and grapheme-cluster conventions) — lossy normalization like NFKC rewrites characters (`ﬃ`→`ffi` is 1→3 code points; `Ａ`→`A` changes the character value and byte length while the code-point count stays 1), so without an alignment map, normalized offsets simply **cannot be mapped back** to the original; emoji/combining characters further decouple code points from "user-perceived characters" (§3.3). Note NFC does not preserve code-point length either (`e`+U+0301 → `é` is 2→1). Engineering answer: prefer identity/near-faithful normalization, keep the normalization's **alignment map** (handles one-to-many/many-to-one spans), have the tokenizer return offset mappings (e.g. fast tokenizers' `return_offsets_mapping`), and verify both byte and code-point conventions in evaluation.

### 5.7 The inference stack: which components require tokenizer alignment

| Component | Requirement on the tokenizer | Why |
| --- | --- | --- |
| **speculative decoding** (classic per-token rejection sampling) | draft and target on the **same alignable token support** | compares $p_{\text{target}}$ vs $p_{\text{draft}}$ per token; different vocabularies make the probabilities incomparable; cross-tokenizer variants need retokenization/string-level bridging (no longer the basic algorithm) |
| **logit distillation** (per-position KL) | vocabulary **and** positions aligned | each position's distribution support must match; sequence-level distillation and character-level objectives cross tokenizers, hidden-state matching needs span alignment/pooling first |
| **KV / prefix cache** | same string → **same token-ID prefix** | hits are exact ID-prefix matches; a tokenizer change does not "abstractly break the cache" — the real invalidation conditions are the same string becoming a different ID sequence, a changed chat template, or changed model weights/positional state |
| **BPB evaluation** (§1.3) | one shared byte-stream convention | unify lossy normalization first, or the comparison is meaningless |

### 5.8 Tokenizer versioning: saving only vocab.json is how reproduction accidents start

To **exactly reproduce input IDs**, version together: the normalization config, the pre-tokenization regex, merge/vocab files, the special-token map plus added-token metadata (`lstrip/rstrip/normalized` etc.), post-processor/decoder configs, the chat template, library and implementation versions, and **hashes** of all of the above artifacts. Saving a single `vocab.json` (losing merge order and chunking rules) → six months later the same text tokenizes to different IDs, and training/evaluation/caching all drift silently. Treat the tokenizer like model weights: version control, experiment logging, and regression tests on every change (the §4.5 checklist).

## §6 The Tokenizer-Free Frontier

What "tokenizer-free" precisely means: **no dependence on a pretrained subword vocabulary** — these models still need UTF-8 byte encoding, special IDs, and sequence chunking or patching; there is no such thing as "no discretization at all". Three main lines:

- **ByT5** (Xue et al., 2021, arXiv 2105.13626): the mT5 architecture consuming UTF-8 bytes directly (vocabulary = 256 + specials). Gains: more robust to spelling perturbations/noise, a nearly free embedding table, natively zero OOV; costs: severalfold longer sequences for the same text, slower training and inference, and a heavier encoder allocation.
- **MegaByte** (Yu et al., 2023, arXiv 2305.07185): **fixed-size byte patches**, hierarchically — a global Transformer models the patch sequence while a light local model predicts bytes within each patch, amortizing the bulk of $O(L^2)$ to patch granularity. It is not "dump all bytes into one ordinary Transformer".
- **BLT (Byte Latent Transformer)** (Pagnoni et al. (Meta), 2024, arXiv 2412.09871): **dynamic patches** — a light next-byte entropy model marks boundaries so **hard-to-predict regions get denser patches** (spend compute where it is hard, stride over the easy parts); a local byte encoder aggregates patches → a global latent Transformer processes the patch sequence → a local decoder generates byte-by-byte. Average patch length is tunable, so the effective "bytes processed per FLOP" can compete with subword models.

| Model | Input granularity | Chunking mechanism | Primarily for | One-line limitation |
| --- | --- | --- | --- | --- |
| ByT5 | UTF-8 byte | none (full byte sequence) | seq2seq understanding+generation | long sequences, slow |
| MegaByte | byte | **fixed** patches | autoregressive generation | patch boundaries ignore content |
| BLT | byte | **entropy-driven dynamic** patches | autoregressive generation | needs an extra entropy model; young ecosystem |
| CANINE (Clark et al., 2021, arXiv 2103.06874) | Unicode code point | hash embedding + downsampling | **understanding-task** encoder | not an autoregressive byte LM — don't conflate |

> 💡 **"Tokenizer-free has solved scaling" is still an overstatement**
> Hierarchical byte models genuinely mitigate long-sequence cost and show competitive scaling curves, but training efficiency, inference-engine support, cache granularity (what does a prefix cache key on?), and mature deployment tooling remain real obstacles. The safe interview stance: **bullish on the direction, honest about the engineering gap** — and able to explain the essential difference between BLT's entropy patching and MegaByte's fixed patches.

## §7 From-Scratch Implementation (pure Python stdlib)

The full runnable script is [`code/tokenization.py`](code/tokenization.py) (pure stdlib, no torch, all 8 demos [D01]–[D08] in about a second on CPU). Keep two teaching simplifications in mind: the script's toy BPE is **codepoint-level** (initial symbols = Unicode characters, no pre-tokenization; real GPT-2 regex-chunks first, byte-maps within chunks, then runs BPE — see §3.1; [D01] demos that byte-proxy layer separately); and the two Unigram DPs use linear-space probabilities to match the §2.3 worked table — production implementations should switch to **log space** against underflow. Two more toy scopes: the toy BPE does not freeze a base alphabet (unseen characters simply become base tokens; real implementations either byte-fallback or emit `<unk>`); and [D05]'s `byte_fallback()` demos the byte representation and its restoration per se (it splits into bytes unconditionally), whereas real SentencePiece falls back only on spans that normal pieces cannot cover. The body shows only the four core demos' key code; WordPiece whole-word `[UNK]` ([D03]), NFKC vs byte fallback ([D05]), prefix instability ([D06]), and the vocab-extension regression test ([D08]) live as short test functions in the same script and are not reprinted.

**[D01] The GPT-2-style 256-byte reversible mapping** (§3.1's proxy alphabet, incl. `0x20 → Ġ`):

```python
def bytes_to_unicode():
    """Printable bytes keep their own char; the other 68 bytes shift to chr(256+n) — a 256↔256 bijection."""
    bs = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)                 # unprintable byte → shifted into printable range
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))

B2U = bytes_to_unicode()
U2B = {u: b for b, u in B2U.items()}

assert len(B2U) == 256 and len(set(B2U.values())) == 256   # bijection → reversible
assert B2U[0x20] == "Ġ"                                    # the space byte's proxy character
s = " A\n中文🙂\x00"                                        # space/newline/CJK/emoji/control char
units = "".join(B2U[b] for b in s.encode("utf-8"))
assert bytes(U2B[u] for u in units).decode("utf-8") == s   # strict round-trip for any UTF-8
```

**[D02] BPE: training records ORDERED merges; encoding only looks up frozen ranks** (the executable version of §2.1's worked example):

```python
def bpe_train(word_freqs, num_merges):
    """TRAIN: each round pick the most frequent adjacent pair (tie-break: count desc → lexicographic), record merges in order."""
    words = {tuple(w): f for w, f in word_freqs.items()}
    merges = []
    for _ in range(num_merges):
        counts = Counter()
        for w, f in words.items():
            for i in range(len(w) - 1):
                counts[(w[i], w[i + 1])] += f
        if not counts:
            break
        pair = min(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        merges.append(pair)                    # order = rank = encoding priority
        words = {merge_word(w, pair): f for w, f in words.items()}
    return merges

def bpe_encode(text, merges):
    """ENCODE: apply FROZEN merge ranks only — merge the lowest-rank applicable pair
    (leftmost first) each time; never re-count frequencies on the input."""
    rank = {p: i for i, p in enumerate(merges)}
    seq = list(text)
    while len(seq) > 1:
        best_rank, best_i = math.inf, -1
        for i in range(len(seq) - 1):
            r = rank.get((seq[i], seq[i + 1]), math.inf)
            if r < best_rank:
                best_rank, best_i = r, i
        if best_rank == math.inf:
            break
        seq = seq[:best_i] + [seq[best_i] + seq[best_i + 1]] + seq[best_i + 2:]
    return seq

merges = bpe_train({"abab": 3, "abac": 1}, num_merges=3)
assert merges[0] == ("a", "b")                 # §2.1 hand count: pair freq 7 = 2×3 + 1×1
text = "ababac"
assert "".join(bpe_encode(text, merges)) == text            # round-trip
assert bpe_encode(text, merges) == bpe_encode(text, merges) # deterministic
# rank semantics pinned: even though (b,c) applies too, rank-0 (a,b) goes first — NOT "re-pick the more frequent"
assert bpe_encode("abc", [("a", "b"), ("b", "c")]) == ["ab", "c"]
assert bpe_encode("abc", [("b", "c"), ("a", "b")]) == ["a", "bc"]   # ranks reversed, outcome follows the ranks
```

**[D04] Unigram: Viterbi best path vs forward all-segmentation marginal** (the executable version of §2.3's DP table):

```python
def viterbi(s, p):
    """Best single segmentation: argmax over products of piece probabilities."""
    best = [(1.0, [])] + [None] * len(s)
    for i in range(1, len(s) + 1):
        cands = [(best[j][0] * p[s[j:i]], best[j][1] + [s[j:i]])
                 for j in range(i) if s[j:i] in p and best[j] is not None]
        best[i] = max(cands) if cands else None
    return best[len(s)][1]

def forward_prob(s, p):
    """P(s) = sum over ALL segmentations (the marginal), via forward DP — the forward half of the E-step's forward-backward."""
    f = [1.0] + [0.0] * len(s)
    for i in range(1, len(s) + 1):
        for j in range(i):
            if s[j:i] in p:
                f[i] += f[j] * p[s[j:i]]
    return f[len(s)]

p = {"a": .15, "b": .15, "ab": .40, "aba": .20, "bab": .10}
assert viterbi("abab", p) == ["ab", "ab"]                    # best path, probability 0.16
assert abs(forward_prob("abab", p) - 0.22350625) < 1e-12     # f4 from §2.3's DP table
assert forward_prob("abab", p) >= math.prod(p[x] for x in viterbi("abab", p))
```

**[D07] Fertility and BPB** (the executable version of §1.2/§1.3's metric conventions):

```python
texts = ["def add(x): return x + 1", "把变量加一🙂"]        # mixed Chinese/English/code corpus
total_bytes = sum(len(t.encode("utf-8")) for t in texts)
char_toks = sum(len(list(t)) for t in texts)                 # char tokenizer
byte_toks = sum(len(t.encode("utf-8")) for t in texts)       # byte tokenizer
merges7 = bpe_train({t: 1 for t in texts}, num_merges=12)    # train 12 merges on the corpus itself
bpe_toks = sum(len(bpe_encode(t, merges7)) for t in texts)   # toy-BPE (12 merges)
assert byte_toks / total_bytes == 1.0        # byte tokenizer: fertility is always 1 token/byte
assert char_toks < byte_toks                 # CJK/emoji chars cost 3–4 bytes each → char wins
assert bpe_toks < char_toks                  # merges compress further (micro-average convention)

nll, n_bytes = 4.0, 8                        # same text, same total NLL (nats), 8 bytes
loss_per_token_a, loss_per_token_b = nll / 2, nll / 4        # 2 vs 4 tokens
bpb_a = (2 * loss_per_token_a) / (n_bytes * math.log(2))     # recover BPB from loss/token, independently
bpb_b = (4 * loss_per_token_b) / (n_bytes * math.log(2))
assert loss_per_token_a == 2.0 and loss_per_token_b == 1.0   # loss/token differs 2×
assert bpb_a == bpb_b == 4.0 / (8 * math.log(2))             # yet BPB is identical
```

> ✅ **Each of the four snippets pins one frequent misconception**
> [D01]: byte-level's "zero OOV" comes from the full 256-byte base alphabet being in the vocabulary (the bijection only guarantees the proxy mapping is reversible); [D02]: **encoding only looks up ranks** — no frequency re-counting; [D04]: **the marginal $\geq$ the best-path probability** — the E-step needs forward-backward's expected posterior counts (D04 demos its marginal ingredient, forward), and only inference uses Viterbi; [D07]: **with identical total NLL, token count alone changes loss/token severalfold** — compare across tokenizers with BPB.

## §8 25 High-Frequency Interview Questions

Three difficulty tiers; expand each for answer keys + common pitfalls. L2/L3 are top-lab deep water (training-vs-encoding semantics, the BPB derivation, vocab-extension system design, inference-stack alignment, tokenizer-free). When answering, **don't re-derive the body text** — organize as "framework → key formula → common mistakes".

### L1 must-know

<details>

<summary>Q1. Why do LLMs use subword instead of word-level or char-level?</summary>

- word-level: OOV + vocabulary explosion + morphological variants sharing no statistical strength
- char/byte-level: byte-level is zero-OOV (char-level can still OOV with incomplete coverage) but sequences grow severalfold — attention $O(L^2)$ / KV cache suffer
- subword: frequent words whole, rare words split, bytes as the net → bounded vocabulary + strong coverage (§1.1)

Answering just "a compromise" without the concrete costs at both ends; or not knowing zero OOV requires a complete-alphabet net (in practice, a byte alphabet / byte fallback).

</details>

<details>

<summary>Q2. Hand-write BPE: what do training and encode each do? (the #1 hand-coding question)</summary>

- Training: iteratively count weighted adjacent-pair frequencies → merge the most frequent → **record the merge table in order** (deterministic tie-breaking)
- encode: repeatedly merge "the currently lowest-rank applicable pair" by **frozen rank** — no frequency re-counting, not longest-match
- decode: concatenate token strings (byte-level restores UTF-8; real systems also undo BPE boundary markers like `▁`/`@@`); also note: merges never cross pre-tokenization chunk boundaries, and overlapping pairs replace "left-to-right, non-overlapping"

Writing encode as "merge the currently most frequent pair in the input" — conflating training with encoding semantics is the #1 cause of death on this question (§2.1).

</details>

<details>

<summary>Q3. Why does byte-level BPE have essentially no OOV? At what cost?</summary>

- Initial alphabet = the full 256 bytes; any valid UTF-8 text decomposes to byte level → the strong coverage guarantee
- GPT-2 maps bytes through the `bytes_to_unicode` bijection into printable proxy characters before running BPE (§3.1)
- Costs: high fertility on rare Unicode text (low-resource languages get shredded), longer byte sequences; lossy normalization can still rewrite the original

Believing "no OOV" is a property of the BPE algorithm itself (it is a property of the byte alphabet); or not knowing the proxy alphabet is just an implementation bridge.

</details>

<details>

<summary>Q4. What are the Ġ and ▁ in token visualizations?</summary>

- `Ġ`: the image of byte `0x20` (space) under GPT-2's proxy alphabet (U+0120) — every space byte maps to it; it commonly appears at token starts because the regex attaches the space to the following chunk
- `▁`: SentencePiece's U+2581, explicitly symbolizing the space so it participates in modeling; decode restores it
- Common ground: both are characters that genuinely exist in the serialized vocabulary, and both mean "whitespace stored explicitly"; the mechanisms are entirely different (byte remap vs whitespace escaping) — and `▁` decodes back to the POST-normalization space (with dummy prefix / whitespace folding, the original whitespace is not faithful)

Calling `Ġ`/`▁` "weird vocab characters" without stating their semantics, or conflating the two mechanisms (§3.2 callout).

</details>

<details>

<summary>Q5. WordPiece's `##` prefix and the `[UNK]` semantics?</summary>

- `##` marks a "non-word-initial continuation subword" (`##ing`); encoding is longest-match-first
- A match failure at ANY position → the **whole word** becomes `[UNK]`, not "successful prefix + `[UNK]`"
- Example: with vocabulary `{a, ##b, ##bc}`, `abc → [a, ##bc]`; remove `##bc` and `abc → [[UNK]]` (§2.2, [D03])

Answering `[a, ##b, [UNK]]` — not knowing BERT-style failure semantics is whole-word rollback.

</details>

<details>

<summary>Q6. What is fertility? How do you judge a tokenizer?</summary>

- fertility = average tokens per unit; **the denominator must be declared** (narrow literature usage often means tokens/word specifically; tokens/char and tokens/byte are more often called compression ratios)
- Cross-lingual comparison: on content/domain-matched (ideally semantically parallel) eval text, report bytes/token + chars/token + (segmenter-declared) tokens/word + UNK rate, with per-language macro aggregation (§1.2)
- Ultimately end-to-end: total token count on the same corpus, throughput, task quality — the tokenizer is a systems question

Reporting a single "compression ratio" without the convention; quoting tokens/word for Chinese without naming the segmentation standard.

</details>

<details>

<summary>Q7. What do BOS/EOS/PAD each do? Is setting PAD = EOS a problem?</summary>

- BOS marks the start, EOS marks the end (EOS is usually the generation stop condition); PAD pads the batch, masked in BOTH the attention mask and the loss mask
- PAD=EOS is **not inherently wrong** (many models have no dedicated PAD); the classic accident is building masks indiscriminately from `input_ids==pad_id`, masking real EOS's loss/attention too → EOS supervision removed, stopping ability markedly weakened (worst when training from scratch) — build masks from true padding positions
- Related pits: chat template + `add_special_tokens` double-adding BOS; EOS/attention handling at document boundaries when packing (§5.5)

Reciting the absolutized "PAD=EOS is wrong" without the actual failure chain (real EOS's loss getting masked).

</details>

<details>

<summary>Q8. Does `decode(encode(x)) == x` always hold? And the reverse?</summary>

- Not necessarily. The essence is end-to-end pipeline fidelity: the normalizer loses nothing, no `<unk>`, symmetric special/post-processing, no decoder cleanup; the common configuration = identity normalization + whitespace fidelity + byte net + cleanup off + injective escaping of internal markers/special surfaces (§3.3)
- One NFKC pass is already lossy: `Ａ①` → `A1`, the original is gone forever (byte fallback cannot save it)
- `encode(decode(ids)) == ids` **generally fails**: non-canonical token sequences decode to the same string; re-encoding returns only the canonical segmentation

Assuming reversibility is a default perk; missing that the second direction (ID round-trip) is simply not guaranteed; or conflating this ID-round-trip infidelity (canonicalization — idempotent under typical configs) with prefix instability (§5.4, the direct root of token healing).

</details>

### L2 advanced

<details>

<summary>Q9. The core differences between BPE, WordPiece, and Unigram? (the #1 concept question)</summary>

- Training: BPE counts the most frequent pair (greedy, additive); WordPiece picks merges by likelihood gain / PMI-like score; Unigram EM-estimates probabilities over a large seed vocabulary + prunes by likelihood loss (subtractive)
- Encoding: frozen-rank merging / longest-match-first / Viterbi (sampleable)
- The finishing move: `[UNK]` semantics (whole-word vs `<unk>` vs byte-net zero-OOV), randomization (BPE-dropout vs native sampling), training direction and artifacts (§2.4 table)

Reciting "three names + who uses which" without the main line that **training objective and encoding algorithm are two independent programs**.

</details>

<details>

<summary>Q10. What is the relationship between SentencePiece and Unigram?</summary>

- SentencePiece is a **framework** (training + encoding + normalization + serialization); `model_type` can be unigram/bpe/char/word — not a synonym for Unigram
- Properties: trains straight from raw sentences (no language-specific pre-segmentation, though normalization/whitespace escaping still exist); `▁` whitespace symbolization; default normalizer `nmt_nfkc`
- `byte_fallback=true` reserves 256 `<0xXX>` pieces as the net (§3.2)

"SentencePiece IS Unigram" and "SentencePiece has no preprocessing at all" — both are classic misstatements.

</details>

<details>

<summary>Q11. What's the difference between byte fallback and byte-level BPE?</summary>

- byte fallback (SentencePiece): the **safety net** — retreat to `<0xXX>` byte pieces only when ordinary Unicode pieces lack coverage; decode restores them
- byte-level BPE (GPT-2/Llama 3): the **foundation** — learn merges on the full 256-byte alphabet from step one
- Both give strong coverage, but byte fallback cannot recover text rewritten by lossy normalization; the fragmentation patterns differ too (§3.2 callout, [D05])

Treating the two as synonyms; or believing byte fallback recovers the pre-NFKC original (coverage ≠ fidelity).

</details>

<details>

<summary>Q12. Why can't the vocabulary grow without bound? (derivation)</summary>

- Cost side: embedding $\approx Vd$, an untied output head adds another $Vd$; tying saves parameters but softmax **compute** still $\propto Vd$ (§4.2)
- Benefit side: larger V → (usually) lower fertility → shorter sequences → attention $O(L^2)$ / KV cache / generation steps all cheaper
- A further statistical cost: on a fixed corpus, an oversized vocabulary breeds masses of low-frequency, under-trained embedding/output rows, weakening parameter sharing and compositional generalization, with diminishing compression returns
- Opposite pulls → optimal V depends on language coverage, code share, model width $d$, deployment budget; 32k→256k is a set of design points, not a timeline

Answering only "embedding grows" (omitting the benefit side fails to explain why vocabularies kept growing); or treating convention gaps like 100,277 vs 100,261 as the same number (§4.3).

</details>

<details>

<summary>Q13. How are digits tokenized? What does it do to arithmetic?</summary>

- Three generations: GPT-2 unbounded digit runs (irregular groupings) → Llama 2 `split_digits` per-digit → cl100k/o200k/Llama 3 `\p{N}{1,3}` at most 3 per chunk (§5.1)
- `{1,3}` is a pre-token cap: BPE can still split within the chunk, but cannot merge across chunks
- Affects place alignment and cross-token carrying; but arithmetic ability cannot be attributed solely to the tokenizer (data, positional encoding, CoT/tools matter equally)

Blaming "LLMs are bad at arithmetic" entirely on the tokenizer; or believing `{1,3}` guarantees final tokens of exactly 3 digits.

</details>

<details>

<summary>Q14. What are subword regularization and BPE-dropout?</summary>

- Unigram sampling: sample segmentations from $q(z)\propto P(z)^{\alpha}$ — `alpha` is an inverse temperature (larger → closer to Viterbi, smaller → more random); the same word segments differently across epochs; data augmentation + robustness
- BPE-dropout: BPE has no probabilistic model, so randomly **block** merges with probability $p$ at encoding time to produce fragmented splits
- Similar purpose, different mechanisms (posterior sampling vs merge blocking); standard deployments usually disable randomization and return to deterministic segmentation (§2.3)

Mixing up the semantics of `alpha` and the dropout probability $p$ (`alpha` is an INVERSE temperature — don't state the direction backwards); or not knowing BPE has no sampleable probabilistic model.

</details>

<details>

<summary>Q15. How do you fairly compare a tokenizer's Chinese vs English efficiency?</summary>

- Refuse a bare tokens/word: Chinese "words" depend on an external segmentation standard — undeclared conventions are incomparable
- Use content/domain-matched (ideally semantically parallel) bilingual corpora; report UTF-8 bytes/token, chars/token, (segmenter-declared) tokens/word, UNK rate, total token count, and end-to-end throughput, declaring macro/micro aggregation (§1.2, §4.4)
- Remember the cause: the tokenizer corpus's language mix significantly shapes per-language fertility — low-resource languages pay more for the same content under per-token billing and get less usable context

Applying English-convention tokens/word directly to Chinese; or not knowing Chinese tokens are actually a mix of "single characters + multi-character words + byte fragments + punctuation combos" (§4.4).

</details>

<details>

<summary>Q16. What does Unigram's E-step use? Why not Viterbi?</summary>

- The objective is the marginal likelihood $P(x)=\sum_{z\in S(x)}\prod_i p(z_i)$ — the E-step aggregates expected posterior counts over **all segmentations** via forward-backward
- Taking only Viterbi = hard-EM, which optimizes a different objective; Viterbi is for inference (deterministic encoding) (§2.3)
- Executable intuition: in [D04], the forward probability 0.2235 $\gt$ the best path 0.16 — the marginal and a single path are two different quantities

"Both training and inference use Viterbi" — writing soft EM as hard EM, a frequent error the design review called out by name.

</details>

<details>

<summary>Q17. What is prefix instability? How does token healing fix it?</summary>

- The general law: $\text{encode}(s_1)$ need not be a token-prefix of $\text{encode}(s_1+s_2)$ — appended characters can trigger merges across old boundaries ([D06]: `enc("ab")=[ab]` but `enc("abc")=[abc]`)
- Victims: "dangling" trailing tokens in completion prompts (distribution mismatch), streaming input, prefix caches
- token healing: back up the last few possibly-incomplete tokens and constrain subsequent generation's decoded prefix to reproduce that suffix (the classic single-token case = a first-token constraint); an inference-side wrapper, not standard in every API (§5.4)

Knowing only the "trailing space" special case without the general prefix instability; or believing token healing is a model capability rather than an inference wrapper.

</details>

<details>

<summary>Q18. What's the deal with glitch tokens (SolidGoldMagikarp)?</summary>

- Definition: a token present in the vocabulary that almost never appeared under its ID during model training → under-trained embedding/output row
- Cause (the typical explanation, not a proven sole mechanism): tokenizer-corpus vs model-corpus mismatch (text that entered the vocabulary got cleaned out later); symptoms: cannot repeat it, gibberish, anomalous behavior
- Diagnosis: embedding-norm outlier scans, repeat tests; boundary: a case study, not a universal failure of all GPT/byte-level tokenizers (§5.3)

Telling it as an urban legend without the mechanism chain "vocabulary–corpus mismatch → under-trained embedding".

</details>

### L3 advanced

<details>

<summary>Q19. Derive BPB. Why can't perplexities of different tokenizers be compared directly?</summary>

- loss/token's denominator (token count) depends on the tokenizer — even at identical total NLL, the finer-cutting side gets lower loss/token ([D07]); real gaps additionally stack model quality on top, and the two effects are inseparable
- The fixed invariant is the **same text's UTF-8 byte count** $B$: $\mathrm{BPB}=\frac{-\sum_i\log p(t_i\mid t_{\lt i})}{B\ln 2}$; equal total NLL ⇒ equal BPB, regardless of how many cuts ([D07]: loss/token 2.0 vs 1.0, BPB both 0.7213)
- Precondition: both sides score the same byte stream — unify lossy normalization first (§1.3)

Comparing PPL while forgetting that the "per token" in the exponent is itself the thing being compared; or not checking whether normalization rewrote the eval text.

</details>

<details>

<summary>Q20. How do you safely extend an existing LLM's vocabulary? (system design)</summary>

- Freeze the old world: old IDs / merge ranks / normalization untouched; control symbols via the special channel, domain words as ordinary lexical tokens (they WILL change old-text segmentation — bound the compatibility domain with regression tests instead of marking everything special)
- Resize both ends + optimizer state: embedding, LM head, Adam's $m,v$ all extend; confirm tied weights still share
- Mean-init new rows (lexical tokens: mean over their own surface's old segmentation; specials: mean over the old segmentation of a description-text proxy) + **keep training**; after extending the LM head, new rows enter the softmax denominator — old-token distributions have changed, so regress old-set NLL/generation behavior too; plus encoding regression: old test set ID-identical, special behavior unchanged, round-trip behavior matching the pre-extension baseline (§4.5, [D08])

Editing the tokenizer file and assuming the model knows the new tokens; or missing any one of the LM head / optimizer state / tied-weights checks.

</details>

<details>

<summary>Q21. Which system components require tokenizer alignment, and why?</summary>

- Classic speculative decoding: draft/target need the **same alignable token support** (per-token rejection sampling compares two distributions); cross-tokenizer setups need dedicated string-level proposal-verification designs — naive bridging does not preserve exact equivalence
- Per-position logit distillation: vocabulary AND positions must align; sequence-level KD and character-level objectives cross tokenizers, hidden-state KD needs span alignment/pooling
- KV/prefix cache: hits on the **exact token-ID prefix** — invalidation means the same string becoming different IDs / template change / weight change; "same string" is not sufficient (§5.7)

Vaguely saying "tokenizer changed, cache broken" without the precise invalidation conditions; or not knowing sequence-level distillation crosses tokenizers.

</details>

<details>

<summary>Q22. Design an offset-mapping scheme: how do tokens map back to the original text?</summary>

- Multiple coordinate systems: normalized-text char offsets / original code-point offsets / original UTF-8 byte offsets (plus UTF-16 code units, grapheme clusters) — pin down which one the API returns (§5.6)
- Lossy normalization (NFKC) rewrites characters (`ﬃ`→`ffi` is 1→3 code points; `Ａ`→`A` changes value/byte length with code-point count unchanged); without an alignment map you cannot map back; NFC does not preserve code-point length either (`e`+U+0301→`é` 2→1); emoji ZWJ/combining characters decouple code points from user-perceived characters
- Scheme: near-faithful normalization + keep the alignment map (handling one-to-many/many-to-one spans) + the tokenizer's native offset mapping + dual-convention (byte and code point) tests; downstream (NER / extractive QA / code editing) consumes its own coordinate system

Giving the one-liner "use return_offsets_mapping" without the two-layer coordinate mismatch from normalization and Unicode.

</details>

<details>

<summary>Q23. Design the corpus and configuration for training a new LLM's tokenizer from scratch.</summary>

- Corpus: language mix (significantly shapes per-language fertility; deliberate rebalancing of low-resource languages is fine — record the mix), code/math share, dedup and cleaning; **cleaning and normalization semantics must match the model corpus**, and vocabulary tokens need sufficient exposure in the model corpus — vocabulary–corpus mismatch is the glitch-token breeding ground (§4.1)
- Config: vocabulary size from the $Vd$ ledger + fertility targets (§4.2); digit policy (`{1,3}` vs per-digit); byte net; special/reserved slots allocated once, generously
- Acceptance: multi-language multi-convention fertility, round-trip tests (identity pipelines assert equality with the original; lossy normalization asserts decoding to the canonical form), small controlled pretraining comparing BPB/downstream quality/throughput, regression baselines, versioning the full file set + hashes (§5.8)

Agonizing over "BPE or Unigram" upfront — corpus design usually affects final fertility more than the algorithm choice (§4.1's thesis).

</details>

<details>

<summary>Q24. Tokenizer-free: the mechanisms of ByT5 / MegaByte / BLT? The practical obstacles?</summary>

- ByT5: raw UTF-8 bytes in (vocabulary 256+specials), robust but severalfold longer sequences; MegaByte: **fixed** byte patches, global model over the patch sequence + local model over bytes within patches, amortizing $O(L^2)$
- BLT: **entropy-driven dynamic patches** — a light next-byte entropy model sets boundaries, denser patches where prediction is hard (compute allocated by difficulty); local encoder → global latent Transformer → local decoder
- Precise phrasing: "tokenizer-free" = no pretrained subword vocabulary (UTF-8/special IDs/patching still needed); scaling is competitive but training efficiency, inference ecosystem, and cache granularity remain obstacles (§6)

Describing all three as "just dump bytes into a Transformer"; or declaring "the tokenizer is solved" (the overstatement §6 names).

</details>

<details>

<summary>Q25. Chat templates and special-token injection: how to defend? What other training-side pits?</summary>

- Injection surface: whether a literal `<|assistant|>` in user text becomes a special ID depends on the API (registered first, then admitted by `allowed_special` / template escaping) — the fundamental defense is **two-channel encoding**: template control symbols injected as IDs by code, user content encoded separately with special recognition off; tests are only the backstop; this is a correctness + prompt-injection double boundary (§5.5)
- Training side: attention mask ≠ loss mask; assistant-only loss; packing document boundaries; template + tokenizer double-adding BOS; with PAD=EOS, never mask all EOS loss
- Companion practice: tokenizer versioning (normalization + regex + merges + special/added-token metadata + post-processor/decoder + chat template + library versions + hashes); saving only vocab.json drifts silently (§5.8)

Answering only "filter special strings from user input" — not knowing the behavior is decided by tokenizer API configuration, and that training-side mask errors are just as fatal.

</details>

## §A Appendix: sanity check

The from-scratch implementations in this tutorial must satisfy the following key invariants (pure Python stdlib, no third-party dependencies, seconds on CPU; script at [`code/tokenization.py`](code/tokenization.py)):

1. **[D01] The 256-byte bijection**: `bytes_to_unicode` maps exactly 256 in, 256 out (injective → reversible), 188 printable bytes as themselves and 68 shifted bytes ascending into the U+0100 range (`0x00→U+0100`, `0x0A→Ċ`, `0x20→Ġ`); a string containing space/newline/CJK/emoji/control chars round-trips strictly through UTF-8 → proxy alphabet → inverse map → UTF-8.
2. **[D02] BPE training-vs-encoding semantics**: on corpus `{"abab"×3, "abac"×1}` the first merge must be `(a,b)` (count 7); `decode(encode(x))==x`, deterministic encoding, token count $\leq$ byte count; the rank-conflict case `encode("abc")==["ab","c"]` flips to `["a","bc"]` when the ranks are reversed (encoding follows frozen ranks only, no runtime re-picking); `"aaa"` merges left-to-right non-overlapping into `["aa","a"]`.
3. **[D03] WordPiece whole-word failure**: with `{a, ##b, ##bc}`, `abc → [a, ##bc]`; removing `##bc` gives `abc → [[UNK]]` (not `[a, ##b, [UNK]]`); with `{a, ab, ##bc}`, greedy picks `ab`, dead-ends → whole-word `[UNK]` (greedy does not backtrack, even though `a`+`##bc` would have worked).
4. **[D04] Unigram's two DPs**: Viterbi gives `["ab","ab"]` (0.16); the forward all-segmentation marginal $=0.22350625$ (independently hand-verified) and $\geq$ the best-path probability.
5. **[D05] Coverage ≠ fidelity**: NFKC rewrites `Ａ①` into `A1` (lossy, irreversible); byte fallback splits `龘` into `<0xE9><0xBE><0x98>` and restores it exactly.
6. **[D06] Prefix instability**: under merge table `[a+b, ab+c]`, `encode("ab")=["ab"]` is not a token-prefix of `encode("abc")=["abc"]`.
7. **[D07] Metric conventions**: the byte tokenizer's fertility is always 1 token/byte; on the mixed Chinese/English corpus toy-BPE $\lt$ char $\lt$ byte (micro-average); with the same text at NLL=4.0 nats and 8 bytes, loss/token is 2.0 vs 1.0 while BPB is identical.
8. **[D08] Vocab-extension regression**: old IDs / old test-set encodings / special behavior all unchanged; the new embedding row = the per-dimension mean of the semantic-proxy text's old-segmentation rows; the tied input embedding and LM head remain the same parameter object.

The **real output** of `python3 code/tokenization.py`:

```text
[D01] bytes_to_unicode: 256/256 bijective, 0x20 -> 'Ġ', roundtrip ' A\n中文🙂\x00' OK  PASS
[D02] BPE: merges=[('a', 'b'), ('ab', 'ab'), ('a', 'c')], encode('ababac')=['abab', 'ac'], rank-conflict 'abc'->['ab', 'c']  PASS
[D03] WordPiece: 'abc'->['a','##bc']; drop '##bc' -> ['[UNK]'] (whole word)  PASS
[D04] Unigram: viterbi=['ab', 'ab'] (prob 0.1600), forward=0.22350625 >= best path  PASS
[D05] NFKC('Ａ①')='A1' (lossy); byte_fallback('龘')=<0xE9><0xBE><0x98> roundtrips  PASS
[D06] prefix instability: enc('ab')=['ab'] but enc('abc')=['abc'] -> not a token-prefix  PASS
[D07] fertility tokens/byte: byte=1.000, char=0.698, toy-BPE=0.419; loss/token 2.0 vs 1.0 yet BPB both 0.7213  PASS
[D08] vocab extension: old IDs/encodings/specials unchanged, new row = mean(rows [0, 2]) = [4.0, 5.0, 6.0, 7.0], tied head intact  PASS

all tokenization sanity checks passed ✓
```

## 📚 References

- **BPE (compression origin)** — Gage, *A New Algorithm for Data Compression*, C Users Journal (1994) (**no arXiv id**).
- **BPE for NMT** — Sennrich, Haddow & Birch, *Neural Machine Translation of Rare Words with Subword Units*, arXiv 1508.07909 (2015), ACL 2016.
- **WordPiece (origin)** — Schuster & Nakajima, *Japanese and Korean Voice Search*, ICASSP 2012 (**no arXiv id**).
- **GNMT (WordPiece at scale)** — Wu et al., *Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation*, arXiv 1609.08144 (2016).
- **BERT** — Devlin, Chang, Lee & Toutanova, *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*, arXiv 1810.04805 (2018), NAACL 2019.
- **Unigram LM / subword regularization** — Kudo, *Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates*, arXiv 1804.10959 (2018), ACL 2018.
- **SentencePiece** — Kudo & Richardson, *SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing*, arXiv 1808.06226 (2018), EMNLP 2018 (System Demonstrations).
- **T5 (SentencePiece-Unigram user)** — Raffel et al., *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer*, arXiv 1910.10683 (2019), JMLR 2020.
- **ALBERT (SentencePiece-Unigram user)** — Lan et al., *ALBERT: A Lite BERT for Self-supervised Learning of Language Representations*, arXiv 1909.11942 (2019), ICLR 2020.
- **BPE-dropout** — Provilkov, Emelianenko & Voita, *BPE-Dropout: Simple and Effective Subword Regularization*, arXiv 1910.13267 (2019), ACL 2020.
- **GPT-2 (byte-level BPE)** — Radford et al., *Language Models are Unsupervised Multitask Learners*, OpenAI technical report (2019) (**no arXiv id**).
- **tiktoken** — OpenAI, https://github.com/openai/tiktoken (BPE implementation library, cl100k_base / o200k_base encodings; **no paper**).
- **Llama 2** — Touvron et al., *Llama 2: Open Foundation and Fine-Tuned Chat Models*, arXiv 2307.09288 (2023).
- **Llama 3** — Grattafiori et al. (Meta), *The Llama 3 Herd of Models*, arXiv 2407.21783 (2024).
- **Qwen2** — Yang et al. (Qwen Team), *Qwen2 Technical Report*, arXiv 2407.10671 (2024).
- **Gemma** — Gemma Team (Google DeepMind), *Gemma: Open Models Based on Gemini Research and Technology*, arXiv 2403.08295 (2024).
- **ByT5** — Xue et al., *ByT5: Towards a token-free future with pre-trained byte-to-byte models*, arXiv 2105.13626 (2021), TACL 2022.
- **MegaByte** — Yu et al. (Meta), *MEGABYTE: Predicting Million-byte Sequences with Multiscale Transformers*, arXiv 2305.07185 (2023), NeurIPS 2023.
- **BLT** — Pagnoni et al. (Meta), *Byte Latent Transformer: Patches Scale Better Than Tokens*, arXiv 2412.09871 (2024).
- **CANINE** — Clark, Garrette, Turc & Wieting, *CANINE: Pre-training an Efficient Tokenization-Free Encoder for Language Representation*, arXiv 2103.06874 (2021), TACL 2022.
- **Speculative decoding** — Leviathan, Kalman & Matias, *Fast Inference from Transformers via Speculative Decoding*, arXiv 2211.17192 (2022), ICML 2023.
- **Speculative sampling** — Chen et al. (DeepMind), *Accelerating Large Language Model Decoding with Speculative Sampling*, arXiv 2302.01318 (2023).
- **SolidGoldMagikarp (glitch tokens)** — Rumbelow & Watkins, *SolidGoldMagikarp (plus, prompt generation)*, LessWrong blog post (2023) (**no arXiv id**; cited as a blog post).
