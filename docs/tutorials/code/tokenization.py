"""
Tokenization - minimal runnable implementation
==============================================

Pure-stdlib (Python 3.9+, no torch) sanity checks, runs in <1s on CPU.
Pairs with: docs/tutorials/tokenization_tutorial.md (concept reference).

    [D01] 256-byte reversible mapping (bytes_to_unicode); space byte 0x20 -> 'Ġ'
    [D02] BPE trainer (ORDERED merges) + frozen-rank encoder + decoder
    [D03] WordPiece longest-match-first; failure = WHOLE word -> [UNK]
    [D04] Unigram Viterbi (best path) vs forward probability (all segmentations)
    [D05] NFKC normalization is lossy; byte fallback roundtrips its input
    [D06] prefix instability: encode("ab") is NOT a token-prefix of encode("abc")
    [D07] fertility (tokens/byte, micro-average) and BPB vs loss-per-token
    [D08] vocab extension regression: old IDs/encodings/specials unchanged;
          new row = mean of decomposed old embeddings; tied head intact

Run:
    python3 tokenization.py
"""
import math
import sys
import unicodedata
from collections import Counter


# ---- [D01] GPT-2-style reversible byte -> unicode mapping ----

def bytes_to_unicode():
    """Printable bytes keep their own char; the other 68 bytes get chr(256+n)."""
    bs = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)                     # shift into printable Unicode range
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))

B2U = bytes_to_unicode()
U2B = {u: b for b, u in B2U.items()}

def to_units(text):
    """UTF-8 bytes -> printable proxy alphabet (what GPT-2 BPE actually sees)."""
    return "".join(B2U[b] for b in text.encode("utf-8"))

def from_units(units):
    """Proxy alphabet -> raw bytes (inverse map)."""
    return bytes(U2B[u] for u in units)


# ---- [D02]/[D06] BPE: trainer with ordered merges + frozen-rank encoder ----
# Teaching simplification: this is a CODEPOINT-level toy BPE (initial symbols =
# Unicode chars, no pre-tokenization). Real GPT-2 first regex-splits the text,
# then maps each chunk's UTF-8 bytes through bytes_to_unicode ([D01]) and runs
# BPE inside chunk boundaries — see tutorial §3.1.
# Pair counting uses the standard sliding-window convention: in "aaaa" the pair
# (a,a) counts 3, but left-to-right non-overlapping replacement applies twice.
# The toy also does NOT freeze a base alphabet: unseen symbols pass through as
# base tokens (a real tokenizer would byte-fallback or emit <unk>).

def merge_word(w, pair):
    out, i = [], 0
    while i < len(w):
        if i < len(w) - 1 and (w[i], w[i + 1]) == pair:
            out.append(w[i] + w[i + 1]); i += 2
        else:
            out.append(w[i]); i += 1
    return tuple(out)

def bpe_train(word_freqs, num_merges):
    """TRAIN: each round pick the most frequent adjacent pair (deterministic
    tie-break: higher count, then lexicographic) and record the ORDERED merge."""
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
        merges.append(pair)
        words = {merge_word(w, pair): f for w, f in words.items()}
    return merges

def bpe_encode(text, merges):
    """ENCODE: apply FROZEN merge ranks only - always the lowest-rank applicable
    pair, leftmost first; never re-count frequencies on the input."""
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

def bpe_decode(tokens):
    return "".join(tokens)


# ---- [D03] BERT-style WordPiece: longest-match-first, whole-word [UNK] ----

def wordpiece(word, vocab):
    """BERT-style greedy longest-match (teaching version; real impls also cap
    word length, e.g. max_input_chars_per_word=100 -> whole-word [UNK])."""
    pieces, start = [], 0
    while start < len(word):
        end, cur = len(word), None
        while end > start:
            sub = ("##" if start > 0 else "") + word[start:end]
            if sub in vocab:
                cur = sub; break
            end -= 1
        if cur is None:
            return ["[UNK]"]                       # whole word fails, not partial pieces
        pieces.append(cur); start = end
    return pieces


# ---- [D04] Unigram LM: Viterbi vs forward probability ----
# NOTE: linear-space probs for readability on toy inputs; production uses
# log-space (Viterbi: sum of log p; forward: log-sum-exp) to avoid underflow.

def viterbi(s, p):
    """Best single segmentation: argmax over products of piece probs."""
    best = [(1.0, [])] + [None] * len(s)
    for i in range(1, len(s) + 1):
        cands = [(best[j][0] * p[s[j:i]], best[j][1] + [s[j:i]])
                 for j in range(i) if s[j:i] in p and best[j] is not None]
        best[i] = max(cands) if cands else None
    if best[len(s)] is None:
        raise ValueError("no valid segmentation for: " + s)
    return best[len(s)][1]

def forward_prob(s, p):
    """P(s) = sum over ALL segmentations (marginal), via forward DP."""
    f = [1.0] + [0.0] * len(s)
    for i in range(1, len(s) + 1):
        for j in range(i):
            if s[j:i] in p:
                f[i] += f[j] * p[s[j:i]]
    return f[len(s)]

def path_prob(pieces, p):
    return math.prod(p[x] for x in pieces)


# ---- [D05] NFKC loss vs byte fallback ----

def byte_fallback(ch):
    """UTF-8 byte pieces for one char — demos the REPRESENTATION and its
    round-trip; real byte_fallback triggers only when normal pieces lack
    coverage (the coverage decision is not modeled here)."""
    return ["<0x%02X>" % b for b in ch.encode("utf-8")]

def byte_fallback_decode(pieces):
    return bytes(int(x[3:5], 16) for x in pieces).decode("utf-8")


# ---- [D08] toy tokenizer + tied embedding for the extension regression test ----

def greedy_encode(text, vocab, specials=("<eos>",)):
    """Specials are globally atomic: the text is first split on special-token
    occurrences (longest special first), then normal tokens are longest-matched
    inside each segment — a normal token can never swallow a later special."""
    specials = sorted(specials, key=len, reverse=True)
    assert all(specials), "empty special token"
    ids, i, seg_start = [], 0, 0

    def encode_segment(seg):
        out, j = [], 0
        while j < len(seg):
            for length in range(len(seg) - j, 0, -1):
                sub = seg[j:j + length]
                if sub in vocab and sub not in specials:
                    out.append(vocab[sub]); j += length; break
            else:
                raise ValueError("no coverage for: " + seg[j:])
        return out

    while i < len(text):
        for sp in specials:
            if text.startswith(sp, i):
                ids.extend(encode_segment(text[seg_start:i]))
                ids.append(vocab[sp]); i += len(sp); seg_start = i
                break
        else:
            i += 1
    ids.extend(encode_segment(text[seg_start:]))
    return ids


def main():
    if not __debug__:
        raise SystemExit("run without -O: the checks below are assert-based")
    if hasattr(sys.stdout, "reconfigure"):         # survive non-UTF-8 terminals
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

    # [D01] 256-byte reversible mapping
    assert len(B2U) == 256
    assert len(set(B2U.values())) == 256           # injective -> reversible
    assert from_units("".join(B2U[b] for b in range(256))) == bytes(range(256))
    # pin the EXACT GPT-2 mapping, not just "some bijection":
    printable = list(range(0x21, 0x7F)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    assert len(printable) == 188 and all(B2U[b] == chr(b) for b in printable)
    assert B2U[0x00] == chr(256) and B2U[0x0A] == "Ċ"   # shifted bytes, ascending order
    assert B2U[0x20] == "Ġ"                        # space byte 0x20 proxy (its familiar
    # word-leading position comes from regex chunking, not from the mapping itself)
    s = " A\n中文🙂\x00"
    assert from_units(to_units(s)).decode("utf-8") == s
    print(f"[D01] bytes_to_unicode: 256/256 bijective, 0x20 -> 'Ġ', roundtrip {s!r} OK  PASS")

    # [D02] BPE trainer + frozen-rank encoder
    merges = bpe_train({"abab": 3, "abac": 1}, num_merges=3)
    assert merges[0] == ("a", "b")                 # pair freq 7 = 2*3 + 1*1
    text = "ababac"
    assert bpe_decode(bpe_encode(text, merges)) == text
    assert bpe_encode(text, merges) == bpe_encode(text, merges)   # deterministic
    assert len(bpe_encode(text, merges)) <= len(text.encode("utf-8"))
    # rank beats runtime re-selection: with ranks [a+b, b+c], "abc" -> [ab, c]
    assert bpe_encode("abc", [("a", "b"), ("b", "c")]) == ["ab", "c"]
    # ...and the mirror case: swap the ranks and the outcome follows the ranks,
    # killing both "always leftmost pair wins" and "re-count frequencies" encoders
    assert bpe_encode("abc", [("b", "c"), ("a", "b")]) == ["a", "bc"]
    assert bpe_encode("abcbc", [("a", "b"), ("b", "c")]) == ["ab", "c", "bc"]
    assert bpe_encode("aaa", [("a", "a")]) == ["aa", "a"]   # leftmost, non-overlapping
    print(f"[D02] BPE: merges={merges}, encode({text!r})={bpe_encode(text, merges)}, "
          f"rank-conflict 'abc'->{bpe_encode('abc', [('a','b'),('b','c')])}  PASS")

    # [D03] WordPiece whole-word [UNK] failure
    assert wordpiece("abc", {"a", "##b", "##bc", "[UNK]"}) == ["a", "##bc"]
    assert wordpiece("abc", {"a", "##b", "[UNK]"}) == ["[UNK]"]   # NOT ["a","##b","[UNK]"]
    assert wordpiece("abc", {"a", "ab", "##c", "[UNK]"}) == ["ab", "##c"]  # longest FIRST piece
    # greedy does NOT backtrack: "ab" wins, dead-ends, whole word -> [UNK]
    # (a + ##bc would have worked — that's the point)
    assert wordpiece("abc", {"a", "ab", "##bc", "[UNK]"}) == ["[UNK]"]
    print("[D03] WordPiece: 'abc'->['a','##bc']; drop '##bc' -> ['[UNK]'] (whole word)  PASS")

    # [D04] Unigram Viterbi + forward probability
    p = {"a": .15, "b": .15, "ab": .40, "aba": .20, "bab": .10}
    assert viterbi("abab", p) == ["ab", "ab"]
    assert abs(forward_prob("abab", p) - 0.22350625) < 1e-12
    assert forward_prob("abab", p) >= path_prob(viterbi("abab", p), p)
    print(f"[D04] Unigram: viterbi={viterbi('abab', p)} (prob {path_prob(viterbi('abab', p), p):.4f}), "
          f"forward={forward_prob('abab', p):.8f} >= best path  PASS")

    # [D05] NFKC loss vs byte fallback
    assert unicodedata.normalize("NFKC", "Ａ①") == "A1"
    assert unicodedata.normalize("NFKC", "Ａ①") != "Ａ①"          # lossy: original gone
    assert byte_fallback("龘") == ["<0xE9>", "<0xBE>", "<0x98>"]
    assert byte_fallback_decode(byte_fallback("龘")) == "龘"       # coverage != normalization
    print("[D05] NFKC('Ａ①')='A1' (lossy); byte_fallback('龘')=<0xE9><0xBE><0x98> roundtrips  PASS")

    # [D06] prefix instability
    m = [("a", "b"), ("ab", "c")]
    assert bpe_encode("ab", m) == ["ab"]
    assert bpe_encode("abc", m) == ["abc"]
    assert bpe_encode("abc", m)[:len(bpe_encode("ab", m))] != bpe_encode("ab", m)
    print("[D06] prefix instability: enc('ab')=['ab'] but enc('abc')=['abc'] -> not a token-prefix  PASS")

    # [D07] fertility (micro-average tokens/byte) + BPB vs loss/token
    texts = ["def add(x): return x + 1", "把变量加一🙂"]
    total_bytes = sum(len(t.encode("utf-8")) for t in texts)
    char_toks = sum(len(list(t)) for t in texts)
    byte_toks = sum(len(t.encode("utf-8")) for t in texts)
    merges7 = bpe_train({t: 1 for t in texts}, num_merges=12)
    bpe_toks = sum(len(bpe_encode(t, merges7)) for t in texts)
    for t in texts:                                # toy-BPE must round-trip (char/byte trivially do)
        assert bpe_decode(bpe_encode(t, merges7)) == t
    assert byte_toks / total_bytes == 1.0          # byte tokenizer: 1 token per byte
    assert char_toks < byte_toks                   # CJK/emoji chars cost 3-4 bytes each
    assert bpe_toks < char_toks                    # merges compress further
    nll, n_bytes = 4.0, 8                          # same text, same total NLL (nats)
    loss_per_token_a, loss_per_token_b = nll / 2, nll / 4
    bpb_a = (2 * loss_per_token_a) / (n_bytes * math.log(2))   # recover BPB from loss/token
    bpb_b = (4 * loss_per_token_b) / (n_bytes * math.log(2))   # independently for each side
    assert loss_per_token_a == 2.0
    assert loss_per_token_b == 1.0
    assert bpb_a == bpb_b == 4.0 / (8 * math.log(2))
    print(f"[D07] fertility tokens/byte: byte=1.000, char={char_toks/total_bytes:.3f}, "
          f"toy-BPE={bpe_toks/total_bytes:.3f}; loss/token 2.0 vs 1.0 yet BPB both {bpb_a:.4f}  PASS")

    # [D08] vocabulary-extension regression test (CONTROL-token path; extending
    # ordinary lexical tokens intentionally changes old segmentations instead)
    vocab = {"a": 0, "b": 1, "ab": 2, "<eos>": 3}
    assert sorted(vocab.values()) == list(range(len(vocab)))  # toy assumes dense IDs
    # (real ID space may have holes; embedding may already be padded past max ID)
    dim = 4
    emb = [[float(i * dim + j) for j in range(dim)] for i in range(len(vocab))]
    head = emb                                     # tied LM head: SAME object
    old_ids = dict(vocab)
    tests = ["abab", "aab<eos>"]
    before = [greedy_encode(t, vocab) for t in tests]
    # "aab" is a semantic PROXY text for the new special token (real vocab
    # extension mean-inits a new special from the old segmentation of its
    # description text — a special's surface usually has no MEANINGFUL old
    # segmentation, though byte-complete tokenizers can segment it literally)
    pieces = greedy_encode("aab", vocab)
    assert pieces == [0, 2]                        # "a" + "ab"
    new_id = len(vocab)
    vocab["<fn>"] = new_id                         # new SPECIAL token (atomic match only)
    emb.append([sum(emb[i][j] for i in pieces) / len(pieces) for j in range(dim)])
    after = [greedy_encode(t, vocab) for t in tests]
    assert all(vocab[k] == old_ids[k] for k in old_ids)      # old IDs frozen
    assert before == after                                   # old test set re-encodes identically
    assert greedy_encode("<eos>", vocab) == [3]              # special behavior unchanged
    assert emb[new_id] == [4.0, 5.0, 6.0, 7.0]               # mean of rows 0 and 2, per dim
    assert head is emb and len(head) == len(vocab)           # tied weights still shared
    assert greedy_encode("<fn>", vocab, specials=("<eos>", "<fn>")) == [new_id]
    # specials are globally atomic: a normal token must NOT swallow a later special
    assert greedy_encode("x<eos>", {"x<eos>": 0, "x": 1, "<eos>": 2}) == [1, 2]
    # post-extension config with BOTH specials active, on text containing the new
    # surface: <fn> is matched atomically, normal tokens cannot swallow it
    assert greedy_encode("aab<fn>", vocab, specials=("<eos>", "<fn>")) == [0, 2, new_id]
    print(f"[D08] vocab extension: old IDs/encodings/specials unchanged, "
          f"new row = mean(rows {pieces}) = {emb[new_id]}, tied head intact  PASS")

    print("\nall tokenization sanity checks passed ✓")


if __name__ == "__main__":
    main()
