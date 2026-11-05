# Bridge lesson 9: Subword Tokenization. Train a BPE tokenizer from scratch and encode with it.
# Run: python3 labs/05-19-subword-tokenization/my_bpe.py   (stdlib only)
# A "word" is a tuple of symbols, e.g. ("l", "o", "w", "</w>"). A vocab maps each word tuple to its count.
from collections import Counter

END = "</w>"


def word_to_symbols(word):
    return tuple(word) + (END,)


def decode(tokens):
    # Provided: join the pieces back and drop the end-of-word marker.
    return "".join(tokens).replace(END, "")


# ---------------------------------------------------------------------------
# Your TODOs
# ---------------------------------------------------------------------------

def get_pair_counts(vocab):
    # vocab: {("l","o","w","</w>"): 5, ...}
    # Return a Counter of adjacent pairs, weighted by word frequency: {("l","o"): 7, ...}
    pair_counts = Counter()
    for word, freq in vocab.items():
        for i in range(len(word)-1):
            pairs = (word[i],word[i+1])
            pair_counts[pairs] += freq
    return pair_counts
        

    
def merge_pair(symbols, pair):
    # symbols: one word as a tuple of symbols. pair: (a, b).
    # Return a new tuple where every adjacent (a, b) is replaced by the single symbol a+b.
    # Scan left to right; after merging at position i, continue from i + 2 (no overlapping merges).
    #   merge_pair(("a","a","a","a"), ("a","a")) -> ("aa","aa")
    result = []
    i = 0
    while i < len(symbols):
        if i < len(symbols) - 1 and symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
            result.append(pair[0] + pair[1])
            i += 2
        else:
            result.append(symbols[i])
            i += 1
    return tuple(result)

def train_bpe(corpus, num_merges):
    # corpus: {"low": 5, "lower": 2, ...}. Return the ORDERED list of merges, e.g. [("e","s"), ("es","t"), ...].
    # Each round: count pairs, pick the most frequent, record it, apply it to every word in the vocab.
    # Ties: pick the pair that comes first in sorted() order (so results are deterministic).
    # Stop early if no pairs are left.
    vocab = {word_to_symbols(word): count for word, count in corpus.items()}
    merges = []
    for _ in range(num_merges):
        pair_counts = get_pair_counts(vocab)
        if not pair_counts:
            break
        most_common_pair = min((-count, pair) for pair, count in pair_counts.items())[1]
        merges.append(most_common_pair)
        vocab = {merge_pair(word, most_common_pair): count for word, count in vocab.items()}
    return merges


def encode(word, merges):
    # Split the word into symbols (word_to_symbols), then apply each merge IN TRAINING ORDER.
    # Return a list of string tokens, e.g. ["low", "est</w>"].
    symbols = word_to_symbols(word)
    for pair in merges:
        symbols = merge_pair(symbols, pair)
    return list(symbols)


# ---------------------------------------------------------------------------
# Self-checks. Don't edit below this line; make them pass.
# ---------------------------------------------------------------------------

TOY = {"low": 5, "lower": 2, "newest": 6, "widest": 3}

TEXT = """the model reads the prompt and predicts the next token . the tokenizer splits
the prompt into tokens before the model reads anything . longer prompts cost more tokens
and more tokens cost more money . retrieval adds chunks to the prompt so the prompt grows .
the tokenizer learned its pieces from training text so common words become single tokens
while rare words split into smaller pieces . tokenization decides cost , context and speed .
models predict tokens , tokenizers produce tokens , and engineers count tokens ."""


def check(name, fn):
    try:
        ok, detail = fn()
    except NotImplementedError:
        ok, detail = False, "not implemented yet"
    except Exception as e:
        ok, detail = False, f"{type(e).__name__}: {e}"
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    return ok


def t_pairs():
    c = get_pair_counts({word_to_symbols(w): n for w, n in TOY.items()})
    want = {("e", "s"): 9, ("s", "t"): 9, ("t", END): 9, ("l", "o"): 7, ("w", "e"): 8, ("n", "e"): 6}
    ok = all(c[k] == v for k, v in want.items()) and len(c) == 14
    return ok, f"es={c[('e', 's')]}, lo={c[('l', 'o')]}, we={c[('w', 'e')]}, distinct pairs={len(c)}"


def t_merge():
    a = merge_pair(("a", "a", "a", "a"), ("a", "a"))
    b = merge_pair(("n", "e", "w", "e", "s", "t", END), ("e", "s"))
    c = merge_pair(("e", "x", "s"), ("e", "s"))
    ok = a == ("aa", "aa") and b == ("n", "e", "w", "es", "t", END) and c == ("e", "x", "s")
    return ok, f"aaaa -> {a}, newest -> {b}, non-adjacent untouched -> {c}"


def t_train_toy():
    m = train_bpe(TOY, 3)
    return m == [("e", "s"), ("es", "t"), ("est", END)], f"first 3 merges: {m}"


def t_order_matters():
    fwd = encode("abc", [("a", "b"), ("b", "c")])
    rev = encode("abc", [("b", "c"), ("a", "b")])
    return fwd == ["ab", "c", END] and rev == ["a", "bc", END], f"[ab, bc] -> {fwd}; [bc, ab] -> {rev}"


def t_unseen():
    merges = train_bpe(TOY, 10)
    lowest = encode("lowest", merges)
    alien = encode("qz", merges)
    ok = decode(lowest) == "lowest" and len(lowest) <= 3 and alien == ["q", "z", END]
    return ok, f"'lowest' (never seen) -> {lowest}; 'qz' -> {alien} (no [UNK], ever)"


def t_compression():
    words = TEXT.split()
    merges = train_bpe(Counter(words), 60)
    chars = sum(len(w) + 1 for w in words)
    toks = sum(len(encode(w, merges)) for w in words)
    roundtrip = all(decode(encode(w, merges)) == w for w in words)
    return roundtrip and toks < 0.45 * chars, f"{len(words)} words: {chars} char-level symbols -> {toks} BPE tokens ({toks / len(words):.2f} tokens/word); round-trip OK={roundtrip}"


if __name__ == "__main__":
    tests = [
        ("pair counts weighted by word frequency", t_pairs),
        ("merge_pair: left to right, no overlaps", t_merge),
        ("train_bpe discovers es -> est -> est</w>", t_train_toy),
        ("encode applies merges in training order", t_order_matters),
        ("unseen words decompose, nothing is unknown", t_unseen),
        ("60 merges compress text and decode round-trips", t_compression),
    ]
    passed = sum(check(name, fn) for name, fn in tests)
    print(f"\n  {passed}/{len(tests)} checks passed")
