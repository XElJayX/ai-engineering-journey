# 05/19 Subword Tokenization (BPE)

> Everything about LLMs (cost, context, speed, odd failures) is measured in tokens. The tokenizer decides what a token is.

**Done:** 2026-10-24 · Quiz 3/3 · Bridge lesson 9 of 14
**Lesson:** [phases/05-nlp-foundations-to-advanced/19-subword-tokenization](../phases/05-nlp-foundations-to-advanced/19-subword-tokenization/)
**My code:**
- [`labs/05-19-subword-tokenization/my_bpe.py`](../labs/05-19-subword-tokenization/my_bpe.py): BPE trainer and encoder, 6/6 checks
- [`use_it.py`](../labs/05-19-subword-tokenization/use_it.py): measuring tiktoken and fixing a token-budget bug

**Run it:**
- `python3 labs/05-19-subword-tokenization/my_bpe.py` (stdlib only)
- `/usr/local/bin/python3 labs/05-19-subword-tokenization/use_it.py` (needs tiktoken)

---

## Remember these 3 things

1. **BPE learns its pieces from counts.** Start from characters (or bytes), repeatedly merge the most frequent adjacent pair, and record the merges **in order**. Common words become one token, and rare words split into pieces. Nothing is ever `[UNK]`.
2. **Never estimate tokens, count them with the model's own tokenizer.** "Words / 0.75" was right for English, but **1 Japanese "word" was 15 tokens** and 4 words of code were 16.
3. **Tokenization explains famous LLM weirdness.**
   - `strawberry` → `st|raw|berry`: the model never sees letters, which is why it miscounts the r's
   - `12345678` → `123|456|78`: weak arithmetic
   - 👨‍👩‍👧‍👦 is **11 tokens** of byte fragments

---

## In plain English

Three ways to chop text:

| Approach | Vocabulary | "unbelievably" | Problem |
|----------|-----------|----------------|---------|
| Words | ~50k words | `[UNK]` if it's unseen | Typos, new words and names all lose their meaning |
| Characters | ~100 | 12 tokens | Sequences are 5× longer, and attention costs (5×)² = **25×** more |
| **Subwords** | 50k to 200k pieces | `un` `bel` `ievably` (3) | The best of both: no UNK, and short sequences |

**Analogy.** Text messaging shorthand. Very common phrases get their own shortcut ("brb", "lol"), and everything else gets spelled out from smaller parts. BPE learns which shortcuts are worth having by counting what shows up most.

---

## BPE by hand

Corpus `{low: 5, lower: 2, newest: 6, widest: 3}`, with `</w>` marking the end of a word:

```text
l o w </w>          ×5
l o w e r </w>      ×2
n e w e s t </w>    ×6
w i d e s t </w>    ×3
```

**Pair counts** (weighted by how often each word appears):
- `e s` = `s t` = `t </w>` = **9** (6 from newest + 3 from widest)
- `w e` = 8
- `l o` = **7** (5 + 2, *not* 9. I slipped on this one.)

**The first merges:**
1. `e s` → `es`
2. `es t` → `est`
3. `est </w>` → `est</w>`, the superlative suffix, discovered from counts alone

**Why `</w>`:** the `est` at the *end* of "newest" and the `est` at the *start* of "estimate" become different tokens. GPT-style tokenizers do the same thing with a leading space: `" hello"` and `"hello"` are different tokens.

**Why merges apply in training order:** the merges `[(a,b), (b,c)]` turn "abc" into `[ab, c]`. Apply them in a different order and you get `[a, bc]`, a sequence the model never saw in training. That's the same failure as **tokenizer drift**.

---

## My code

```python
def get_pair_counts(vocab):
    pair_counts = Counter()
    for word, freq in vocab.items():
        for i in range(len(word) - 1):
            pair_counts[(word[i], word[i + 1])] += freq      # weighted by word frequency
    return pair_counts

def merge_pair(symbols, pair):
    result, i = [], 0
    while i < len(symbols):
        if i < len(symbols) - 1 and symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
            result.append(pair[0] + pair[1])
            i += 2                                             # skip both, so no overlapping merges
        else:
            result.append(symbols[i])
            i += 1
    return tuple(result)

def train_bpe(corpus, num_merges):
    vocab = {word_to_symbols(w): c for w, c in corpus.items()}
    merges = []
    for _ in range(num_merges):
        pair_counts = get_pair_counts(vocab)
        if not pair_counts:
            break
        best = min((-count, pair) for pair, count in pair_counts.items())[1]   # max count, ties → smallest pair
        merges.append(best)
        vocab = {merge_pair(w, best): c for w, c in vocab.items()}
    return merges

def encode(word, merges):
    symbols = word_to_symbols(word)
    for pair in merges:                                        # in TRAINING order
        symbols = merge_pair(symbols, pair)
    return list(symbols)
```

**Results:**
- 60 merges compressed 522 characters into 207 tokens (2.3 tokens per word), and decoding round-trips exactly.
- The unseen word "lowest" became `low` + `est</w>`.

**Review (scaling):**
1. `encode` scans *every* merge for every word, which means about 200k merges for GPT-4o's vocabulary. Production keeps `rank[pair]` and repeatedly merges the lowest-ranked adjacent pair *in the word*: at most `len(word)` steps, same output.
2. `train_bpe` recounts every pair after each merge. Real trainers update only the pairs that touch the merged symbol.
3. The `i += 2` is essential. With `i += 1`, merges overlap: `aaaa` becomes `aa aa aa a`, and training learns garbage like `ess`.

---

## Real tokenizer measurements (o200k_base, GPT-4o)

| Text | Tokens | Pieces | Lesson |
|------|--------|--------|--------|
| The quick brown fox … dog. | 10 | one per word, with the leading space attached | English is cheap |
| strawberry | 3 | `st` `raw` `berry` | The model can't see letters, hence the "how many r's" failures |
| 12345678 | 3 | `123` `456` `78` | Digit chunks don't line up with place value, hence weak arithmetic |
| こんにちは、元気ですか | 6 | `こんにちは` `、` `元` `気` `です` `か` | Non-English costs more |
| 👍 | 1 | `👍` | Common enough to get its own token (3 in the older cl100k) |
| 👨‍👩‍👧‍👦 | **11** (18 in cl100k) | byte fragments `�` | 7 code points = 25 UTF-8 bytes. A rare combination stays split. |

**Why the family emoji is 11 tokens:** 4 emoji (4 bytes each) plus 3 invisible zero-width joiners (3 bytes each) = 25 bytes. Byte-level BPE only merges byte sequences it saw often. Each `�` token is **half an emoji**, not valid text on its own.

**Consequences:**
- **Streaming:** buffer bytes until they form valid UTF-8, or the UI shows `��`.
- **Truncation:** cutting at N tokens can slice an emoji or a Japanese character in half.

---

## The silent bug I fixed

```python
def count_tokens(text):
    return int(len(text.split()) / 0.75)     # BUG: the "1 token ≈ 0.75 words" rule
# fix:
    return len(enc.encode(text))             # count with the model's own tokenizer
```

| Chunk | Words | Rule of thumb | Real |
|-------|-------|---------------|------|
| English refund sentence | 10 | 13 | 13 |
| Same sentence in Japanese | **1** | 1 | **15** |
| One line of Python | 4 | 5 | **16** |

The rule of thumb gave 25 total and said the prompt **fits** a 40-token limit. The real total was 50, so it **overflows**. In production that means a truncated prompt or an API error.

---

## Algorithms and libraries (know the names)

| Name | How it builds the vocabulary | Used by |
|------|----------------------------|---------|
| **BPE** | Merge the most *frequent* pair | GPT family, Llama, Mistral, Qwen |
| **Byte-level BPE** | BPE over 256 raw bytes, so any input encodes with no UNK | GPT-2 onward, Llama |
| **WordPiece** | Merge the pair that most increases corpus *likelihood* | BERT |
| **Unigram** | Start huge, then *prune* the tokens whose removal hurts likelihood least | T5, Gemma |

| Library | Does | Use for |
|---------|------|---------|
| **tiktoken** | Encodes only, fast (Rust). No training. | Counting tokens and cost for OpenAI models |
| **SentencePiece** | Trains BPE or Unigram on raw text. Space becomes `▁`. | Multilingual tokenizers |
| **HF Tokenizers** | Trains and serves | Most open models (`tokenizer.json`) |

**Vocabulary size heuristic:** about 32k for models under 1B parameters, 50 to 100k for 1 to 10B, and 200k+ for multilingual or frontier models. Bigger means fewer tokens per text, but a bigger embedding table and output softmax.

---

## Where this shows up in AI engineering

- **Cost estimation and context budgeting:** count with the right tokenizer. `cl100k` counts are wrong for GPT-4o, and every provider has its own tokenizer.
- **RAG chunking** (next lessons): chunk size is measured in tokens, not characters or words.
- **Multilingual products:** the same feature costs more, and fills the context faster, for non-English users.
- **Debugging model failures:** letter counting, arithmetic, and odd behaviour around rare strings often trace back to tokenization.
- **Deploying fine-tunes:** ship the exact tokenizer the model was trained with. **Hash-check `tokenizer.json` in CI** to catch drift.
- **Streaming UIs:** buffer partial UTF-8.

## Mistakes to avoid

- Estimating tokens with word counts or characters ÷ 4.
- Using the wrong model's tokenizer to count.
- Merging overlapping pairs in BPE (`i += 1` instead of `i += 2`).
- Applying merges out of training order.
- Decoding streamed tokens one at a time without buffering bytes.

## Words

- **Token:** a vocabulary piece. The model's unit of input and output.
- **Merge list:** the ordered `(a, b) → ab` rules that BPE learned.
- **Byte-level:** the base alphabet is the 256 byte values, so any text encodes.
- **Code point / ZWJ:** a single Unicode character, and the invisible joiner that glues emoji together.
- **Tokenizer drift:** serving with a different tokenizer than the one used in training. Different IDs, silently worse output.

## Connects to

- **Before:** [03/05 Loss Functions](03-05-loss-functions.md). The LLM's softmax runs over these V tokens, and a loss of ln(V) means guessing (see [01/09](01-09-information-theory.md)).
- **Next:** 05/22 Embedding Models Deep Dive. Turning token sequences into vectors for retrieval.

---

## Self-test

1. Why do subword tokenizers beat both word-level and character-level ones?
2. Corpus `{low: 5, lower: 2, newest: 6, widest: 3}`: what are the top pair count, and the first 3 merges?
3. Why must encoding apply merges in training order?
4. Why can an LLM miscount the r's in "strawberry"?
5. Why is 👨‍👩‍👧‍👦 11 tokens, and what does that mean for a streaming chat UI?
6. What's wrong with `tokens ≈ words / 0.75` in a RAG budget check?
7. tiktoken vs SentencePiece: which one trains?

<details>
<summary>Answers</summary>

1. No `[UNK]` (rare words decompose into known pieces) **and** short sequences (common words are 1 token). Character-level is about 25× more attention compute on a 5× longer sequence.
2. 9 (`es`, `st`, `t</w>`). The merges are `es`, then `est`, then `est</w>`.
3. The model was trained on the token sequences produced by that order. A different order gives different, unfamiliar tokens: the same failure as tokenizer drift.
4. It sees `st|raw|berry`, three opaque IDs, never individual letters.
5. It's 7 code points (4 emoji plus 3 joiners), 25 UTF-8 bytes, and a rare combination without merges. Single tokens are partial bytes, so buffer them until the bytes form valid UTF-8 before displaying.
6. It's badly wrong for non-English text and code. 1 Japanese "word" was 15 tokens, and 4 words of code were 16, so the check said "fits" when the prompt overflowed. Count with the model's own tokenizer.
7. SentencePiece (and HF Tokenizers) train. tiktoken only encodes.

</details>
