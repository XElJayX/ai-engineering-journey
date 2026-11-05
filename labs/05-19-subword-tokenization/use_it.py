# Bridge lesson 9, "Use it": measure a real tokenizer (GPT-4o's o200k_base) before trusting your intuition.
# Run: /usr/local/bin/python3 labs/05-19-subword-tokenization/use_it.py   (needs tiktoken)
import tiktoken

enc = tiktoken.get_encoding("o200k_base")

# ---------------------------------------------------------------------------
# Part 1: PREDICT FIRST. Replace each None with how many tokens you think o200k_base uses.
# Don't run the script until all six are filled in.
# ---------------------------------------------------------------------------
PREDICTIONS = {
    "The quick brown fox jumps over the lazy dog.": 10,
    "strawberry": 3,
    "12345678": 3,
    "こんにちは、元気ですか": 5,                 # "Hello, how are you?" in Japanese
    "👍": 1,
    "👨‍👩‍👧‍👦": 1,                                   # family emoji: 4 people joined by invisible joiners
}

# ---------------------------------------------------------------------------
# Part 2: a teammate's context-budget check for a RAG prompt. It has a silent bug.
# Fix count_tokens so the budget check is correct. (Keep the function signature.)
# ---------------------------------------------------------------------------
CONTEXT_LIMIT = 40


def count_tokens(text):
    # "Everyone knows a token is about 0.75 words."

    return int(len(enc.encode(text)))


def fits_in_context(chunks, question):
    total = sum(count_tokens(c) for c in chunks) + count_tokens(question)
    return total <= CONTEXT_LIMIT, total


# ---------------------------------------------------------------------------
# Runner. Don't edit below.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if any(v is None for v in PREDICTIONS.values()):
        raise SystemExit("Fill in every prediction in PREDICTIONS first.")
    print("Part 1: predicted vs actual (o200k_base)")
    for text, guess in PREDICTIONS.items():
        ids = enc.encode(text)
        pieces = [enc.decode_single_token_bytes(t) for t in ids]
        shown = [p.decode("utf-8", errors="replace") for p in pieces]
        mark = "OK " if guess == len(ids) else "   "
        print(f"  {mark}{text!r:50} guess {guess:>2}  actual {len(ids):>2}  {shown}")

    print("\nPart 2: context budget")
    chunks = [
        "Refunds are issued within 14 days of a return request.",
        "返品リクエストから14日以内に返金されます。",             # same sentence in Japanese
        "def refund(order_id): return api.post(f'/refunds/{order_id}')",
    ]
    question = "How long do refunds take?"
    ok, est = fits_in_context(chunks, question)
    real = sum(len(enc.encode(c)) for c in chunks) + len(enc.encode(question))
    print(f"  your count_tokens total: {est}   real o200k total: {real}   limit: {CONTEXT_LIMIT}")
    print(f"  fits_in_context says {ok}; truth is {real <= CONTEXT_LIMIT}")
    for c in chunks:
        print(f"    {len(c.split()):>2} words -> {count_tokens(c):>3} by your function, {len(enc.encode(c)):>3} real   {c[:40]!r}")
    print("  PASS: budget check matches reality" if est == real else "  FAIL: your count does not match the tokenizer")
