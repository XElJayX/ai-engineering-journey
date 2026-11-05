# Bridge lesson 11, "Use it": your chunkers + real BGE retrieval on the contract. Measure answer hits AND prompt cost.
# Run: python labs/05-23-chunking/use_it.py   (needs sentence-transformers and tiktoken)
import numpy as np
import tiktoken
from sentence_transformers import SentenceTransformer

from my_chunking import CONTRACT, chunk_fixed, chunk_recursive, parent_child

PREFIX = "Represent this sentence for searching relevant passages: "
QUERIES = [  # (question, strings that must ALL appear in what the LLM is shown)
    ("How can we cancel the contract?", ["terminate this Agreement with 30 days"]),
    ("How much do we pay each month?", ["$12,000"]),
    ("What do we get if the service is down too much?", ["credit of 10%"]),
    ("When is an invoice due?", ["due within 30 days"]),
    ("Which state's law applies?", ["Delaware"]),
    ("How long is our data kept after the contract ends?", ["delete it within 60 days"]),
    ("Who owns the machine learning models?", ["Client owns its data and models"]),
    ("Is there a cap on damages?", ["capped at the fees paid"]),
    ("Does the agreement renew on its own?", ["renews automatically"]),
    ("Where exactly must a termination notice be sent?", ["certified mail", "signature page"]),  # multi-hop: 12.2 + 2
]


# ---------------------------------------------------------------------------
# TODO: build each strategy with YOUR functions (one line each).
# Each entry is a list of (text_to_embed, text_shown_to_llm) pairs.
# For plain chunkers both are the same chunk. For parent-child: embed the child, show the parent.
# ---------------------------------------------------------------------------
def build_strategies():
    return {
        "whole document":         [(CONTRACT, CONTRACT)],
        "fixed 200":              [(chunk, chunk) for chunk in chunk_fixed(CONTRACT, 200)],   # TODO: from chunk_fixed(CONTRACT, 200)
        "fixed 200, overlap 50":  [(chunk, chunk) for chunk in chunk_fixed(CONTRACT, 200, 50)],   # TODO: from chunk_fixed(CONTRACT, 200, 50)
        "recursive 200":          [(chunk, chunk) for chunk in chunk_recursive(CONTRACT, 200)],   # TODO: from chunk_recursive(CONTRACT, 200)
        "parent 800 / child 150": [(entry["child"], entry["parent"]) for entry in parent_child(CONTRACT, 800, 150)]
    }


# ---------------------------------------------------------------------------
# Runner. Don't edit below.
# ---------------------------------------------------------------------------
def retrieve(model, pairs, query, k):
    E = model.encode([e for e, _ in pairs], normalize_embeddings=True)
    q = model.encode([PREFIX + query], normalize_embeddings=True)[0]
    shown, seen = [], set()
    for j in np.argsort(-(E @ q)):
        if pairs[j][1] not in seen:          # dedupe parents (and identical chunks)
            seen.add(pairs[j][1])
            shown.append(pairs[j][1])
        if len(shown) == k:
            break
    return shown


if __name__ == "__main__":
    strategies = build_strategies()
    if any(v is None for v in strategies.values()):
        raise SystemExit("Fill in every TODO in build_strategies() first.")
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    enc = tiktoken.get_encoding("o200k_base")
    print(f"  contract: {len(enc.encode(CONTRACT))} tokens\n")
    print(f"  {'strategy':24} {'chunks':>6}  {'hit@1':>5}  {'hit@3':>5}  {'avg prompt tokens (top-3)':>26}")
    for name, pairs in strategies.items():
        h1 = h3 = toks = 0
        misses = []
        for query, needles in QUERIES:
            top3 = retrieve(model, pairs, query, 3)
            h1 += all(n in top3[0] for n in needles)
            ok3 = all(any(n in t for t in top3) for n in needles)
            h3 += ok3
            toks += sum(len(enc.encode(t)) for t in top3)
            if not ok3:
                misses.append(query)
        n = len(QUERIES)
        print(f"  {name:24} {len(pairs):>6}  {h1 / n:>5.1f}  {h3 / n:>5.1f}  {toks / n:>26.0f}")
        for m in misses:
            print(f"  {'':24}   missed@3: {m}")
