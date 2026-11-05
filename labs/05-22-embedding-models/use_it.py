# Bridge lesson 10, "Use it": a real embedding model (BGE-small) vs BM25 vs your RRF hybrid, scored with YOUR metrics.
# Run: /usr/local/bin/python3 labs/05-22-embedding-models/use_it.py   (needs sentence-transformers)
import math
import re
from collections import Counter

import numpy as np
from sentence_transformers import SentenceTransformer

from my_retrieval import recall_and_mrr, rrf, truncate_renorm

DOCS = {
    "d01": "Troubleshooting E4021: payment gateway timeout at checkout. Wait 30 seconds and retry the payment.",
    "d02": "Troubleshooting E4012: the coupon code entered at checkout is invalid or expired.",
    "d03": "Troubleshooting E4210: card declined by the issuing bank during checkout.",
    "d04": "Refunds are issued to the original payment method within 14 days of approval.",
    "d05": "To return an item, open Orders, choose the item and select Start a return.",
    "d06": "Shipping to Canada takes 5 to 8 business days; customs fees are paid by the buyer.",
    "d07": "Express shipping arrives in 1 to 2 business days inside the continental US.",
    "d08": "Reset your password from the login page using the Forgot password link.",
    "d09": "Two-factor authentication codes expire after 10 minutes.",
    "d10": "Error E1007 means your session expired; sign in again to continue.",
    "d11": "Gift cards cannot be exchanged for cash and never expire.",
    "d12": "SKU KB-2210-BLK is the black mechanical keyboard; SKU KB-2210-WHT is the white version.",
    "d13": "Price matching applies if a competitor lists the identical item for less within 7 days of purchase.",
    "d14": "Orders can be cancelled free of charge until they are marked as shipped.",
}
QUERIES = [  # (query, relevant doc ids)
    ("error E4021 at checkout", {"d01"}),
    ("my payment keeps failing when I try to pay", {"d01", "d03"}),
    ("how long until I get my money back", {"d04"}),
    ("E1007", {"d10"}),
    ("KB-2210-WHT", {"d12"}),
    ("can I stop my order before it ships", {"d14"}),
    ("delivery time to Toronto", {"d06"}),
    ("I forgot my login", {"d08"}),
    ("coupon not working", {"d02"}),
    ("will my gift card run out", {"d11"}),
]
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "   # from the BGE model card


# ---------------------------------------------------------------------------
# Provided: a small BM25 (keyword search), so you can compare against it.
# ---------------------------------------------------------------------------
def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.ids = list(docs)
        self.toks = [tokenize(docs[i]) for i in self.ids]
        self.avg = sum(map(len, self.toks)) / len(self.toks)
        df = Counter(t for ts in self.toks for t in set(ts))
        n = len(self.toks)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.k1, self.b = k1, b

    def rank(self, query):
        q = tokenize(query)
        scores = []
        for i, ts in zip(self.ids, self.toks):
            tf = Counter(ts)
            s = sum(self.idf.get(t, 0) * tf[t] * (self.k1 + 1) / (tf[t] + self.k1 * (1 - self.b + self.b * len(ts) / self.avg))
                    for t in q)
            scores.append((s, i))
        return [i for s, i in sorted(scores, key=lambda x: (-x[0], x[1]))]


# ---------------------------------------------------------------------------
# Your TODOs (each is ONE line).
# ---------------------------------------------------------------------------
def dense_rank(model, doc_matrix, query, use_prefix):
    # doc_matrix: (n_docs, d) unit-length BGE embeddings of DOCS, in DOCS order.
    # TODO 1: build the text to embed: the query, with BGE_QUERY_PREFIX in front if use_prefix is True
    text = BGE_QUERY_PREFIX + query if use_prefix else query 
    q = model.encode([text], normalize_embeddings=True)[0]
    scores = doc_matrix @ q
    ids = list(DOCS)
    return [ids[j] for j in np.argsort(-scores)]


def hybrid_rank(dense_list, bm25_list):
    # TODO 2: fuse the two ranked lists with your rrf().
    return rrf([dense_list, bm25_list])


# ---------------------------------------------------------------------------
# Runner. Don't edit below.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    D = model.encode(list(DOCS.values()), normalize_embeddings=True)
    bm25 = BM25(DOCS)
    rel = [r for _, r in QUERIES]

    runs = {
        "BM25 (keywords)": [bm25.rank(q) for q, _ in QUERIES],
        "Dense, no prefix": [dense_rank(model, D, q, False) for q, _ in QUERIES],
        "Dense + prefix": [dense_rank(model, D, q, True) for q, _ in QUERIES],
    }
    runs["Hybrid RRF (dense+prefix, BM25)"] = [hybrid_rank(d, b) for d, b in zip(runs["Dense + prefix"], runs["BM25 (keywords)"])]

    print(f"  {'retriever':34} recall@1  recall@3   MRR")
    for name, ranked in runs.items():
        r1, mrr = recall_and_mrr(ranked, rel, 1)
        r3, _ = recall_and_mrr(ranked, rel, 3)
        print(f"  {name:34} {r1:8.2f}  {r3:8.2f}  {mrr:6.3f}")

    print("\n  Per query: rank of the first relevant doc (1 = perfect)")
    print(f"  {'query':45} " + "  ".join(f"{n.split()[0]:>6}" for n in runs))
    for qi, (q, r) in enumerate(QUERIES):
        firsts = [next((k for k, d in enumerate(runs[n][qi], 1) if d in r), "-") for n in runs]
        print(f"  {q:45} " + "  ".join(f"{str(f):>6}" for f in firsts))

    print("\n  Matryoshka check: truncate BGE-small (NOT Matryoshka-trained) and re-run dense + prefix")
    for dim in (384, 128, 64, 32):
        Dt = truncate_renorm(D, dim)
        ranked = []
        for q, _ in QUERIES:
            qv = truncate_renorm(model.encode([BGE_QUERY_PREFIX + q], normalize_embeddings=True), dim)[0]
            ranked.append([list(DOCS)[j] for j in np.argsort(-(Dt @ qv))])
        r1, mrr = recall_and_mrr(ranked, rel, 1)
        print(f"    dim {dim:>3}: recall@1 {r1:.2f}   MRR {mrr:.3f}   index size per vector {dim * 4} bytes")
