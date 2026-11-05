# Lesson 11/07 Advanced RAG: BM25 from the formula, Reciprocal Rank Fusion, hybrid retrieval, cross-encoder rerank.
# Run: python labs/11-07-advanced-rag/my_hybrid.py      (self-checks are pure numpy; no model needed)
# Reuses: exact_topk (your 11/04 code) for the vector side.
# The "Use it" run (experiment.py) compares vector / BM25 / hybrid / hybrid+rerank on fictional 10-K chunks.
import math
import os
import re
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "11-04-embeddings"))
from my_vector_index import exact_topk  # noqa: E402  (your 11/04 code)


# ---------------------------------------------------------------------------
# Your TODOs
# ---------------------------------------------------------------------------

def tokenize(text):
    # TODO 1: lowercase, then keep runs of letters/digits only (regex [a-z0-9]+). Punctuation splits tokens.
    #   "Halcyon's capex (FY2024)" -> ["halcyon", "s", "capex", "fy2024"]
    #   "case No. 2:24-cv-03817"   -> ["case", "no", "2", "24", "cv", "03817"]
    # (The lesson's text.split() would keep "halcyon's" and "(fy2024)" as tokens that never match a query.)  
    return re.findall(r"[a-z0-9]+",text.lower())

    


class BM25:
    def __init__(self, k1=1.2, b=0.75):
        self.k1, self.b = k1, b

    def fit(self, texts):
        # TODO 2: precompute everything that doesn't depend on the query.
        self.N      = len(texts)
        self.tf     = [Counter(tokenize(t)) for t in texts]  
        self.dl     = np.array( [sum(counts.values()) for counts in self.tf], dtype = float)
        self.avgdl = self.dl.mean()
        self.df     = {}
        for counts in self.tf:
            for term in counts:
                self.df[term] = self.df.get(term, 0) + 1
        return self

    def idf(self, term):
        # Given. Lucene's BM25 IDF: rare terms get big weights, a term in every doc gets ~0, never negative.
        df = self.df.get(term, 0)
        return math.log((self.N - df + 0.5) / (df + 0.5) + 1)

    def scores(self, query):
        # TODO 3: return a float np.array of shape (N,), the BM25 score of the query against every doc.
        #   For each UNIQUE query term t (a repeated query word counts once):
        #       for each doc d:  idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl[d] / avgdl))
        #   where tf = count of t in d (0 -> contributes 0). Sum over terms.
        #   Terms not in any doc contribute 0. Unknown words must not crash.
        scores = np.zeros(self.N,dtype=float)
        tokens = set(tokenize(query))
        for token in tokens:
            term_idf = self.idf(token)
            for d in range(self.N):
                tf = self.tf[d].get(token,0)

                if tf == 0:
                    continue
                length_norm = 1 - self.b + self.b * (self.dl[d]/self.avgdl)

                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * length_norm

                scores[d]+= term_idf * numerator / denominator
        return scores



    def topk(self, query, k):
        # Given: ids of the k best docs that match at least one query term, best first (ties -> lower id).
        # (A real keyword index only returns docs containing a query word; zero-score docs are not "hits".)
        s = self.scores(query)
        order = np.lexsort((np.arange(len(s)), -s))
        return [int(i) for i in order[:k] if s[i] > 0]


def rrf(ranked_lists, k=60):
    # TODO 4: Reciprocal Rank Fusion. ranked_lists = [[doc ids, best first], [...], ...].
    #   score(d) = sum over lists containing d of 1 / (k + rank), with rank starting at 1 for the best.
    #   A doc missing from a list gets nothing from that list.
    # Return [(doc_id, score), ...] sorted by score descending; ties -> lower doc_id first.
    scores ={}
    for ranked_list in ranked_lists:
        for rank,doc_id in enumerate(ranked_list, start =1):
            contribution = 1 / (k + rank)
            scores[doc_id] = scores.get(doc_id,0) + contribution
    return sorted(scores.items(), key = lambda item: (-item[1],item[0]))
    

def hybrid(query, q_vec, D, bm25, n=50, k=60):
    # TODO 5: retrieve n candidates from EACH retriever, fuse with rrf, return the fused doc ids (best first).
    #   vector side: your exact_topk on q_vec reshaped to (1, d) against D (rows normalized) -> ids best first
    #   keyword side: bm25.topk(query, n)
    # Return a plain list of ints.
    q = np.asarray(q_vec).reshape(1, -1)
    vector_ids, vector_scores = exact_topk(q, D, n)
    keyword_ids = bm25.topk(query, n)
    vector_ids = vector_ids.flatten().tolist()
    fused = rrf([vector_ids,keyword_ids],k=k)
    return [doc_id for doc_id,score in fused]



def rerank(query, ids, texts, cross_score, top_k=5):
    # TODO 6: the second stage. cross_score(pairs) takes [(query, text), ...] and returns one float per pair
    #   (higher = more relevant). Score EVERY candidate in ONE call (batching is the whole point of the
    #   shortlist), then return the top_k ids by that score, best first (ties -> keep the incoming order).
    pairs = [(query,texts[doc_id]) for doc_id in ids]
    scores = cross_score(pairs)
    order = sorted(range(len(ids)), key = lambda i: -scores[i])
    return [ids[i] for i in order[:top_k]]
# ---------------------------------------------------------------------------
# Self-checks. Don't edit below.
# ---------------------------------------------------------------------------

def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"   ({detail})" if detail and not cond else ""))
    return cond


if __name__ == "__main__":
    ok = []
    ok.append(check("1 tokenize punctuation", tokenize("Halcyon's capex (FY2024)") == ["halcyon", "s", "capex", "fy2024"], str(tokenize("Halcyon's capex (FY2024)"))))
    ok.append(check("1 tokenize codes", tokenize("case No. 2:24-cv-03817") == ["case", "no", "2", "24", "cv", "03817"]))

    docs = ["the cat sat on the mat",
            "the dog sat",
            "cat cat cat food",
            "a bird flew over the house and the barn and the field"]
    bm = BM25().fit(docs)
    ok.append(check("2 corpus stats", bm.N == 4 and bm.df["the"] == 3 and bm.df["cat"] == 2 and bm.df["sat"] == 2
                    and np.allclose(bm.dl, [6, 3, 4, 12]) and abs(bm.avgdl - 6.25) < 1e-9, f"df={dict(bm.df)}"))
    ok.append(check("2 tf counts", bm.tf[2]["cat"] == 3 and bm.tf[0]["the"] == 2))

    s = bm.scores("cat")
    want = [0.6931471806 * 2.2 / (1 + 1.2 * (0.25 + 0.75 * 6 / 6.25)), 0.0,
            0.6931471806 * 3 * 2.2 / (3 + 1.2 * (0.25 + 0.75 * 4 / 6.25)), 0.0]
    ok.append(check("3 single-term score matches the formula", s.shape == (4,) and np.allclose(s, want, atol=1e-6), f"{np.round(s, 4)} vs {np.round(want, 4)}"))
    ok.append(check("3 tf saturates (3x the count is NOT 3x the score)", s[2] / s[0] < 2.0, f"ratio {s[2] / max(s[0], 1e-9):.2f}"))
    ok.append(check("3 repeated query word counts once", np.allclose(bm.scores("cat cat"), s)))
    ok.append(check("3 unknown words score 0, no crash", np.allclose(bm.scores("zebra"), 0)))
    s2 = bm.scores("dog sat")
    ok.append(check("3 multi-term sums terms; rare term dominates", int(np.argmax(s2)) == 1 and s2[1] > s2[0] > 0 and s2[2] == 0, str(np.round(s2, 4))))
    ok.append(check("3 longer doc gets less per occurrence", bm.scores("the")[3] < bm.scores("the")[0]))

    f = rrf([[7, 3, 9], [3, 5]], k=60)
    d = dict(f)
    ok.append(check("4 rrf sums 1/(k+rank), rank from 1", abs(d[3] - (1 / 62 + 1 / 61)) < 1e-12 and abs(d[7] - 1 / 61) < 1e-12 and abs(d[5] - 1 / 62) < 1e-12, str(f)))
    ok.append(check("4 rrf order + ties by lower id", [i for i, _ in f] == [3, 7, 5, 9], str([i for i, _ in f])))
    ok.append(check("4 in both lists beats #1 in one", rrf([[1, 2], [2, 3]])[0][0] == 2))

    rng = np.random.default_rng(0)
    D = rng.normal(size=(6, 8)).astype(np.float32)
    D /= np.linalg.norm(D, axis=1, keepdims=True)
    texts = ["alpha report", "beta report", "gamma", "delta code X9", "epsilon", "zeta"]
    hb = BM25().fit(texts)
    q_vec = D[0] + 0.01
    ids = hybrid("code X9", q_vec, D, hb, n=3)
    ok.append(check("5 hybrid returns fused ints", isinstance(ids, list) and all(isinstance(i, int) for i in ids) and len(set(ids)) == len(ids)))
    ok.append(check("5 hybrid keeps the vector #1 and the keyword #1", ids[:2] == [0, 3] or ids[:2] == [3, 0], str(ids)))

    calls = []
    def fake_ce(pairs):
        calls.append(len(pairs))
        return [len(t) if "code" in t else -1.0 for _, t in pairs]
    out = rerank("code", [0, 3, 1, 5], texts, fake_ce, top_k=2)
    ok.append(check("6 rerank orders by cross score", out == [3, 0], str(out)))
    ok.append(check("6 rerank scores in ONE batch call", calls == [4], f"calls={calls}"))
    out = rerank("q", [4, 2, 5], texts, lambda pairs: [0.0] * len(pairs), top_k=3)
    ok.append(check("6 ties keep incoming order", out == [4, 2, 5], str(out)))

    print(f"\n{sum(ok)}/{len(ok)} checks passed")
