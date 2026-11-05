# Bridge lesson 10: Embedding Models. Build the retrieval pieces: Matryoshka truncation, MaxSim, RRF, eval metrics.
# Run: /usr/local/bin/python3 labs/05-22-embedding-models/my_retrieval.py   (needs numpy)
import numpy as np
from collections import defaultdict


# ---------------------------------------------------------------------------
# Your TODOs
# ---------------------------------------------------------------------------

def truncate_renorm(V, dim):
    # V: (n, d) array of embeddings (rows). Keep the first `dim` columns, then make every row unit length again,
    # so that a dot product equals cosine similarity. Return shape (n, dim).
    truncated = V[:,:dim]
    fnorms = np.linalg.norm(truncated, axis=1, keepdims=True)
    truncated = truncated / np.maximum(fnorms, 1e-12)
    return truncated




def maxsim(Q, D):
    # Late-interaction (ColBERT) score. Q: (q_tokens, d) query token vectors, D: (d_tokens, d) doc token vectors,
    # all rows already unit length. For EACH query token, take its best dot product over all doc tokens; return the sum.
    # Hint: one matrix multiply gives every query-token x doc-token similarity at once.
    sim = Q @ D.T
    maximum = np.max(sim,axis=1)
    total = np.sum(maximum)
    return total



def rrf(rankings, k=60):
    # Reciprocal Rank Fusion. rankings: list of ranked lists of doc ids, best first, e.g. [["d3","d1"], ["d1","d7"]].
    # score(doc) = sum over the lists that contain it of 1 / (k + rank), rank starting at 1. Missing from a list = adds 0.
    # Return all doc ids sorted by fused score, best first. Ties: smaller doc id first.
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] += 1 / (k + rank)
    sorted_docs = sorted(scores, key=lambda doc: (-scores[doc], doc))
    return sorted_docs

def recall_and_mrr(ranked, relevant, k):
    # ranked: list of ranked doc-id lists, one per query. relevant: list of sets of relevant ids, one per query.
    # recall@k = average over queries of (relevant ids in the top k) / (number of relevant ids)
    # MRR      = average over queries of 1 / (rank of the FIRST relevant doc), or 0 if none appears anywhere.
    # Return (recall_at_k, mrr) as floats.
    total_recall = 0.0
    total_mrr = 0.0
    num_queries = len(ranked)
    for query_ranked, query_relevant in zip(ranked, relevant):
        top_k = query_ranked[:k]
        num_relevant = len(query_relevant)
        if num_relevant > 0:
            relevant_in_top_k = sum(1 for doc_id in top_k if doc_id in query_relevant)
            total_recall += relevant_in_top_k / num_relevant
        else:
            total_recall += 0.0
        first_relevant_rank = next((rank for rank, doc_id in enumerate(query_ranked, start=1) if doc_id in query_relevant), None)
        if first_relevant_rank is not None:
            total_mrr += 1 / first_relevant_rank
        else:
            total_mrr += 0.0
    recall_at_k = total_recall / num_queries
    mrr = total_mrr / num_queries
    return recall_at_k, mrr


# ---------------------------------------------------------------------------
# Self-checks. Don't edit below this line; make them pass.
# ---------------------------------------------------------------------------

def check(name, fn):
    try:
        ok, detail = fn()
    except NotImplementedError:
        ok, detail = False, "not implemented yet"
    except Exception as e:
        ok, detail = False, f"{type(e).__name__}: {e}"
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    return ok


def unit(M):
    return M / np.linalg.norm(M, axis=-1, keepdims=True)


def t_truncate():
    rng = np.random.default_rng(0)
    V = unit(rng.normal(size=(5, 64)))
    T = truncate_renorm(V, 16)
    norms = np.linalg.norm(T, axis=1)
    raw = V[:, :16]
    cos = raw @ raw.T / np.outer(np.linalg.norm(raw, axis=1), np.linalg.norm(raw, axis=1))
    ok = T.shape == (5, 16) and np.allclose(norms, 1.0) and np.allclose(T @ T.T, cos) and np.allclose(V, unit(V))
    return ok, f"shape {T.shape}, row norms {np.round(norms, 3).tolist()}, input left unchanged"


def t_maxsim():
    Q = unit(np.array([[1.0, 0.0], [0.0, 1.0]]))
    D = unit(np.array([[1.0, 0.0], [1.0, 1.0], [-1.0, 0.0]]))
    s = maxsim(Q, D)
    one, two = unit(np.array([[1.0, 0.0]])), unit(np.array([[1.0, 0.0], [1.0, 0.0]]))
    ok = abs(s - (1.0 + 2 ** -0.5)) < 1e-9 and abs(maxsim(one, two) - 1.0) < 1e-9 and abs(maxsim(two, one) - 2.0) < 1e-9
    return ok, f"score {s:.4f} (expected 1 + 0.7071); not symmetric: MaxSim(Q, D) sums over QUERY tokens"


def t_maxsim_identifier():
    # Toy token vectors. "E4012" is a look-alike of "E4021" (cosine 0.8), like a dense model would encode it.
    e = np.eye(7)
    tok = {"error": e[0], "E4021": e[1], "E4012": 0.8 * e[1] + 0.6 * e[2],
           "gateway": e[3], "timeout": e[4], "retry": e[5], "seconds": e[6]}
    emb = lambda words: unit(np.array([tok[w] for w in words]))
    q = emb(["error", "E4021"])
    doc_a = emb(["error", "E4021", "gateway", "timeout", "retry", "seconds"])   # the right doc, but long
    doc_b = emb(["error", "E4012"])                                               # the wrong doc, short look-alike
    a, b = maxsim(q, doc_a), maxsim(q, doc_b)
    pooled = lambda M: unit(M.mean(0))
    mean_a, mean_b = float(pooled(q) @ pooled(doc_a)), float(pooled(q) @ pooled(doc_b))
    ok = a > b and mean_b > mean_a
    return ok, (f"MaxSim picks the RIGHT doc: E4021 {a:.2f} vs E4012 {b:.2f};  "
                f"one averaged vector picks the WRONG one: {mean_a:.2f} vs {mean_b:.2f}")


def t_rrf():
    fused = rrf([["x", "y", "z"], ["y", "w"]])
    ok = fused[0] == "y" and set(fused) == {"x", "y", "z", "w"} and fused.index("x") < fused.index("w")
    tie = rrf([["b"], ["a"]])
    return ok and tie == ["a", "b"], f"fused order {fused} (y is #2 in A and #1 in B, beats x which is #1 in A only)"


def t_metrics():
    ranked = [["d1", "d2", "d3", "d4"], ["d9", "d8", "d5", "d7"], ["d6", "d1", "d2", "d3"]]
    relevant = [{"d2", "d4"}, {"d5"}, {"d0"}]
    r, m = recall_and_mrr(ranked, relevant, k=3)
    ok = abs(r - (0.5 + 1.0 + 0.0) / 3) < 1e-9 and abs(m - (1 / 2 + 1 / 3 + 0) / 3) < 1e-9
    return ok, f"recall@3 = {r:.4f} (expected 0.5000), MRR = {m:.4f} (expected 0.2778)"


if __name__ == "__main__":
    tests = [
        ("truncate_renorm: first dims, unit length again", t_truncate),
        ("maxsim: sum of each query token's best match", t_maxsim),
        ("maxsim keeps an exact identifier match that one averaged vector blurs", t_maxsim_identifier),
        ("rrf: consistent agreement beats one strong vote", t_rrf),
        ("recall@k and MRR on 3 queries", t_metrics),
    ]
    passed = sum(check(name, fn) for name, fn in tests)
    print(f"\n  {passed}/{len(tests)} checks passed")
