# Lesson 11/04 "Use it": semantic search over EVERY lesson in this repo, with your index code and a real embedding model.
# Run: python labs/11-04-embeddings/use_it.py     (first run embeds ~6K chunks with bge-small, ~1-3 min; then cached)
# Queries are each lesson's one-line hook ("> ..."), which is removed from the indexed text, so a hit means the
# model matched MEANING, not a copied sentence. A query "hits" if a chunk from its own lesson comes back.
import glob
import os
import sys
import time

import numpy as np
from sentence_transformers import SentenceTransformer, util
from sentence_transformers.util.quantization import quantize_embeddings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from my_vector_index import IVFIndex, exact_topk, hamming_topk, normalize, recall_at_k, rescore, to_bits  # your code

# Corpus = the lesson docs of the curriculum repo. Point CURRICULUM_DIR at a clone of it.
ROOT = os.environ.get("CURRICULUM_DIR", os.path.abspath(os.path.join(HERE, "..", "..")))
CACHE = os.path.join(HERE, "cache_bge_small.npz")
MODEL = "BAAI/bge-small-en-v1.5"
# bge's model card: queries get this instruction prefix, documents get none. (05/22: follow the model card.)
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def load_corpus(words=200, overlap=40):
    chunks, owner, queries = [], [], []
    for lesson_id, path in enumerate(sorted(glob.glob(os.path.join(ROOT, "phases", "*", "*", "docs", "en.md")))):
        lines = open(path, encoding="utf-8").read().splitlines()
        hook = next((l[2:].strip() for l in lines[:6] if l.startswith("> ")), None)
        body = " ".join(l for l in lines if not l.startswith("# ") and not l.startswith("> ")).split()
        if not hook or len(body) < 50:
            continue
        for start in range(0, len(body), words - overlap):
            chunks.append(" ".join(body[start:start + words]))
            owner.append(lesson_id)
        queries.append((lesson_id, hook))
    return chunks, np.array(owner), queries


def lesson_hits(ids, owner, q_owner, k):
    # fraction of queries whose top-k chunk ids include a chunk from the query's own lesson; plus MRR at lesson level
    hit, rr = 0, 0.0
    for row, want in zip(ids, q_owner):
        row = row[:k]
        lessons = owner[row[row >= 0]]  # -1 = padding (an IVF probe that found fewer than k docs)
        if want in lessons:
            hit += 1
            rr += 1.0 / (int(np.argmax(lessons == want)) + 1)
    return hit / len(ids), rr / len(ids)


if __name__ == "__main__":
    chunks, owner, queries = load_corpus()
    q_owner = np.array([o for o, _ in queries])
    model = SentenceTransformer(MODEL)
    if os.path.exists(CACHE):
        D = np.load(CACHE)["D"]
    else:
        t0 = time.perf_counter()
        D = model.encode(chunks, batch_size=64, normalize_embeddings=True, show_progress_bar=True).astype(np.float32)
        np.savez(CACHE, D=D)
        print(f"embedded {len(chunks)} chunks in {time.perf_counter() - t0:.0f}s")
    Q = model.encode([QUERY_PREFIX + h for _, h in queries], normalize_embeddings=True).astype(np.float32)
    print(f"{len(chunks)} chunks from {len(set(owner.tolist()))} lessons, {len(Q)} queries, dim {D.shape[1]}\n")

    print("A. your code vs the library")
    lib_bits = quantize_embeddings(D[:200], precision="ubinary")
    print(f"   to_bits == sentence-transformers ubinary:  {np.array_equal(to_bits(D[:200]), lib_bits)}")
    lib = util.semantic_search(Q[:50], D, top_k=10)
    mine, _ = exact_topk(Q[:50], D, 10)
    same = all([h["corpus_id"] for h in lib[i]] == mine[i].tolist() for i in range(50))
    print(f"   exact_topk ids == util.semantic_search:    {same}")

    print("\nB. retrieval quality (exact search)")
    exact10, _ = exact_topk(Q, D, 10)
    t0 = time.perf_counter()
    for q in Q:  # one query at a time, to compare fairly with IVF's per-query search below
        exact_topk(q[None], D, 10)
    t_exact = (time.perf_counter() - t0) / len(Q) * 1000
    for k in (1, 5, 10):
        h, mrr = lesson_hits(exact10, owner, q_owner, k)
        print(f"   hit@{k:<2} {h:5.1%}   " + (f"MRR@10 {mrr:.3f}" if k == 10 else ""))
    worst = [i for i, row in enumerate(exact10) if q_owner[i] not in owner[row]][:3]
    for i in worst:
        print(f"   miss: {queries[i][1][:80]!r}")

    print("\nC. binary quantization")
    Db, Qb = to_bits(D), to_bits(Q)
    print(f"   memory: float32 {D.nbytes / 1e6:.1f} MB  ->  bits {Db.nbytes / 1e6:.2f} MB  ({D.nbytes // Db.nbytes}x)")
    bin10 = hamming_topk(Qb, Db, 10)
    print(f"   binary top-10:            recall@10 vs exact {recall_at_k(bin10, exact10):.2f}   hit@10 {lesson_hits(bin10, owner, q_owner, 10)[0]:.1%}")
    for c in (20, 50, 100):
        cand = hamming_topk(Qb, Db, c)
        res = rescore(Q, D, cand, 10)
        print(f"   binary top-{c:<3} + rescore: recall@10 vs exact {recall_at_k(res, exact10):.2f}   hit@10 {lesson_hits(res, owner, q_owner, 10)[0]:.1%}")

    print("\nD. IVF index")
    n_lists = int(np.sqrt(len(D)))
    t0 = time.perf_counter(); idx = IVFIndex(D, n_lists); t_build = time.perf_counter() - t0
    print(f"   {n_lists} clusters, built in {t_build:.1f}s; exact search {t_exact:.2f} ms/query")
    for nprobe in (1, 4, 16):
        t0 = time.perf_counter()
        got = [idx.search(q, 10, nprobe) for q in Q]
        ms = (time.perf_counter() - t0) / len(Q) * 1000
        ids = np.array([np.pad(g[0], (0, 10 - len(g[0])), constant_values=-1) for g in got])
        scanned = np.mean([g[1] for g in got]) / len(D)
        print(f"   nprobe {nprobe:>2}: scans {scanned:5.1%} of vectors   recall@10 vs exact {recall_at_k(ids, exact10):.2f}   "
              f"hit@10 {lesson_hits(ids, owner, q_owner, 10)[0]:.1%}   {ms:.2f} ms/query")
