# 11/04 Embeddings & Vector Representations: from a numpy array to a vector database

> Brute force costs N × d memory and N × d multiply-adds per query. There are two levers: compress each vector (binary quantization, 32×) and avoid scanning every vector (IVF, HNSW). Both are approximate, so measure recall against exact search, and measure the task metric too.

**Done:** 2026-11-02 · Quiz 3/3 · Route row 4 of 109
**Lesson:** [phases/11-llm-engineering/04-embeddings](../phases/11-llm-engineering/04-embeddings/)
**My code:**
- [`labs/11-04-embeddings/my_vector_index.py`](../labs/11-04-embeddings/my_vector_index.py): `normalize`, `exact_topk` (one matmul + argpartition), `to_bits`, `hamming_topk`, `rescore`, spherical `kmeans`, `IVFIndex`, `recall_at_k`. 16/16 checks.
- [`use_it.py`](../labs/11-04-embeddings/use_it.py): BGE-small on all 523 lessons of this repo (5,800 chunks, 384-d). Each lesson's hook is the query.

**Run it:** `python labs/11-04-embeddings/my_vector_index.py`, then `python labs/11-04-embeddings/use_it.py` (the embeddings are cached in `cache_bge_small.npz`)

---

## Remember these 3 things

1. **Brute force has two costs that grow with N, and each has its own fix.**
   - Memory = N × d × 4 bytes. At 1M chunks × 1024-d that's **4.1 GB**. Fix: compress. Binary = 1 bit per dim = **128 MB** (32×). Matryoshka uses fewer dims; int8 gives 4×.
   - Compute = N × d multiply-adds per query: **~1 billion** at 1M × 1024. Fix: don't scan everything. IVF scans a few clusters; HNSW walks a graph.
2. **Approximate search loses results silently, so measure it offline.**
   - A chunk the index didn't return leaves no trace. The scores of the chunks you did get look normal, and cosine isn't calibrated anyway.
   - The check: run **exact search on a sample of queries** and compute recall@k of the index against it. Then also measure the **task** metric (golden set), because the two can disagree a lot. Mine: binary recall 0.57 vs exact, yet hit@10 was 90.1% vs 92.9%.
3. **Search cheap, then rescore exact. Use flat search until it's too slow.**
   - Binary top-100 + float32 rescore: recall **0.93** at 32× less memory in the first pass.
   - At 5,800 vectors, exact search (**0.07 ms**) beat IVF nprobe=16 (**0.21 ms**). One BLAS matmul beats per-query bookkeeping. Start measuring for an index at ~100k vectors.

---

## In plain English

Exact search is like checking **every book in a library** for each question. It's always right, but the work grows with the size of the library.

- **Binary quantization** is like replacing every book with a 1-page summary card of yes/no ticks. The cards fit in your pocket (32× smaller) and comparing ticks is very fast, but you lose detail. So you use the cards to shortlist 100 books, then read those 100 properly (**rescoring**).
- **IVF** is like sorting the library into 76 rooms by topic. For each question you go to the 1, 4 or 16 rooms whose signs best match it (**nprobe**). Books near a room boundary are the risk: the right book can be one room over. More rooms visited means better recall and more walking.
- **HNSW** (what pgvector, Qdrant and Weaviate use by default) is a road network instead of rooms: highways on top for long jumps, side streets below for fine steps. `ef_search` is how wide you look while walking.

Whatever shortcut you use, **you can't tell from the answer that a better book was skipped**. You find out only by checking some questions the slow, exact way.

---

## Worked example

**Back-of-envelope for Project 1** (1M chunks, 1024-d, float32):

| | Calculation | Result |
|---|---|---|
| RAM | 1,000,000 × 1024 × 4 B | 4.1 GB |
| Binary | 1024 bits / 8 = 128 B per vector × 1M | 128 MB (÷32) |
| One brute-force query | 1,000,000 × 1024 | ~1.0 × 10⁹ multiply-adds |

**Binary quantization by hand** (8-dim vectors):

```text
a = [ 0.5, -0.2,  0.8,  0.1, -0.6,  0.3, -0.4,  0.2]  -> bits 10110101  (packbits -> byte 181)
b = [ 0.4, -0.1,  0.7, -0.2, -0.5,  0.4,  0.1,  0.3]  -> bits 10100111
c = [-0.3,  0.6, -0.2,  0.5,  0.4, -0.1,  0.3, -0.5]  -> bits 01011010
```

- a vs b: cosine **0.865**; XOR = 00010010, so Hamming = **2** of 8 bits differ.
- a vs c: cosine **−0.617**; Hamming = **7** of 8.
- High cosine means few sign flips, so the ranking roughly survives. But dims 4 and 7 (values 0.1 and −0.2, 0.1 and −0.4) flipped on small values, which is exactly the detail 1 bit can't keep. That's why the rescore exists.

**My real-data results** (BGE-small, 5,800 chunks, 523 hook queries):

| Method | Scans | Recall@10 vs exact | hit@10 (right lesson) | ms/query |
|---|---|---|---|---|
| Exact | 100% | 1.00 | 92.9% (hit@1 74.6%, MRR 0.813) | 0.07 |
| Binary top-10 | 100% of bits | 0.57 | 90.1% | |
| Binary top-20 + rescore | | 0.73 | 92.0% | |
| Binary top-50 + rescore | | 0.86 | 93.1% | |
| Binary top-100 + rescore | | 0.93 | 92.7% | |
| IVF nprobe 1 (76 clusters) | 1.5% | 0.53 | 74.6% | 0.02 |
| IVF nprobe 4 | 5.8% | 0.78 | 89.5% | 0.05 |
| IVF nprobe 16 | 22.7% | 0.94 | 92.5% | 0.21 |

- 93.1% > 92.9% for the top-50 rescore is noise: 1 query out of 523 ≈ 0.2 percentage points.
- nprobe 16 scans 22.7%, not 16/76 = 21%, because the clusters are uneven in size.
- My predictions: hit@1 70% (real 74.6), binary recall 0.6 (0.57), nprobe=1 scans 2% (1.5) with recall 0.7 (**0.53**, too optimistic: boundary losses).

---

## Formulas / cheat sheet

- **cosine(q, d)** = q·d / (‖q‖ ‖d‖). On unit vectors this equals q·d, so normalize once and use a matmul: `S = Q @ D.T` → shape (m queries, n docs).
- **Top-k without a full sort:** `np.argpartition(-S, k-1, axis=1)[:, :k]` is O(n) but **unordered**, so sort just those k after. Use `k-1`: `kth=k` crashes when k == n.
- **Binary quantization:** bit = 1 if x > 0, else 0. `np.packbits` stores 8 bits per byte, so d dims → d/8 bytes.
- **Hamming(q, d)** = popcount(q XOR d), the number of bits that differ. Smaller = more similar. Sum the popcounts as int, because uint8 overflows at 255.
- **recall@k vs exact** = |approx top-k ∩ exact top-k| / k, averaged over queries. Order doesn't matter.
- **IVF:** k-means with K ≈ √N clusters (76 for 5,800). Each doc goes on the list of its nearest centroid. A query scans the lists of its `nprobe` nearest centroids. Cost ≈ K + nprobe × N/K dot products.
- **Spherical k-means:** assign by highest dot product; new centroid = mean of its members, **re-normalized**; an empty cluster keeps its old centroid.

---

## My code (what I wrote and why)

```python
scores = Q @ D.T                                              # all queries at once, one BLAS call
candidate_id = np.argpartition(-scores, k-1, axis=1)[:, :k]   # top k in O(n), unordered
candidate_scores = np.take_along_axis(scores, candidate_id, axis=1)
order = np.argsort(-candidate_scores, axis=1)                 # order only the k
candidate_id = np.take_along_axis(candidate_id, order, axis=1)
```

```python
xor = np.bitwise_xor(Qb[:, None, :], Db[None, :, :])   # (m, n, bytes) via broadcasting
distances = np.sum(POPCOUNT[xor], axis=2, dtype=int)    # popcount per byte, summed as int
ids = np.argsort(distances, axis=1, kind="stable")      # stable: ties go to the lower id
```

IVF search: rank the centroids by `self.centroids @ q`, take the top `nprobe` lists, `np.concatenate` the ids, score `self.D[ids] @ q` exactly, and argsort.

**Fixes still to make:**
- `hamming_topk` builds an (m, n, bytes) array: 146 MB here, but **12.8 GB** for 100 queries × 1M docs × 128 B. Loop over queries or doc chunks (FAISS `IndexBinaryFlat` uses the CPU popcount instruction).
- `hamming_topk` fully argsorts n. Use argpartition and then a stable sort of the k (ties need care).
- `kmeans` loops over clusters with a boolean mask (O(K·n)). `np.add.at(sums, assignments, X)` vectorizes it.
- `IVFIndex.search` argsorts all centroids. `argpartition` with nprobe would do.

---

## Where this shows up in AI engineering

- **Project 1 (SEC RAG):** start with an exact scan (pgvector with no index). It's your recall baseline. Add HNSW when p95 latency needs it, then tune `hnsw.ef_search` (or `ivfflat.probes`) by measuring recall against exact search on your golden queries.
- **Memory budgets:** with float32 vs binary vs Matryoshka-256, you can calculate RAM per million chunks in an interview.
- **Production pattern:** binary or int8 in RAM for the first pass, full vectors on disk for the rescore, then a cross-encoder for the final rerank (route lesson 22).
- **Debugging misses:** first check whether exact search also missed. If it did, it's not the index, so look at the query, the chunking or the model.

---

## Mistakes to avoid

- **Expecting high recall from nprobe=1.** I predicted 0.7; the real value was 0.53. True neighbours often sit in the next cluster over. Recall climbs steeply with the first few probes.
- **Treating recall@k as something you can see at query time.** You can't: it needs the exact top-k. The scores of the chunks you got won't warn you about the one you missed.
- **Blaming the model or index for a vague query.** "Your tools shape your thinking" missed even with **exact** search because the query has no topic words. Tag golden-set queries by type (specific, vague, multi-hop, numeric) and fix vague ones upstream (query rewriting/HyDE, or a clarifying question).
- **Adding an ANN index to a small corpus.** At 5,800 vectors, IVF was 3× slower than exact search.
- **Forgetting the second sort after argpartition.** Both the ids check and the sorted-scores check fail.

---

## Words

- **ANN (approximate nearest neighbour):** search that may miss some true neighbours in exchange for speed.
- **Flat / exact index:** brute force over every vector (FAISS `IndexFlatIP`, pgvector with no index).
- **IVF / ivfflat:** inverted file, meaning k-means clusters plus per-cluster id lists. Knob: `nprobe` (`ivfflat.probes`).
- **HNSW:** a layered proximity graph. Knobs: `M` (number of links), `ef_construction`, `ef_search`.
- **Binary quantization:** a sign bit per dimension, compared with Hamming distance, 32× smaller.
- **Rescoring:** recomputing exact scores on a shortlist from a cheap first pass.
- **Bi-encoder vs cross-encoder:** a bi-encoder embeds query and document separately (precomputable, used for retrieval). A cross-encoder reads both together (slow, accurate, used to rerank).

## Connects to

- [01-14 Norms and Distances](01-14-norms-and-distances.md): unit vectors make cosine = dot, which is why one matmul does the search.
- [05-22 Embedding Models](05-22-embedding-models.md): model cards, query prefixes, Matryoshka (the third compression lever).
- [05-23 Chunking](05-23-chunking.md): what each vector stands for.
- [02-09 Model Evaluation](02-09-model-evaluation.md): recall@k and the habit of measuring the trade-off.
- Next: 11/05 Context Engineering (what you do with the chunks once they're retrieved).

---

## Self-test

1. **1M chunks, 1536-d, float32. RAM for the vectors? With binary? With Matryoshka-256 float32?**
   <details><summary>Answer</summary>1M × 1536 × 4 = 6.1 GB. Binary: 1536/8 = 192 B per vector, so 192 MB. Matryoshka-256: 1M × 256 × 4 = 1.0 GB (6× less). Matryoshka-256 plus binary is 32 B per vector, so 32 MB.</details>
2. **Your index returns 9 of the true top-10, and the missing one holds the answer. Why don't the scores warn you, and how do you catch it?**
   <details><summary>Answer</summary>You only see the scores of the chunks you got. They look normal, and cosine isn't calibrated. The missing one leaves no trace. Catch it offline: compare the index's top-k with exact search on sample queries (recall@k), and track the golden-set task metric.</details>
3. **Binary top-10 had recall 0.57 vs exact but hit@10 90.1% vs 92.9%. How can both be true, and would that hold for SEC filings?**
   <details><summary>Answer</summary>Each lesson has many chunks with near-tied scores, so swapping in a sibling chunk still finds the right lesson. In SEC filings the answer is usually one specific chunk (one table, one number), so lost recall becomes wrong answers. Rescore, and measure on your own golden set.</details>
4. **Why was exact search (0.07 ms) faster than IVF nprobe=16 (0.21 ms) here?**
   <details><summary>Answer</summary>5,800 × 384 is about 2.2M multiply-adds, one contiguous BLAS call. IVF adds per-query overhead (ranking centroids, concatenating lists, gathering scattered rows) that costs more than the work it saves at this size. Start measuring for an ANN index at around 100k vectors.</details>
5. **nprobe=1 scans 1.5% of the vectors. Why is recall only 0.53?**
   <details><summary>Answer</summary>Cluster boundaries. Many true neighbours fall into a neighbouring cluster that isn't probed. With nprobe = 4 recall was 0.78, and with 16 it was 0.94.</details>
6. **"Your tools shape your thinking" missed even with exact search. Index, model, or query? What's the fix?**
   <details><summary>Answer</summary>The query. It has no topic information, so it could match dozens of lessons. Put tagged vague queries in the golden set and handle them upstream (query rewriting/HyDE, or ask a clarifying question). Don't tune the index for them.</details>
7. **After `np.argpartition(-S, k-1, axis=1)[:, :k]`, what still needs doing, and why `k-1`?**
   <details><summary>Answer</summary>The k ids are the top k but unordered, so sort the k by score (take_along_axis twice). `kth` is the 0-based position that ends up in its sorted place, so k-1 is the k-th best. kth=k is out of range when k == n.</details>
