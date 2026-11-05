# 05/22 Embedding Models Deep Dive

> When RAG answers wrong, the usual cause is retrieval fetching the wrong chunk. The embedding model decides what "close" means.

**Done:** 2026-10-24 · Quiz 3/3 · Bridge lesson 10 of 14
**Lesson:** [phases/05-nlp-foundations-to-advanced/22-embedding-models-deep-dive](../phases/05-nlp-foundations-to-advanced/22-embedding-models-deep-dive/)
**My code:**
- [`labs/05-22-embedding-models/my_retrieval.py`](../labs/05-22-embedding-models/my_retrieval.py): Matryoshka truncation, MaxSim, RRF, recall@k / MRR. 5/5 checks.
- [`use_it.py`](../labs/05-22-embedding-models/use_it.py): real BGE-small vs BM25 vs hybrid, scored with my own metrics.

**Run it:** `python labs/05-22-embedding-models/my_retrieval.py`, then `python labs/05-22-embedding-models/use_it.py`. (`py` is my alias for `/usr/local/bin/python3`, the Python that has numpy, torch and sentence-transformers.)

---

## Remember these 3 things

1. **There are three kinds of embedding:**
   - **dense:** 1 vector per passage. Good at meaning and paraphrase.
   - **sparse:** a weight per word (BM25, SPLADE). Good at exact codes and names.
   - **multi-vector:** 1 vector per token, scored with MaxSim (ColBERT). Fine-grained matching, but a big index.

   Fuse them with **RRF**, which uses ranks, not raw scores.
2. **Follow the model card.** BGE's query prefix was worth **+10 points of recall@1** in my lab. After Matryoshka truncation, **re-normalise**. Only truncate models that were *trained* for it.
3. **Measure on your own data. Defaults are hypotheses.** In my lab, "hybrid is the production default" **lost** to dense alone, because BM25 only added mistakes. Pick the metric that matches what goes into the prompt: **MRR for top-1, recall@k for top-k.**

---

## In plain English

An embedding model reads a passage and outputs a list of numbers, a point in space, so that passages with similar meaning land near each other. Retrieval embeds the query and returns the nearest chunks by cosine similarity (01/14). The model was trained with InfoNCE (03/05).

**The catch:** squeezing a whole passage into one vector is lossy compression. Meaning survives well. **Exact identifiers don't.** "E4021" vs "E4012" is a tiny difference inside a long passage.

**Analogy:**
- **Dense** is a librarian who understands what you *mean* but may hand you the wrong edition.
- **BM25** is the card catalogue: exact titles, but no idea that "money back" means "refund".
- **Hybrid** asks both. That helps only when the catalogue knows something the librarian doesn't.

---

## The three modes

| Mode | Representation | Wins at | Loses at | Index cost |
|------|---------------|---------|----------|------------|
| **Dense** (bi-encoder: BGE, OpenAI text-3) | 1 vector per passage | Paraphrase, synonyms, cross-lingual | Exact codes, SKUs, rare names | Small |
| **Sparse** (BM25, SPLADE) | Weight per vocabulary term, mostly 0 | Exact terms | "money back" vs "refund" (no shared words) | Small |
| **Multi-vector** (ColBERTv2) | 1 vector per token | Fine-grained and long queries, identifiers | Storage and speed | × tokens per doc |

**BGE-M3** outputs all three from a single model.

### MaxSim (multi-vector scoring)

For **each query token**, find its best-matching document token, then add up those best matches. It isn't symmetric, because the sum runs over the *query's* tokens.

```python
def maxsim(Q, D):                      # Q: (q_tokens, d), D: (d_tokens, d), unit rows
    return (Q @ D.T).max(axis=1).sum() # axis=1: best doc token for each query token
```

**My check showed why it matters.** Query "error E4021":
- the right doc is long: "error E4021 gateway timeout retry seconds"
- the wrong doc is a short look-alike: "error E4012"

| Method | Right doc | Wrong doc | Picks |
|--------|-----------|-----------|-------|
| One averaged vector | 0.58 | **0.90** | ✘ wrong (the long doc's match is diluted) |
| MaxSim | **2.00** | 1.80 | ✔ right (E4021 matches E4021 directly) |

### RRF (Reciprocal Rank Fusion)

```text
RRF(doc) = Σ over retrievers  1 / (k + rank)      k = 60, rank starts at 1, missing = 0
```

- **Ranks, not scores**, because cosine runs 0 to 1 and BM25 runs 0 to 30+. Adding the raw scores lets BM25 dominate.
- **Agreement beats a single strong vote:** #2 in both lists (2/62 = 0.032) beats #1 in only one list (1/61 = 0.016).

---

## Dimensions cost money: Matryoshka

**Storage:** 10M chunks × 1,536 dimensions × 4 bytes = **61 GB**. Truncate to 256 dimensions and it's **10 GB**, and with int8 on top, **2.5 GB**. Vector indexes (HNSW) usually sit in **RAM**, so this decides your hardware.

**Matryoshka-trained** models (OpenAI text-3, Nomic v1.5, Voyage, …) make the first N dimensions a usable embedding on their own.

```python
def truncate_renorm(V, dim):
    T = V[:, :dim]
    return T / np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-12)   # re-normalise, so dot = cosine again
```

**My lab: BGE-small is NOT Matryoshka-trained.** Truncating it hurt:

| dims | recall@1 | MRR | bytes per vector |
|------|----------|-----|------------------|
| 384 | 0.95 | 1.000 | 1536 |
| 128 | 0.75 | 0.864 | 512 |
| 64 | 0.85 | 0.920 | 256 |
| 32 | 0.65 | 0.798 | 128 |

**64 beat 128? That's noise.** With 10 queries, one query changing rank moves recall@1 by 0.1. A 10-query eval can't separate small differences. Real decisions need **hundreds** of labelled queries.

---

## My lab results (14 support docs, 10 queries, BGE-small)

| Retriever | recall@1 | recall@3 | MRR |
|-----------|----------|----------|-----|
| BM25 | 0.70 | 0.85 | 0.820 |
| Dense, no prefix | 0.85 | 1.00 | 0.950 |
| **Dense + BGE prefix** | **0.95** | 1.00 | **1.000** |
| Hybrid RRF (dense + BM25) | 0.75 | 0.95 | 0.900 |

- **0.95 recall@1 is the maximum here.** One query has 2 relevant docs, so with k = 1 it can reach at most 0.5.
- **Prefix:** +10 points. A silent bug when you forget it.
- **Why hybrid lost:** on "how long until I get my money back", BM25 ranked the refund doc **5th** (no shared words), so RRF pulled dense's #1 down to #2. "Gift card run out" vs "never expire" went the same way. Dense already handled E4021, E1007 and the SKU, because the corpus is small with few look-alikes. **BM25 added only mistakes.**
- **When hybrid pays off:** large corpora with many near-identical identifiers, rare jargon or unseen names, where sparse catches what dense **misses**. Otherwise, weight the retrievers or route by query type. Always verify.

---

## Eval metrics (my code)

```python
def recall_and_mrr(ranked, relevant, k):
    # recall@k: share of each query's relevant docs found in its top k, averaged
    # MRR: 1 / rank of the FIRST relevant doc (0 if none appears), averaged
```

**Which metric matches what goes into the prompt:**

| Prompt gets | Optimise | Why |
|-------------|----------|-----|
| Top-1 chunk | **MRR** | Only whether the right chunk is *first* matters. At #2 the LLM never sees it. |
| Top-5 chunks, and answers need 2 of them | **recall@5** | Order within the top 5 matters little. Getting *both* in matters. |

**Review point:** a query with an empty `relevant` set counted as 0 recall but stayed in the average, which drags it down. Skip unlabelled queries or raise an error, because it's a data bug, not a score.

---

## Pitfalls

- **Missing query prefix** (BGE, E5 and others): read the model card.
- **Asymmetric models:** some encode queries and docs differently, so use the right method for each.
- **Truncating a non-Matryoshka model**, or truncating too far (1,536 → 64) without validating.
- **Silent input truncation:** inputs past the model's maximum length are cut off. Chunk long documents (next lesson).
- **Trusting MTEB rank alone:** it's an average over many tasks. Your domain can reverse the ranking, and latency isn't in the score.

## Where this shows up in AI engineering

- **Picking a model for RAG:** shortlist from MTEB, then evaluate on *your* labelled queries with recall@k / MRR / nDCG, and check latency and cost.
- **Infrastructure cost:** dimensions × bytes × number of chunks is RAM. Matryoshka and int8 cut it 4 to 24×.
- **Hybrid search:** most vector databases (Qdrant, Weaviate, Elasticsearch, pgvector plus full-text) support dense + BM25 with RRF. Turn it on *because your eval says so*.
- **Interview:** "Our RAG retrieves the wrong doc for error-code queries. What do you do?" Check the prefix and tokenization, add sparse or hybrid, consider a reranker or ColBERT, and measure the change.

## Words

- **Bi-encoder:** query and doc are encoded separately, compared with a dot product. Fast.
- **Cross-encoder / reranker:** reads the query and doc *together*. More accurate, slower, used on the top 50.
- **MTEB:** the Massive Text Embedding Benchmark leaderboard.
- **nDCG:** a ranking metric that rewards relevant docs placed higher, with graded relevance.

## Connects to

- **Before:** [05/19 Tokenization](05-19-subword-tokenization.md) (how "E4021" gets split), [03/05 Loss Functions](03-05-loss-functions.md) (InfoNCE trained these models), [01/14 Distances](01-14-norms-and-distances.md) (cosine, normalisation), [02/09 Model Evaluation](02-09-model-evaluation.md) (precision and recall@k).
- **Next:** 05/23 Chunking Strategies for RAG. What exactly gets embedded.

---

## Self-test

1. Name the three embedding modes and what each one is good at.
2. Why might a dense model rank "E4012 coupon invalid" above "E4021 gateway timeout" for the query "E4021 at checkout"?
3. Write MaxSim in one line of numpy. Which axis gets the max?
4. Why does RRF fuse ranks instead of raw scores? Which wins: #1 in one list, or #2 in both?
5. 10M chunks × 1,536 float32 dimensions: how many GB? At 256 dimensions?
6. In my lab, why did hybrid lose to dense, and when does hybrid help?
7. Only the top-1 chunk goes in the prompt: which metric do you optimise?

<details>
<summary>Answers</summary>

1. Dense: meaning and paraphrase. Sparse: exact terms like codes and names. Multi-vector: fine-grained token matching.
2. The identifier is a few tokens in a passage that's otherwise nearly identical ("Troubleshooting … checkout"). Compressed into one vector, the difference is a whisper.
3. `(Q @ D.T).max(axis=1).sum()`. Axis 1: the best doc token for each query token.
4. The scales differ (cosine 0 to 1, BM25 0 to 30+). #2 in both: 2/62 ≈ 0.032 beats 1/61 ≈ 0.016.
5. 61 GB, and about 10 GB (2.5 GB with int8 as well).
6. BM25 ranked the paraphrase queries low ("money back" vs "Refunds"), dragging fused ranks down, and dense already got the codes. Hybrid helps when sparse catches what dense misses: many look-alike identifiers, rare jargon, unseen names.
7. MRR.

</details>
