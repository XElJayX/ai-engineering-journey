# 11/07 Advanced RAG: BM25, Reciprocal Rank Fusion, hybrid search, cross-encoder rerank

> Vector search blurs exact tokens (case numbers, model codes). BM25 is blind to paraphrase. RRF-merging the two **lowered** hit@1 from 39/48 (vector) to 32/48, because RRF rewards agreement and one retriever's noise outvoted the other's #1. A cross-encoder rerank of the top-50 fused results took it to **44/48** for 58 ms/query. Hybrid search is the wide net; the reranker is the precision step.

**Done:** 2026-11-05 · Quiz 3/3 · Route row 7 of 109
**Lesson:** [phases/11-llm-engineering/07-advanced-rag](../phases/11-llm-engineering/07-advanced-rag/)
**My code:**
- [`labs/11-07-advanced-rag/my_hybrid.py`](../labs/11-07-advanced-rag/my_hybrid.py): `tokenize`, `BM25.fit` / `scores` (from the formula), `rrf`, `hybrid` (with my 11/04 `exact_topk`), `rerank` (one batched cross-encoder call). 18/18 checks.
- [`experiment.py`](../labs/11-07-advanced-rag/experiment.py): 192 fictional 10-K chunks (with headers), 48 queries (code / paraphrase / metric), vector vs BM25 vs hybrid vs hybrid+rerank.

**Run it:** `python labs/11-07-advanced-rag/my_hybrid.py`, then `python labs/11-07-advanced-rag/experiment.py` (~1 min, no LLM)

---

## Remember these 3 things

1. **The two retrievers are blind in opposite places.**
   - BM25 = exact words, with rare words weighted heavily (IDF). Codes: **16/16** vs vector 8/16.
   - Vector = meaning. Paraphrase: **16/16** vs BM25 4/16.
   - Neither can find a token that **isn't in the chunk**. A missing company name is fixed by headers or metadata filters, not by BM25.
2. **RRF rewards agreement, and that can hurt.**
   - `score = Σ 1/(60 + rank)`. #1 in one list only = 0.0164; ~#10 in both = 0.0286, which wins.
   - On paraphrase, BM25's junk list dragged hybrid down from 16/16 to **7/16** (hit@1) and to **10/16** (hit@5).
   - Use hybrid as a **recall** stage (a wide top-50), then rerank. Or weight it (`alpha`) or route by query type.
3. **Rerank = a cross-encoder on a shortlist.**
   - The query and chunk are read **together**, so it can tell which fiscal year a paragraph is about (metric questions 16/16).
   - It can't precompute anything: **58 ms** for 50 pairs (MiniLM, 22M) vs 0.2 ms for retrieval. So the pipeline is retrieve 50 → rerank → keep 5.
   - Check the "misses" against your gold labels: the reranker's top hit was the right company in a different paragraph.

---

## In plain English

Two librarians. One (**BM25**) only matches exact words: ask for "case 49810" and he finds it instantly, but ask "who depends on one overseas factory?" and he looks for the word "factory". The other (**vector**) understands what you mean but is fuzzy on specifics: every case number looks the same to her.

**RRF** is a vote: books that both librarians put fairly high on their lists win. That's great when both are half-right. It's bad when one librarian is clueless on this question, because his random picks still get votes.

The **reranker** is a careful expert who reads your question next to each of the 50 shortlisted books and scores them one by one. Too slow for the whole library, perfect for a shortlist.

---

## Worked example

**BM25 by hand** (k1 = 1.2, b = 0.75; 4 docs, avgdl = 6.25; "cat" in 2 of 4 docs → idf = ln(2.5/2.5 + 1) = **0.693**):
- "cat" ×1 in a 6-word doc: 0.693 × 1 × 2.2 / (1 + 1.2 × (0.25 + 0.75 × 6/6.25)) = 1.525 / 2.164 = **0.705**
- "cat" ×3 in a 4-word doc: 0.693 × 3 × 2.2 / (3 + 1.2 × (0.25 + 0.75 × 4/6.25)) = 4.575 / 3.876 = **1.180**
- Ratio **1.67×**, not 3×: saturation, even with the shorter doc's boost.

**RRF by hand** (k = 60): X is #1 in vector and absent from BM25 → 1/61 = **0.0164**. Y is #4 in both → 2/64 = **0.0313**. Y wins.

**My experiment** (192 chunks, BGE-small, my BM25, ms-marco-MiniLM-L-6-v2 on the top-50 fused results):

| Method | code @1 / @5 | paraphrase @1 / @5 | metric @1 / @5 | all @1 | ms/query |
|---|---|---|---|---|---|
| vector | 8 / 11 | **16 / 16** | 15 / 16 | 39/48 | ~0 |
| BM25 | **16 / 16** | 4 / 6 | 12 / 16 | 32/48 | 0.2 |
| hybrid (RRF) | 13 / 16 | 7 / 10 | 12 / 16 | 32/48 | 0.2 |
| hybrid + rerank | **16 / 16** | 12 / 15 | **16 / 16** | **44/48** | 58 |

My predictions: vector code 8/16 ✓, BM25 paraphrase 4/16 ✓, hybrid lower than vector ✓, rerank 44/48 ✓, 50 ms (real 58). (They matched to the digit; next time, commit before running.)

---

## Formulas / cheat sheet

- **BM25:** `Σ_t idf(t) · tf·(k1+1) / (tf + k1·(1 − b + b·dl/avgdl))`, over unique query terms. k1 = 1.2 (saturation), b = 0.75 (length normalization).
- **IDF (Lucene):** `ln((N − df + 0.5)/(df + 0.5) + 1)`. Rare → big; in every doc → ~0; never negative.
- **RRF:** `Σ_lists 1/(k + rank)`, rank from 1, k = 60. Uses ranks, so the scale doesn't matter.
- **Weighted hybrid (Weaviate etc.):** `alpha` 0 = pure BM25, 1 = pure vector; 0.3-0.7 is typical. Tune it on your golden set.
- **Pipeline:** filter by metadata → (rewrite / HyDE) → vector top-50 + BM25 top-50 → RRF → cross-encoder → top-5 → (parent expansion) → assemble.

---

## My code (what I wrote and why)

```python
tokens = set(tokenize(query))                 # a repeated query word counts once
for token in tokens:
    term_idf = self.idf(token)
    for d in range(self.N):                   # ← O(N) per term; see fix below
        tf = self.tf[d].get(token, 0)
        if tf == 0:
            continue
        length_norm = 1 - self.b + self.b * (self.dl[d] / self.avgdl)
        scores[d] += term_idf * tf * (self.k1 + 1) / (tf + self.k1 * length_norm)
```

```python
for rank, doc_id in enumerate(ranked_list, start=1):   # rank starts at 1
    scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
return sorted(scores.items(), key=lambda item: (-item[1], item[0]))
```

```python
pairs = [(query, texts[doc_id]) for doc_id in ids]
scores = cross_score(pairs)                   # ONE batched call
```

**Improvement still to make (scale):** build an **inverted index** at fit time, `postings[term] = [(doc_id, tf), ...]`, so a query touches only the docs containing its terms, not all N. This is what Lucene/Elasticsearch do.

---

## Where this shows up in AI engineering

- **Project 1 (SEC RAG):** users type tickers, form items ("Item 1A"), CUSIPs and exact phrases, so you need BM25 alongside vectors. Add a cross-encoder rerank, plus metadata filters for company and fiscal year (cheaper and more reliable than hoping the ranking handles it).
- **Interviews:** "Hybrid vs vector?", "Why rerank, why not use a cross-encoder for everything?", "How does BM25 work?", "How does keyword search scale?" (inverted index).
- **Production latency budget:** retrieval ~ms, rerank tens to hundreds of ms, LLM seconds. Rerank 50, not 500.

---

## Mistakes to avoid

- **Assuming hybrid is always ≥ either retriever.** Mine was below vector (32 vs 39). Measure per query type.
- **Adding raw scores** (`cosine + bm25`): different, unbounded, query-dependent scales. Use ranks (RRF) or normalize carefully.
- **`text.split()` tokenizing:** `"halcyon's"` and `"(fy2024)"` never match the query words `halcyon` / `fy2024`.
- **Counting BM25 zero-score docs as hits:** they pad the list with noise. A keyword hit must contain a term.
- **Reranking one pair at a time:** you lose the batching that makes the shortlist affordable.
- **Expecting BM25 to fix missing entities:** it can only match tokens that are present.
- **Calling a reranker miss wrong without checking the labels:** its top hit may be a valid chunk you didn't label.

---

## Words

- **BM25:** a keyword ranking function using tf saturation, IDF and length normalization.
- **IDF:** inverse document frequency; how rare (so informative) a term is.
- **Inverted index:** term → list of docs (and counts) containing it; how keyword search stays fast.
- **RRF:** Reciprocal Rank Fusion; merges ranked lists by Σ 1/(k + rank).
- **Bi-encoder:** embeds the query and doc separately (fast, precomputable).
- **Cross-encoder:** scores (query, doc) jointly (accurate, slow, shortlist only).
- **HyDE:** embed an LLM-written hypothetical answer instead of the question.
- **Parent-child chunking:** retrieve small chunks, send their bigger parent.
- **Metadata filtering:** restrict by company/date/type before searching.
- **hit@k:** a gold chunk appears in the top k.

## Connects to

- [11-06 RAG](11-06-rag.md): headers fixed missing entities; today fixes invisible exact tokens and fine distinctions.
- [11-04 Embeddings](11-04-embeddings.md): `exact_topk` is the vector side; vague-query misses → HyDE / rewrite.
- [05-19 Subword Tokenization](05-19-subword-tokenization.md): BM25 tokenization is word-level and your choice; it decides what can match.
- [01-09 Information Theory](01-09-information-theory.md): IDF is basically "surprise" (−log probability of a term appearing).
- Next: 11/08 Fine-Tuning with LoRA & QLoRA (RAG for knowledge, fine-tuning for behaviour).

---

## Self-test

1. **Compute RRF (k = 60): A is #2 in vector and #3 in BM25; B is #1 in vector only. Which wins?**
   <details><summary>Answer</summary>A = 1/62 + 1/63 = 0.0161 + 0.0159 = 0.0320. B = 1/61 = 0.0164. A wins: agreement beats one top vote.</details>
2. **Why did hybrid score below pure vector on paraphrase queries?**
   <details><summary>Answer</summary>BM25 returned junk on paraphrase (4/16). Under RRF, chunks that appear mid-list in both lists outscore the vector #1, which BM25 never saw. RRF averages in the bad retriever's noise.</details>
3. **Why does BM25 beat embeddings on "case No. 1:23-cv-49810"?**
   <details><summary>Answer</summary>The token `49810` appears in 2 of 192 chunks, so it has a huge IDF and an exact match. To an embedding, every case number is "some case number", so the right paragraph type comes from the wrong company.</details>
4. **Why not run the cross-encoder over all 2M chunks?**
   <details><summary>Answer</summary>Nothing can be precomputed: every (query, chunk) pair needs a forward pass. 50 pairs took 58 ms, so 2M would take ~40 minutes per query. Retrieve 50 cheaply, then rerank.</details>
5. **3× a term in a short doc vs 1× in an average doc: why is the score ratio ~1.67, not 3?**
   <details><summary>Answer</summary>tf saturation: tf·(k1+1)/(tf + k1·norm) approaches k1+1 as tf grows. More occurrences give diminishing returns.</details>
6. **Would BM25 have fixed 11/06's plain-chunk R&D/capex failures?**
   <details><summary>Answer</summary>No. Those chunks didn't contain the company name, so BM25 matched only generic words shared by 16+ chunks. Fix it with headers or a metadata filter.</details>
7. **When is HyDE a bad idea?**
   <details><summary>Answer</summary>For exact-token queries (codes, IDs, numbers): the fake answer invents the identifiers. It also adds an LLM call (0.5-2 s). Use it for vague, natural-language questions.</details>
8. **How does a keyword index answer a query without scanning every doc?**
   <details><summary>Answer</summary>An inverted index: for each query term, read its postings list (docs and tf) and score only those docs.</details>
