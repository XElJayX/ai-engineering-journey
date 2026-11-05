# 01/14 Norms and Distances

> "Similar" is not a fact about the data. It is a choice of distance function.

**Done:** 2026-10-08 · Quiz 3/3 · Bridge lesson 1 of 14
**Lesson:** [phases/01-math-foundations/14-norms-and-distances](../phases/01-math-foundations/14-norms-and-distances/)
**My code:** none (this was before I started writing code each lesson)

---

## Remember these 3 things

1. **Cosine similarity compares direction and ignores length.** Two texts about the same topic point the same way, even if one is much longer.
2. **For unit-length vectors, cosine, dot product and L2 give the same ranking.** Most embedding models output unit vectors, so in RAG the choice often doesn't matter. When the vectors aren't normalised, it matters a lot.
3. **Similarity: sort descending. Distance: sort ascending.** Getting this backwards returns the *least* relevant chunks. This is the bug I caught in the lesson.

---

## In plain English

- A **vector** is a list of numbers, like `[3, 4]`. An embedding is a vector of 768 to 3072 numbers that represents the meaning of a piece of text.
- A **norm** answers "how big is this vector?" It is a length.
- A **distance** answers "how far apart are two vectors?" It is the norm of their difference: `distance(a, b) = norm(a - b)`.
- A **similarity** answers "how alike are they?" Bigger means more alike. It is the opposite of a distance.

**Analogy.** Two people walking from the same spot:
- **Cosine** asks "are they walking in the same direction?" It doesn't care how far each one walked.
- **L2 (Euclidean)** asks "how far apart are they standing now?" Both direction and distance walked count.
- **Dot product** asks "same direction, *times* how far each walked". Long walks in a roughly similar direction score high.

---

## The different ways to measure size (norms)

Worked example with `x = (3, 4)`:

| Norm | How to compute | For (3, 4) | Picture |
|------|----------------|-----------|---------|
| **L1** (Manhattan) | add up the absolute values | 3 + 4 = **7** | Walking city blocks: 3 east, then 4 north |
| **L2** (Euclidean) | square, add, square root | sqrt(9 + 16) = **5** | The straight line, as a crow flies |
| **L-inf** (Chebyshev) | the biggest absolute value | max(3, 4) = **4** | Only the worst single dimension counts |

For any vector, **L-inf ≤ L2 ≤ L1**. You can check it here: 4 ≤ 5 ≤ 7.

A distance is just one of these applied to `a - b`. For A = (1, 1) and B = (4, 5), the difference is (3, 4), so L1 = 7, L2 = 5 and L-inf = 4.

---

## The three that matter for RAG: cosine, dot, L2

| Measure | Formula | Range | Bigger means |
|---------|---------|-------|--------------|
| Dot product | `a · b = a1*b1 + a2*b2 + ...` | anything | more similar (and longer) |
| Cosine similarity | `(a · b) / (‖a‖ × ‖b‖)` | -1 to 1 | more similar |
| Cosine distance | `1 - cosine similarity` | 0 to 2 | **less** similar |
| L2 distance | `‖a - b‖` | 0 and up | **less** similar |

- **Symbols:** `‖a‖` is the L2 length of `a`. `a · b` is the dot product.
- **Relationship:** `a · b = ‖a‖ ‖b‖ cos(θ)`, where θ is the angle between the vectors. Cosine is the dot product with both lengths divided out.

### Worked example: length can change the winner

Query `q = (1, 0)`. Which document is "closest"?

| Doc | Vector | Dot with q | Cosine with q |
|-----|--------|-----------|---------------|
| a | (1, 0.1) | 1.0 | **0.995** (nearly the same direction) |
| b | (5, 5) | **5.0** | 0.707 (a 45° angle) |

- **Dot product picks b**, only because b is long.
- **Cosine picks a**, because it points almost exactly where q points.

Use dot product when length means something real, like popularity or confidence. Use cosine when only meaning matters, which is the usual case for text.

### Why normalised vectors make the choice not matter

If every vector has length 1 (`‖a‖ = ‖b‖ = 1`):
- Cosine = `a · b / (1 × 1)`, which equals the dot product.
- `‖a - b‖² = 2 - 2 × cos`. A larger cosine means a smaller L2 distance, so the ranking is identical, just flipped.

The numbers check out. For u = (0.707, 0.707) and v = (1, 0), `‖u - v‖² = 0.586` and `2 - 2 cos = 0.586`.

---

## Other distances (know they exist)

| Name | What it measures | Example use |
|------|------------------|-------------|
| **Jaccard** | Overlap of two sets: `size of overlap / size of union` | Deduplicating documents by their sets of words. In computer vision the same idea is called IoU. |
| **Edit distance** | Fewest single-character inserts, deletes or substitutions to turn one string into the other | `kitten → sitting` = 3. Used for fuzzy matching and typos. |
| **Mahalanobis** | L2 distance after correcting for correlated, differently-scaled features | Outlier detection. 5'0" and 180 lb is unusual even though each value alone is normal. |
| **Wasserstein** | How much "earth" you must move to turn one distribution into another | Comparing distributions that don't overlap at all, where KL is infinite. |

---

## Where this shows up in AI engineering

- **Vector databases** (FAISS, Qdrant, pgvector) ask you to choose `cosine`, `dot` or `l2` when you create an index. Check whether the embedding model normalises its output.
- **Retrieval code:** `top_k` = sort by similarity **descending** and take the first k.
- **Large-scale search** doesn't compare the query with every vector, which would cost O(n × d). It uses approximate search such as HNSW, a layered graph.
- **Regularisation** in training uses norms of the weights. L1 pushes some weights to exactly 0 (sparse). L2, also called weight decay, makes every weight a bit smaller.

---

## Code to remember

```python
import numpy as np

def top_k_cosine(query, X, k=5):
    # X: (n, d) document vectors, query: (d,)
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)   # make every row length 1
    qn = query / np.linalg.norm(query)
    sims = Xn @ qn                                      # n cosine similarities
    return np.argsort(sims)[::-1][:k]                   # [::-1] = descending, because these are similarities
```

---

## Mistakes to avoid

- **Wrong sort direction.** For a similarity use descending (`[::-1]`). For a distance use ascending. This is the bug I caught.
- **A zero vector in cosine** divides by zero. Guard it with `norms = np.where(norms == 0, 1, norms)`.
- **Features on different scales** (for example age 0 to 100 next to income 0 to 200,000) make L2 depend almost entirely on the large one. Standardise first.
- **KL divergence is not a distance.** KL(P‖Q) ≠ KL(Q‖P). See [01/09](01-09-information-theory.md).

---

## Words

- **Embedding:** a vector that represents meaning. Similar meanings end up near each other.
- **Normalise (L2-normalise):** divide a vector by its length so the length becomes 1.
- **Nearest neighbour:** the stored vector with the highest similarity, or smallest distance, to the query.

## Connects to

- **Next:** [01/10 Dimensionality Reduction](01-10-dimensionality-reduction.md). Why distances get less meaningful in very high dimensions.
- **Later:** 05/22 Embedding Models and Phase 11 RAG lessons, where this becomes the retrieval step.

---

## Self-test

1. What are the L1, L2 and L-inf norms of (6, 8)?
2. Your embeddings are unit length. Do cosine, dot product and L2 rank documents differently?
3. Two vectors have cosine similarity 0.99, but their L2 distance is 50. How?
4. You compute cosine **distance** and sort descending. What do you get back?
5. When would you deliberately choose dot product over cosine?
6. Which regulariser makes some weights exactly 0, L1 or L2?

<details>
<summary>Answers</summary>

1. L1 = 14, L2 = 10, L-inf = 8.
2. No. For unit vectors, dot = cosine and `‖a-b‖² = 2 - 2cos`, so all three give the same order.
3. They point the same way but have very different lengths, e.g. (1, 1) and (36, 36). Cosine ignores length. L2 doesn't.
4. The least relevant documents first. A distance has to be sorted ascending.
5. When length carries real information, such as popularity or confidence, and you want longer vectors to rank higher.
6. L1. Its constraint region is a diamond with corners on the axes, and the best solution tends to land on a corner, where a weight is 0. L2 is a circle with no corners, so weights shrink but rarely reach 0.

</details>
