# 01/10 Dimensionality Reduction (PCA)

> High-dimensional data usually lives on a much smaller "shape". PCA finds the flat version of that shape.

**Done:** 2026-10-08 · Quiz 3/3 · Bridge lesson 2 of 14
**Lesson:** [phases/01-math-foundations/10-dimensionality-reduction](../phases/01-math-foundations/10-dimensionality-reduction/)
**My code:** [`labs/01-10-dimensionality-reduction/my_pca.py`](../labs/01-10-dimensionality-reduction/my_pca.py). I wrote `inverse_transform` and it passes 5/5 checks.
**Run it:** `/usr/local/bin/python3 labs/01-10-dimensionality-reduction/my_pca.py`

---

## Remember these 3 things

1. **PCA = find the directions where the data spreads out the most, then keep only the top k.** You lose the least information possible for k numbers per point.
2. **Always subtract the mean first (centering).** Otherwise the first direction just points at the average point instead of along the spread.
3. **Reconstruction error = the variance you threw away** (the sum of the eigenvalues you dropped). A sharp drop in error at some k tells you the data's real dimensionality.

---

## In plain English

You have 1000 points, each with 768 numbers (an embedding). Many of those 768 numbers move together, so the data really only varies in a few independent ways. PCA:

1. finds the direction in which the points are most spread out (principal component 1, PC1)
2. finds the next most spread-out direction that is at a right angle to PC1 (PC2)
3. keeps going, ranking every direction by how much spread (variance) it captures

You then describe each point by **its position along the top k directions only**. That gives k numbers instead of 768.

**Analogy.** Photographing a flat pancake. From the side you see a thin line, which loses almost everything. From above you see the whole shape. PCA finds the "from above" angle, the one that shows the most spread.

---

## Worked example (2D down to 1D)

Points that lie almost on the line y = 2x:

```text
(1, 2)  (2, 4.1)  (3, 5.9)  (4, 8.2)  (5, 9.8)
```

| Step | What you do | Result |
|------|-------------|--------|
| 1. Center | Subtract the mean (3, 6) from every point | Points are now spread around (0, 0) |
| 2. Covariance | 2×2 matrix of how x and y vary together | Symmetric matrix |
| 3. Eigendecompose | Eigenvectors are directions, eigenvalues are the variance along each | Eigenvalues ≈ 12.22 and 0.005 |
| 4. Sort | Largest eigenvalue first | PC1 ≈ (0.45, 0.89), which is the y = 2x direction |
| 5. Project | Each point's coordinate along PC1 | 5 single numbers |

PC1 explains **99.96%** of the variance (12.22 / 12.225). Dropping PC2 loses almost nothing, so this 2D data is really 1D.

---

## The 5 steps in code

```python
mean = X.mean(axis=0)                     # 1. center
Xc = X - mean
C = np.cov(Xc, rowvar=False)              # 2. covariance (d, d). rowvar=False because rows are samples
vals, vecs = np.linalg.eigh(C)            # 3. eigh is for symmetric matrices
order = np.argsort(vals)[::-1]            # 4. eigh returns ASCENDING order, so reverse it
vals, vecs = vals[order], vecs[:, order]
components = vecs[:, :k].T                # shape (k, d): one direction per ROW
Z = Xc @ components.T                     # 5. project: (n, d) @ (d, k) -> (n, k)
```

**Shapes to remember:**
- `X` is (n, d), with n points and d features.
- `components` is **(k, d)**.
- `Z` is (n, k).

---

## My code: `inverse_transform`

The goal is to go from k numbers back to the original d features, which gives an approximate reconstruction.

```python
def inverse_transform(self, Z):
    return Z @ self.components + self.mean    # (n, k) @ (k, d) -> (n, d), then un-center
```

**Why it works.** Undo `transform` in reverse order, like taking off socks and shoes:

| `transform` did | `inverse_transform` undoes it |
|-----------------|-------------------------------|
| 1. subtract the mean | 2. add the mean back |
| 2. multiply by `components.T` | 1. multiply by `components` |

Multiplying by `components` reverses multiplying by `components.T` because the directions are **orthonormal**: each has length 1 and they're all at right angles. On that subspace the transpose acts as the inverse.

---

## How many components to keep (choosing k)

- **Explained variance ratio** of component i is `eigenvalue_i / sum of all eigenvalues`.
- Common rules:
  - keep enough components to reach 90 to 95% cumulative variance
  - look for the "elbow" in the plot
  - pick the k where your downstream metric stops improving
- **The noise-floor cliff:** if the data is truly 5-dimensional plus a little noise, reconstruction error drops sharply until k = 5 and then flattens. Components after the fifth only describe noise.
- **Anomaly detection:** a point that reconstructs badly doesn't fit the pattern of the rest. Flag it.

---

## The curse of dimensionality (why this matters)

In very high dimensions:

1. **Distances bunch together.** Everything is about equally far from everything else, so "nearest neighbour" means less.
2. **Space is mostly empty.** The volume sits in corners and edges.
3. **You need exponentially more data** to cover the space just as densely.

PCA is one way to fight this.

---

## PCA vs t-SNE vs UMAP

| Method | Use it for | What it keeps | Watch out |
|--------|-----------|---------------|-----------|
| **PCA** | Compressing data, preprocessing | Global spread, using straight-line directions only | Can't unroll curved shapes |
| **t-SNE** | 2D pictures of clusters | Who is near whom | **Distances between clusters mean nothing.** Slow, and the result changes from run to run. |
| **UMAP** | 2D pictures at scale | Local plus some global structure | Its knobs (`n_neighbors`, `min_dist`) change the picture |
| **Kernel PCA** | Curved shapes, like rings | Nonlinear structure | Needs an n × n matrix, so memory limits it |

**Rule of thumb:** use PCA to *compress*, and t-SNE or UMAP to *look*.

---

## Where this shows up in AI engineering

- **Shrinking embeddings** to cut vector database memory and cost. Some providers also offer shorter embeddings directly.
- **Visualising** an embedding space to debug retrieval: do the "refund" questions cluster together?
- **Outlier detection** on inputs through reconstruction error.

---

## Mistakes to avoid

- **Forgetting to center.** In my practice data the offset was +10, so PC1 pointed at the mean. Remember to add the mean back in `inverse_transform` too.
- **`eigh` returns ascending order.** Reverse it.
- **`np.cov` without `rowvar=False`** treats columns as samples. You'd get an (n, n) matrix instead of (d, d).
- **Reading meaning into t-SNE gaps.** The gaps between clusters are not real distances.

## Words

- **Variance:** how spread out values are.
- **Covariance matrix:** how each pair of features varies together. Its shape is (d, d).
- **Eigenvector / eigenvalue:** a direction the matrix only stretches, and how much it stretches it. For a covariance matrix, that's a direction and the variance along it.
- **Orthonormal:** length 1 and at right angles to each other.

## Connects to

- **Before:** [01/14 Norms and Distances](01-14-norms-and-distances.md), the distances that high dimensions make less useful.
- **Next:** [01/06 Probability and Distributions](01-06-probability-and-distributions.md).

---

## Self-test

1. In one sentence, what does PCA do?
2. Why center before PCA?
3. `X` is (1000, 768) and you keep k = 50. What are the shapes of `components` and `Z`?
4. Write `inverse_transform` from memory, and explain the order of operations.
5. With k = d (keep everything), what is the reconstruction error?
6. Error drops sharply until k = 5 and then flattens. What does that tell you?
7. A t-SNE plot shows clusters A and B far apart. Are they very different?

<details>
<summary>Answers</summary>

1. It finds the directions of largest spread and re-describes each point using only its positions along the top k of them.
2. PCA measures spread *around the mean*. Without centering, the offset from the origin dominates PC1.
3. `components` is (50, 768) and `Z` is (1000, 50).
4. `return Z @ self.components + self.mean`. Rotate back first, then un-center, the reverse of transform's subtract-then-rotate.
5. About 0, apart from floating-point noise. Nothing was dropped, so it's a rotation and back.
6. The data is really about 5-dimensional. Components after the fifth describe noise.
7. Not necessarily. t-SNE keeps local neighbourhoods only, so the distances between clusters aren't meaningful.

</details>
