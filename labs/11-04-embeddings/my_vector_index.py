# Lesson 11/04 Embeddings & Vector Representations: exact search, binary quantization + rescoring, and an IVF index.
# Run: python labs/11-04-embeddings/my_vector_index.py     (numpy only; self-checks use random vectors)
# The real-data run (use_it.py) embeds every lesson in this repo with bge-small and measures recall, memory and speed.
# Everything here assumes ROW vectors: a matrix X of shape (n, d) holds n vectors of dimension d.
import numpy as np

# Number of 1 bits in each byte value 0..255. Use it for popcount: POPCOUNT[byte_array] -> bit counts.
POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


# ---------------------------------------------------------------------------
# Your TODOs
# ---------------------------------------------------------------------------

def normalize(X):
    # TODO 1a: return X with every row scaled to unit L2 length. Leave all-zero rows as zeros (no division by 0).
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.where(norms == 0, 1, norms)


def exact_topk(Q, D, k):
    # TODO 1b: brute-force cosine search. Q (m, d) queries and D (n, d) documents are ALREADY normalized,
    # so cosine = dot product. Return (ids, scores), both shape (m, k), best first.
    #   - One matrix multiply for all queries (no Python loop over documents).
    #   - Don't fully sort n scores per query: np.argpartition picks the top k in O(n), then sort just those k.
    scores = Q @ D.T
    candidate_id = np.argpartition(-scores, k-1, axis=1)[:, :k] #get top candidates
    candidate_scores = np.take_along_axis(scores, candidate_id, axis=1) #take scores
    
    order = np.argsort(-candidate_scores, axis=1) # descending order of scores

    candidate_id = np.take_along_axis(candidate_id, order, axis=1) #get their ids
    candidate_scores = np.take_along_axis(candidate_scores, order, axis=1) # get their scores 
    return candidate_id, candidate_scores

def to_bits(X):
    # TODO 2a: binary quantization. Each dimension becomes 1 bit: 1 if the value is > 0, else 0.
    # Pack 8 bits per byte with np.packbits along axis 1, so (n, 384) float32 -> (n, 48) uint8. That's 32x smaller.
    binary = X > 0
    packed = np.packbits(binary, axis = 1)
    return packed 

def hamming_topk(Qb, Db, k):
    # TODO 2b: search packed bits by Hamming distance (number of differing bits). Smaller = more similar.
    #   distance(q, d) = popcount(q XOR d), summed over the bytes. Use np.bitwise_xor with broadcasting and POPCOUNT.
    #   Return ids (m, k), nearest first. Break ties by the lower document id (np.argsort(..., kind="stable")).
    #   Watch the dtype: summing many uint8 counts overflows uint8, so sum as int.
    xor = np.bitwise_xor(Qb[:, None, :], Db[None, :, :]) 
    bit_counts = POPCOUNT[xor]
    distances = np.sum(bit_counts, axis = 2,dtype=int)
    ids = np.argsort(distances,axis=1, kind="stable")
    return ids[:,:k]

def rescore(Q, D, candidate_ids, k):
    # TODO 3: the "search cheap, then rerank exact" pattern. For each query i, compute exact cosine scores
    # against only D[candidate_ids[i]] and return the ids of the best k of those, best first. Shape (m, k).
    result = []
    for i in range(Q.shape[0]):
        q = Q[i]

        candidates = D[candidate_ids[i]] 
        scores = candidates @ q
        order = np.argsort(-scores)[:k] # IDS DESCENDING
        result.append(candidate_ids[i][order])
    return np.array(result)



def kmeans(X, n_clusters, iters=10, seed=0):
    # TODO 4a: spherical k-means for unit vectors. Return centroids (n_clusters, d), each normalized.
    #   init: n_clusters distinct rows of X chosen with np.random.default_rng(seed).choice(len(X), n_clusters, replace=False)
    #   repeat iters times:
    #     assign every row to the centroid with the highest dot product
    #     new centroid = normalized mean of its assigned rows; a cluster with no rows keeps its old centroid
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(X), n_clusters, replace = False) 
    centroids = X[indices].copy()
    for _ in range(iters):
        similarities = X @ centroids.T
        assignments  = np.argmax(similarities, axis = 1)
        new_centroids = centroids.copy()

        for j in range(n_clusters):
            cluster_points = X[assignments == j]

            if len(cluster_points) > 0:
                mean_vector = cluster_points.mean(axis = 0)
                norm = np.linalg.norm(mean_vector)

                if norm > 0:
                    new_centroids[j] = mean_vector / norm
        centroids = new_centroids
    return centroids



class IVFIndex:
    # An inverted-file index (FAISS "IVF", pgvector "ivfflat"): cluster the documents, and at query time
    # scan only the `nprobe` clusters whose centroids are closest to the query.
    def __init__(self, D, n_clusters, seed=0):
        self.D = D
        self.centroids = kmeans(D, n_clusters, seed=seed)
        # TODO 4b: self.lists = a list with one np.array of document ids per cluster
        # (each document goes to its highest-dot centroid).
        similarities = D @ self.centroids.T
        assignments = np.argmax(similarities, axis=1)
        self.lists = []
        for j in range(n_clusters):
            
            doc_ids = np.where(assignments == j)[0]
            self.lists.append(doc_ids)

    def search(self, q, k, nprobe):
        # TODO 4c: one query vector q (d,). Pick the nprobe centroids with the highest dot product,
        # gather the ids in their lists, score those documents exactly, and return (top ids best first, n_scanned).
        # If fewer than k documents were scanned, return all of them, best first.
        scores = self.centroids @ q
        selected_centroids = np.argsort(-scores)[:nprobe]

        selected_lists = []
        for j in selected_centroids:
            selected_lists.append(self.lists[j])

        candidate_ids = np.concatenate(selected_lists) 
        n_scanned = len(candidate_ids)
        candidates = self.D[candidate_ids]
        exact_scores = candidates @ q
        order = np.argsort(-exact_scores)[:k]
        top_ids = candidate_ids[order]
        return (top_ids,n_scanned)



def recall_at_k(approx_ids, exact_ids):
    recalls = []
    for i in range(exact_ids.shape[0]):
        common = np.intersect1d(approx_ids[i], exact_ids[i])
        recall = len(common) / len(exact_ids[i])
        recalls.append(recall)
    return np.mean(recalls)


# ---------------------------------------------------------------------------
# Self-checks (random clustered vectors). Don't edit below.
# ---------------------------------------------------------------------------

def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"   ({detail})" if detail and not cond else ""))
    return cond


if __name__ == "__main__":
    ok = []
    rng = np.random.default_rng(42)
    centers = rng.normal(size=(20, 256))
    D = normalize(centers[rng.integers(0, 20, 3000)] + 0.3 * rng.normal(size=(3000, 256))).astype(np.float32)
    Q = normalize(D[rng.integers(0, 3000, 50)] + 0.15 * rng.normal(size=(50, 256))).astype(np.float32)

    X = np.array([[3.0, 4.0], [0.0, 0.0]])
    ok.append(check("1 normalize", np.allclose(normalize(X), [[0.6, 0.8], [0.0, 0.0]])))
    ids, sc = exact_topk(Q, D, 10)
    full = np.argsort(-(Q @ D.T), axis=1)[:, :10]
    ok.append(check("1 exact_topk ids", ids.shape == (50, 10) and np.array_equal(ids, full)))
    ok.append(check("1 exact_topk scores sorted", np.all(np.diff(sc, axis=1) <= 1e-7) and np.allclose(sc[:, 0], (Q @ D.T).max(1), atol=1e-6)))

    b = to_bits(np.array([[0.5, -1, 2, 0, -3, 1, 1, -1, 7, 7]], dtype=np.float32))
    ok.append(check("2 to_bits packs", b.dtype == np.uint8 and b.shape == (1, 2) and b[0, 0] == 0b10100110 and b[0, 1] == 0b11000000, str(b)))
    Db = to_bits(D)
    ok.append(check("2 32x smaller", D.nbytes // Db.nbytes == 32, f"{D.nbytes} vs {Db.nbytes} bytes"))
    hq = hamming_topk(to_bits(D[:5]), Db, 3)
    ok.append(check("2 hamming: a vector is its own nearest", np.array_equal(hq[:, 0], np.arange(5)), str(hq[:, 0])))
    twin = np.vstack([Db[:1], Db[:1], Db[:1]])
    ok.append(check("2 hamming ties -> lower id", np.array_equal(hamming_topk(Db[:1], twin, 3)[0], [0, 1, 2])))
    big = np.full((2, 600), 255, dtype=np.uint8)
    ok.append(check("2 no uint8 overflow", hamming_topk(np.zeros((1, 600), np.uint8), big, 1).shape == (1, 1)
                    and int(POPCOUNT[np.bitwise_xor(big[0], 0)].astype(int).sum()) == 4800))

    exact10, _ = exact_topk(Q, D, 10)
    ham10 = hamming_topk(to_bits(Q), Db, 10)
    ham100 = hamming_topk(to_bits(Q), Db, 100)
    r_bin = recall_at_k(ham10, exact10)
    r_res = recall_at_k(rescore(Q, D, ham100, 10), exact10)
    ok.append(check("5 recall of exact vs itself is 1", recall_at_k(exact10, exact10) == 1.0))
    ok.append(check("3 rescoring 100 binary candidates beats binary top-10", r_res > r_bin and r_res > 0.8, f"binary {r_bin:.2f}, rescored {r_res:.2f}"))
    rs = rescore(Q, D, ham100, 10)
    ok.append(check("3 rescore order matches exact scores", all(np.all(np.diff(Q[i] @ D[rs[i]].T) <= 1e-7) for i in range(50))))

    C = kmeans(D, 20, seed=0)
    ok.append(check("4 kmeans shape + unit norm", C.shape == (20, 256) and np.allclose(np.linalg.norm(C, axis=1), 1)))
    idx = IVFIndex(D, 20)
    ok.append(check("4 lists cover every doc once", sorted(np.concatenate(idx.lists).tolist()) == list(range(3000))))
    got = [idx.search(q, 10, nprobe=20) for q in Q]
    ok.append(check("4 nprobe = all clusters is exact", recall_at_k(np.array([g[0] for g in got]), exact10) == 1.0 and got[0][1] == 3000))
    got1 = [idx.search(q, 10, nprobe=1) for q in Q]
    r1 = recall_at_k(np.array([g[0] for g in got1]), exact10)
    scanned = np.mean([g[1] for g in got1])
    ok.append(check("4 nprobe=1 scans ~1/20 and still finds most", scanned < 400 and r1 > 0.7, f"recall {r1:.2f}, scanned {scanned:.0f}"))
    tiny = IVFIndex(D[:30], 10)
    ids_t, n_t = tiny.search(Q[0], 25, nprobe=1)
    ok.append(check("4 fewer than k scanned -> return them all", len(ids_t) == n_t))

    print(f"\n{sum(ok)}/{len(ok)} checks passed")
