# Bridge lesson 14: KV cache, prefix caching and Flash-style tiled attention.
# Run: python labs/07-12-kv-cache/my_kv.py   (needs numpy)
# Toy model: ONE attention layer, ONE head, width D. Each token is a (D,) vector.
# "Generation": the attention output for the last token is turned into the next input token by next_input().
import numpy as np

D = 8
rng = np.random.default_rng(0)
Wq, Wk, Wv, Wo = (rng.normal(size=(D, D)) / np.sqrt(D) for _ in range(4))

PROJECTED = {"n": 0}  # counts how many token rows went through the Q/K/V projection (the expensive part)


# ---------------------------------------------------------------------------
# Provided helpers. Use them; don't edit.
# ---------------------------------------------------------------------------

def project(X):
    # X: (t, D) -> Q, K, V each (t, D). Every call adds t to the projection counter.
    PROJECTED["n"] += len(X)
    return X @ Wq, X @ Wk, X @ Wv


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def attend(q, K, V):
    # One query against all keys/values. q: (D,), K, V: (t, D) -> (D,)
    return softmax(K @ q / np.sqrt(D)) @ V


def next_input(out):
    # Toy "LM head + embedding of the sampled token": turns an attention output into the next input vector.
    return np.tanh(out @ Wo)


def generate_naive(prompt, n_new):
    # No cache: every step re-projects the WHOLE prefix, then attends with the last query.
    xs = list(prompt)
    outs = []
    for _ in range(n_new):
        Q, K, V = project(np.stack(xs))
        out = attend(Q[-1], K, V)
        outs.append(out)
        xs.append(next_input(out))
    return np.stack(outs)


# ---------------------------------------------------------------------------
# Your TODOs
# ---------------------------------------------------------------------------

def kv_cache_bytes(n_tokens, n_layers, n_kv_heads, d_head, dtype_bytes=2, batch=1):
    # TODO 1: total KV cache size in bytes (the formula you just used by hand).
    return 2 * n_tokens * n_layers * n_kv_heads * d_head * dtype_bytes * batch


class KVCache:
    def __init__(self):
        self.K = []  # list of (D,) key vectors, one per token seen so far
        self.V = []

    def append(self, k, v):
        self.K.append(k)
        self.V.append(v)

    def arrays(self):
        return np.stack(self.K), np.stack(self.V)

    def __len__(self):
        return len(self.K)

    def copy(self):
        # TODO 2: return a NEW KVCache holding the same K/V entries, so appending to the copy
        # never changes the original. (Think about what `new.K = self.K` would do.)
        new = KVCache()
        new.K = self.K.copy()
        new.V = self.V.copy()
        return new


def prefill(cache, X):
    # TODO 3: process a block of tokens X (t, D) in ONE parallel pass:
    #   - call project(X) exactly once
    #   - append every row's k and v to the cache
    #   - return the query of the LAST row (D,), which the next decode step attends with
    Q,K,V = project(X)
    for k,v in zip(K,V):
        cache.append(k,v)
    return Q[-1]


def decode(cache, q, n_new):
    # TODO 4: generate n_new outputs, one token at a time, using the cache.
    # Each step:
    #   - out = attend(q, K_cache, V_cache); save it
    #   - x = next_input(out)
    #   - project ONLY x (shape (1, D)), append its k, v to the cache, and its q becomes the next q
    # Must give exactly the same outputs as generate_naive(). Skip projecting the very last x
    # (generate_naive never uses it either), so projections = n_new - 1 here.
    # Return np.stack(outs), shape (n_new, D).
    outs = np.zeros((n_new, D))
    for _ in range(n_new):
        out = attend(q,cache.K,cache.V)
        outs[_] = out
        if _<n_new-1:
            x = next_input(out)
            Q, K, V = project(x.reshape(1, -1))
            cache.append(K[0], V[0])
            q = Q[0]
    return outs.reshape(n_new, D)


def generate_cached(prompt, n_new):
    cache = KVCache()
    q = prefill(cache, prompt)
    return decode(cache, q, n_new)


def serve_with_prefix_cache(system, users, n_new):
    # Provided: prompt caching. Prefill the shared system prompt ONCE, then branch per request.
    shared = KVCache()
    prefill(shared, system)
    results = []
    for u in users:
        c = shared.copy()
        q = prefill(c, u)
        results.append(decode(c, q, n_new))
    return results


def tiled_attention(q, K, V, tile=4):
    # TODO 5: Flash-style online softmax. Same result as attend(q, K, V), but look at only
    # `tile` keys/values at a time and never build the full score vector.
    # Keep three running values:  m = max score so far (start -inf),  s = sum of exp(score - m),
    #                             acc = sum of exp(score - m) * v   (shape (D,))
    # For each block: scores = K_blk @ q / sqrt(D);  new_m = max(m, scores.max())
    #   rescale the OLD s and acc by exp(m - new_m)   (exp(-inf) = 0 is fine in numpy, the first block works)
    #   add this block's exp(scores - new_m) terms;  m = new_m
    # Return acc / s.
    D = q.shape[0]
    m = -np.inf
    s = 0.0
    acc = np.zeros(D)
    for start in range(0,K.shape[0],tile):
        K_blk = K[start: start+tile]
        V_blk = V[start: start+tile]
        scores = K_blk @ q / np.sqrt(D)
        new_m = max(m,scores.max())
        scale = np.exp(m-new_m)
        s = s * scale + np.sum(np.exp(scores - new_m))
        acc = acc * scale + (np.exp(scores - new_m) @ V_blk)
        m = new_m
    return acc / s


# ---------------------------------------------------------------------------
# Self-checks. Don't edit below.
# ---------------------------------------------------------------------------

def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"   ({detail})" if detail and not cond else ""))
    return cond


if __name__ == "__main__":
    ok = []
    ok.append(check("1 bytes: Llama-3-8B, 1 token", kv_cache_bytes(1, 32, 8, 128) == 131072))
    ok.append(check("1 bytes: Llama-3-70B, 32K, batch 4, fp8",
                    kv_cache_bytes(32768, 80, 8, 128, dtype_bytes=1, batch=4) == 21474836480))

    a = KVCache(); a.append(np.ones(D), np.ones(D))
    b = a.copy(); b.append(np.zeros(D), np.zeros(D))
    ok.append(check("2 copy is independent", len(a) == 1 and len(b) == 2, f"len(a)={len(a)}, len(b)={len(b)}"))

    P, N = 5, 20
    prompt = rng.normal(size=(P, D))
    PROJECTED["n"] = 0; naive = generate_naive(prompt, N); naive_proj = PROJECTED["n"]
    PROJECTED["n"] = 0; cached = generate_cached(prompt, N); cached_proj = PROJECTED["n"]
    ok.append(check("3+4 cached == naive outputs", cached.shape == naive.shape and np.allclose(cached, naive, atol=1e-12)))
    ok.append(check("3+4 cached projects P + N - 1 rows", cached_proj == P + N - 1, f"got {cached_proj}"))
    print(f"     projections: naive {naive_proj}  vs cached {cached_proj}")

    system = rng.normal(size=(30, D))
    users = [rng.normal(size=(4, D)) for _ in range(3)]
    PROJECTED["n"] = 0
    expected = [generate_naive(np.vstack([system, u]), 6) for u in users]
    full_proj = PROJECTED["n"]
    PROJECTED["n"] = 0
    got = serve_with_prefix_cache(system, users, 6)
    pc_proj = PROJECTED["n"]
    ok.append(check("prefix cache: every request matches a full recompute",
                    all(np.allclose(g, e, atol=1e-12) for g, e in zip(got, expected))))
    ok.append(check("prefix cache: system prompt projected once", pc_proj == 30 + 3 * (4 + 6 - 1), f"got {pc_proj}"))
    print(f"     projections: naive per request {full_proj}  vs prefix-cached {pc_proj}")

    K = rng.normal(size=(13, D)); V = rng.normal(size=(13, D)); q = rng.normal(size=D)
    ref = attend(q, K, V)
    ok.append(check("5 tiled == attend for tiles 1,3,4,7,100",
                    all(np.allclose(tiled_attention(q, K, V, t), ref, atol=1e-12) for t in [1, 3, 4, 7, 100])))
    big = tiled_attention(q * 1000, K, V, 3)
    ok.append(check("5 tiled is stable with huge scores", np.all(np.isfinite(big)) and np.allclose(big, attend(q * 1000, K, V))))

    print(f"\n{sum(ok)}/{len(ok)} checks passed")
