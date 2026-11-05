# Bridge lesson 13: GPT. Wire a decoder-only transformer, its shift-by-one loss, and the sampling loop.
# Run: python labs/07-07-gpt/my_gpt.py   (needs numpy)
import numpy as np


# ---------------------------------------------------------------------------
# Provided building blocks (attention is your 07/02 code, batched over heads).
# ---------------------------------------------------------------------------
def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def log_softmax(x, axis=-1):
    m = np.max(x, axis=axis, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=axis, keepdims=True))


def layer_norm(x, eps=1e-5):
    # Normalise each token's vector to mean 0, variance 1 (learned gain/bias omitted for simplicity).
    return (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + eps)


def causal_self_attention(x, p):
    n, d = x.shape
    h = p["n_heads"]
    dh = d // h
    q, k, v = (x @ p[w] for w in ("Wq", "Wk", "Wv"))
    q, k, v = (t.reshape(n, h, dh).transpose(1, 0, 2) for t in (q, k, v))      # (h, n, dh)
    scores = q @ k.transpose(0, 2, 1) / np.sqrt(dh)                              # (h, n, n)
    scores = np.where(np.tril(np.ones((n, n), dtype=bool)), scores, -np.inf)
    out = softmax(scores) @ v                                                   # (h, n, dh)
    return out.transpose(1, 0, 2).reshape(n, d) @ p["Wo"]


def mlp(x, p):
    # Position-wise feed-forward: expand 4x, nonlinearity, project back.
    return np.maximum(0, x @ p["W1"]) @ p["W2"]


def init_params(vocab_size, d_model=32, n_heads=4, n_layers=2, max_len=64, seed=0):
    rng = np.random.default_rng(seed)
    g = lambda *shape: rng.normal(0, 0.02, shape)
    return {
        "E": g(vocab_size, d_model),            # token embeddings, also reused as the output layer ("tied")
        "P": g(max_len, d_model),               # learned position embeddings (attention is order-blind!)
        "blocks": [{"n_heads": n_heads, "Wq": g(d_model, d_model), "Wk": g(d_model, d_model),
                    "Wv": g(d_model, d_model), "Wo": g(d_model, d_model),
                    "W1": g(d_model, 4 * d_model), "W2": g(4 * d_model, d_model)} for _ in range(n_layers)],
    }


# ---------------------------------------------------------------------------
# Your TODOs
# ---------------------------------------------------------------------------

def shift_for_training(token_ids):
    # [t1, t2, t3, t4] -> inputs [t1, t2, t3], targets [t2, t3, t4]. Return (inputs, targets) as lists.
    return (token_ids[:-1],token_ids[1:])
    


def decoder_block(x, p):
    # Pre-norm residual block (GPT-2 / Llama style). Two lines:
    #   x = x + attention(layer_norm(x))      "read from other tokens, ADD the result to the stream"
    #   x = x + mlp(layer_norm(x))            "think about it per token, ADD the result to the stream"
    x = x + causal_self_attention(layer_norm(x),p)
    x = x + mlp(layer_norm(x),p)
    return x


def gpt_logits(token_ids, params):
    # 1. x = E[token_ids] + P[positions 0..n-1]                         shape (n, d_model)
    # 2. run x through every block in params["blocks"]
    # 3. final layer_norm, then logits = x @ E.T  (tied embeddings)    shape (n, vocab_size)
    n = len(token_ids)
    token_emb = params["E"][token_ids]
    pos_emb = params["P"][np.arange(n)]
    x = token_emb + pos_emb
    for block in params["blocks"]:
        x = decoder_block(x, block)
    x = layer_norm(x)
    logits = x @ params["E"].T
    return logits


def sequence_loss(logits, targets):
    # Mean cross-entropy over positions: average of -log_softmax(logits[i])[targets[i]].
    log_probs = np.array([log_softmax(logit) for logit in logits]) 
    target = log_probs[np.arange(len(logits)),targets]
    return -sum(target) / len(logits)

def top_k_filter(probs, k):
    # Keep the k largest probabilities, zero the rest, renormalise to sum 1.
    top_indices = np.argsort(probs)[-k:]
    filtered = np.zeros_like(probs)
    filtered[top_indices] = probs[top_indices]
    filtered = filtered / filtered.sum()
    return filtered

def top_p_filter(probs, p):
    # Keep the SMALLEST set of most-likely tokens whose cumulative probability is >= p. Zero the rest, renormalise.
    # Always keep at least the top token.
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]
    cumulative = np.cumsum(sorted_probs)
    n_keep = np.searchsorted(cumulative, p) + 1 
    keep_indices = sorted_indices[:n_keep]
    filtered = np.zeros_like(probs)
    filtered[keep_indices] = probs[keep_indices]
    filtered = filtered / filtered.sum()
    return filtered
    


def min_p_filter(probs, min_p):
    # Keep tokens with prob >= min_p * max(probs). Zero the rest, renormalise.
    threshold = min_p * np.max(probs)
    mask = probs >= threshold
    filtered = np.zeros_like(probs)
    filtered[mask] = probs[mask]
    filtered = filtered / filtered.sum()
    return filtered


def generate(prompt_ids, params, n_new, rng, temperature=1.0, filt=None):
    # Autoregressive loop. Repeat n_new times:
    #   logits = gpt_logits(current_ids, params); take the LAST row (the prediction for the next token)
    #   temperature == 0  -> greedy: argmax
    #   otherwise         -> probs = softmax(last_logits / temperature); if filt: probs = filt(probs)
    #                        next = rng.choice(len(probs), p=probs)
    #   append next to current_ids
    # Return the full list of ids (prompt + generated).
    current_ids = list(prompt_ids)
    for _ in range(n_new):
        logits = gpt_logits(current_ids,params)
        last_logits = logits[-1]
        if temperature == 0:
            next_token = int(np.argmax(last_logits))
        else:
            last_logits = last_logits / temperature
            probs = softmax (last_logits)
            if filt is not None:
                probs = filt(probs)
            next_token = rng.choice(len(probs), p=probs)
        current_ids.append(next_token)
    return current_ids


# ---------------------------------------------------------------------------
# Self-checks. Don't edit below this line; make them pass.
# ---------------------------------------------------------------------------
A = np.array([0.90, 0.05, 0.03, 0.01, 0.01])
B = np.array([0.22, 0.21, 0.20, 0.19, 0.18])


def check(name, fn):
    try:
        ok, detail = fn()
    except NotImplementedError:
        ok, detail = False, "not implemented yet"
    except Exception as e:
        ok, detail = False, f"{type(e).__name__}: {e}"
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    return ok


def t_shift():
    i, t = shift_for_training([5, 9, 2, 7, 3, 8])
    return list(i) == [5, 9, 2, 7, 3] and list(t) == [9, 2, 7, 3, 8], f"inputs {list(i)}, targets {list(t)}"


def t_block_residual():
    p = init_params(10)["blocks"][0]
    zero = {**p, "Wo": np.zeros_like(p["Wo"]), "W2": np.zeros_like(p["W2"])}
    x = np.random.default_rng(1).normal(size=(4, 32))
    same = np.allclose(decoder_block(x, zero), x)
    changed = not np.allclose(decoder_block(x, p), x)
    return same and changed, "with attention and MLP outputs zeroed, the block returns x unchanged: the residual highway"


def t_logits_causal():
    params = init_params(20)
    ids = [3, 1, 4, 1, 5, 9]
    L = gpt_logits(ids, params)
    L2 = gpt_logits(ids[:-1] + [7], params)
    ok = L.shape == (6, 20) and np.allclose(L[:5], L2[:5]) and not np.allclose(L[5], L2[5])
    return ok, f"logits {L.shape}; changing the last token changes ONLY the last row"


def t_loss_baseline():
    V = 50
    params = init_params(V)
    ids = list(np.random.default_rng(2).integers(0, V, 33))
    inp, tgt = shift_for_training(ids)
    loss = sequence_loss(gpt_logits(inp, params), tgt)
    return abs(loss - np.log(V)) < 0.05, f"untrained loss {loss:.3f} vs ln(V) = ln {V} = {np.log(V):.3f}  (01/09: uniform guessing)"


def t_filters():
    counts = {name: [int((f(d, a) > 0).sum()) for d in (A, B)]
              for name, f, a in (("top_k=2", top_k_filter, 2), ("top_p=0.9", top_p_filter, 0.9), ("min_p=0.1", min_p_filter, 0.1))}
    sums = all(abs(f(d, a).sum() - 1) < 1e-12 for f, a in ((top_k_filter, 2), (top_p_filter, 0.9), (min_p_filter, 0.1)) for d in (A, B))
    ok = counts == {"top_k=2": [2, 2], "top_p=0.9": [1, 5], "min_p=0.1": [1, 5]} and sums and np.isclose(top_k_filter(A, 2)[1], 0.05 / 0.95)
    return ok, f"tokens kept [A, B]: {counts}; P('Lyon') under top_k=2 = {top_k_filter(A, 2)[1]:.3f}"


def t_generate():
    params = init_params(20)
    g1 = generate([1, 2, 3], params, 6, np.random.default_rng(0), temperature=0)
    g2 = generate([1, 2, 3], params, 6, np.random.default_rng(99), temperature=0)
    s1 = generate([1, 2, 3], params, 6, np.random.default_rng(0), temperature=1.5, filt=lambda q: top_p_filter(q, 0.9))
    s2 = generate([1, 2, 3], params, 6, np.random.default_rng(1), temperature=1.5, filt=lambda q: top_p_filter(q, 0.9))
    prompt = [4, 8, 15, 16]
    first = generate(prompt, params, 1, np.random.default_rng(0), temperature=0)[-1]
    want = int(np.argmax(gpt_logits(prompt, params)[-1]))
    ok = len(g1) == 9 and g1[:3] == [1, 2, 3] and g1 == g2 and s1 != s2 and first == want
    return ok, f"greedy {g1} (same for any seed); sampled seed0 {s1[3:]} vs seed1 {s2[3:]}; next token from the LAST position: {first == want}"


if __name__ == "__main__":
    tests = [
        ("shift-by-one inputs and targets", t_shift),
        ("decoder block is a residual stream", t_block_residual),
        ("logits are causal: no future leaks backwards", t_logits_causal),
        ("untrained GPT loss = ln(vocab)", t_loss_baseline),
        ("top-k / top-p / min-p on distributions A and B", t_filters),
        ("generate: greedy is deterministic, sampling is not", t_generate),
    ]
    passed = sum(check(name, fn) for name, fn in tests)
    print(f"\n  {passed}/{len(tests)} checks passed")
