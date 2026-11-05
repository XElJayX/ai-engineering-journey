# Bridge lesson 12: Self-Attention. Build scaled dot-product attention, causal masking and multi-head attention.
# Run: python labs/07-02-self-attention/my_attention.py   (needs numpy)
# Shapes: X is (n, d_model) = n tokens. Q, K are (n, d_k). V is (n, d_v). Weights matrices act as X @ W.
import numpy as np


def softmax(x, axis=-1):
    # Provided (you built this in 01/06): stable softmax along an axis. exp(-inf) = 0, so masked entries get weight 0.
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


# ---------------------------------------------------------------------------
# Your TODOs
# ---------------------------------------------------------------------------

def attention(Q, K, V, mask=None):
    # Scaled dot-product attention. Return (output, weights).
    #   scores  = Q @ K.T / sqrt(d_k)                      shape (n, n): row i = how much token i attends to each token
    #   mask    : optional boolean (n, n) array, True = "allowed to look". Set NOT-allowed scores to -np.inf.
    #   weights = softmax(scores) over each ROW            each row sums to 1
    #   output  = weights @ V                              shape (n, d_v)
    d_k = K.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    if mask is not None:
        scores[~mask] = -np.inf
    weights = softmax(scores)
    output = weights @ V 
    return (output, weights)


def causal_mask(n):
    # Boolean (n, n): token i may look at tokens 0..i (itself and the past), never the future.
    # Hint: np.tril
    return np.tril(np.ones(n, dtype = bool))
    


def multi_head_attention(X, Wq, Wk, Wv, Wo, n_heads, mask=None):
    # X: (n, d_model). Wq, Wk, Wv, Wo: (d_model, d_model). d_model must divide evenly by n_heads (raise ValueError).
    # 1. Q = X @ Wq, K = X @ Wk, V = X @ Wv                      each (n, d_model)
    # 2. Split the COLUMNS into n_heads blocks of d_head = d_model // n_heads: head h uses columns h*d_head:(h+1)*d_head
    # 3. Run attention() on each head separately (same mask for every head)
    # 4. Concatenate the head outputs side by side -> (n, d_model), then multiply by Wo
    # Return (output, list_of_per_head_weights).
    d_model = X.shape[1]
    if d_model % n_heads != 0:
        raise ValueError("d_model must divide evenly by n_heads")
    d_head = d_model // n_heads
    Q = X @ Wq
    K = X @ Wk
    V = X @ Wv
    head_outputs = []
    head_weights = []
    for h in range(n_heads):
        Q_h = Q[:, h*d_head:(h+1)*d_head]
        K_h = K[:, h*d_head:(h+1)*d_head]
        V_h = V[:, h*d_head:(h+1)*d_head]
        out_h, w_h = attention(Q_h, K_h, V_h, mask)
        head_outputs.append(out_h)
        head_weights.append(w_h)
    concatenated = np.hstack(head_outputs)
    output = concatenated @ Wo
    return (output, head_weights)
    


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


def t_toy():
    q = np.array([[1.0, 0.0]])
    K = np.array([[2.0, 0.0], [0.0, 2.0], [0.5, 0.5]])
    V = np.eye(3)
    out, w = attention(q, K, V)
    ok = w.shape == (1, 3) and np.allclose(w, [[0.629, 0.153, 0.218]], atol=1e-3) and np.allclose(out, w)
    return ok, f"'it' attends to [animal, street, it] = {np.round(w[0], 3).tolist()}"


def t_shapes():
    rng = np.random.default_rng(0)
    Q, K, V = rng.normal(size=(5, 4)), rng.normal(size=(5, 4)), rng.normal(size=(5, 3))
    out, w = attention(Q, K, V)
    ok = out.shape == (5, 3) and w.shape == (5, 5) and np.allclose(w.sum(1), 1) and np.allclose(out, w @ V)
    return ok, f"output {out.shape}, weights {w.shape}, every row sums to 1"


def t_scaling():
    rng = np.random.default_rng(1)
    d = 512
    Q, K, V = rng.normal(size=(8, d)), rng.normal(size=(8, d)), rng.normal(size=(8, d))
    _, w = attention(Q, K, V)
    unscaled = softmax(Q @ K.T)
    return w.max(1).mean() < 0.6 and unscaled.max(1).mean() > 0.85, \
        f"d_k=512: avg top weight scaled {w.max(1).mean():.2f} vs unscaled {unscaled.max(1).mean():.4f} (one-hot = frozen gradients)"


def t_causal():
    m = causal_mask(4)
    rng = np.random.default_rng(2)
    X = rng.normal(size=(4, 6))
    out, w = attention(X, X, X, mask=m)
    X2 = X.copy()
    X2[3] += 100.0                                   # change only the LAST token
    out2, _ = attention(X2, X2, X2, mask=m)
    ok = m.dtype == bool and m.tolist()[1] == [True, True, False, False] and np.allclose(np.triu(w, 1), 0) \
        and abs(w[0, 0] - 1) < 1e-12 and np.allclose(out[:3], out2[:3])
    return ok, "future weights are exactly 0; token 0 sees only itself; editing token 3 leaves tokens 0-2 unchanged"


def t_mha():
    rng = np.random.default_rng(3)
    n, d = 5, 8
    X = rng.normal(size=(n, d))
    Ws = [rng.normal(size=(d, d)) / np.sqrt(d) for _ in range(4)]
    out1, heads1 = multi_head_attention(X, *Ws[:3], np.eye(d), 1)
    ref1, _ = attention(X @ Ws[0], X @ Ws[1], X @ Ws[2])
    out2, heads2 = multi_head_attention(X, *Ws, 2)
    Q, K, V = X @ Ws[0], X @ Ws[1], X @ Ws[2]
    h0, _ = attention(Q[:, :4], K[:, :4], V[:, :4])
    h1, _ = attention(Q[:, 4:], K[:, 4:], V[:, 4:])
    ref2 = np.hstack([h0, h1]) @ Ws[3]
    try:
        multi_head_attention(X, *Ws, 3)
        raised = False
    except ValueError:
        raised = True
    ok = np.allclose(out1, ref1) and np.allclose(out2, ref2) and len(heads2) == 2 and heads2[0].shape == (n, n) and raised
    return ok, f"1 head = plain attention; 2 heads of d_head=4 match; output {out2.shape}; d_model=8 with 3 heads rejected"


def t_order_blind():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(6, 8))
    Ws = [rng.normal(size=(8, 8)) / np.sqrt(8) for _ in range(4)]
    perm = np.array([3, 0, 5, 1, 4, 2])
    out, _ = multi_head_attention(X, *Ws, 2)
    out_p, _ = multi_head_attention(X[perm], *Ws, 2)
    return np.allclose(out_p, out[perm]), "shuffle the tokens -> outputs are the same rows, shuffled. Attention alone can't see word ORDER"


if __name__ == "__main__":
    tests = [
        ("toy example: 'it' attends mostly to 'animal'", t_toy),
        ("shapes, rows sum to 1, output = weights @ V", t_shapes),
        ("dividing by sqrt(d_k) prevents one-hot softmax", t_scaling),
        ("causal mask: no peeking at future tokens", t_causal),
        ("multi-head: split columns, attend per head, concat, project", t_mha),
        ("attention without positions is order-blind", t_order_blind),
    ]
    passed = sum(check(name, fn) for name, fn in tests)
    print(f"\n  {passed}/{len(tests)} checks passed")
