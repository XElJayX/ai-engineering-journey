# Bridge lesson 6: Imbalanced Data. Fix a classifier that learned to always say "safe".
# Run: /usr/local/bin/python3 labs/02-17-imbalanced-data/my_imbalanced.py   (needs numpy)
# Labels: 1 = minority (e.g. "injection"), 0 = majority ("safe").
import numpy as np


def compute_class_weights(y):
    # Return a numpy array with ONE weight per sample (same length as y), using the
    # "balanced" rule: w_c = n_samples / (n_classes * n_c), where n_c is the count of class c.
    # Hint: np.unique(y, return_counts=True)
    weights = np.array([],dtype=float)
    classes,counts = np.unique(y,return_counts=True)
    val = dict(zip(classes,counts))
    w_positive = len(y) / (2 * val[1])
    w_negative = len(y) / (2* val[0])
    for i in y:
        if i == 1:
            weights = np.append(weights,w_positive)
        else:
            weights = np.append(weights,w_negative)
    return weights



def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def train_logreg(X, y, sample_weights, lr=0.1, epochs=300):
    # Logistic regression trained with weighted binary cross-entropy (gradient descent).
    w, b = np.zeros(X.shape[1]), 0.0
    for _ in range(epochs):
        p = sigmoid(X @ w + b)
        # TODO (one line): the per-sample error that drives the gradient.
        # Unweighted BCE gives (p - y). How do the sample weights enter?
        error = sample_weights * (p-y)
        if error is None:
            raise NotImplementedError
        w -=lr * (X.T @ error) / len(y)
        b -= lr * np.mean(error)
    return w, b


def smote(X_min, k, n_synthetic,rng):
    # X_min: (m, d) array of MINORITY samples only. Return an (n_synthetic, d) array of new points.
    # For each new point:
    #   1. pick a random minority point x            (rng.integers(0, m))
    #   2. find its k nearest OTHER minority points  (Euclidean; exclude x itself)
    #   3. pick one of those neighbours n at random
    #   4. new = x + t * (n - x), with t = rng.random()
    # Clamp k to at most m - 1.
    m = len(X_min)
    k = min(k, m - 1)
    if m < 2:
        raise ValueError("SMOTE requires at least 2 minority samples")
    if k == 0:
        raise ValueError("SMOTE requires at least 2 minority samples")
    res = []
    for i in range(n_synthetic):
        index = rng.integers(0,m)
        point = X_min[index]
        
        distances = np.linalg.norm(X_min - point, axis=1)
        distances[index] = np.inf
        indexes = np.argsort(distances)

        neighbours = indexes[:k]
        index = rng.integers(0,k)
        neighbour = neighbours[index]
        
        t = rng.random()
        new = point + (t * ( X_min[neighbour] - point))
        res.append(new)

    return np.array(res).reshape(-1, X_min.shape[1])



def mcc(tp, fp, fn, tn):
    # Matthews Correlation Coefficient:
    #   (tp*tn - fp*fn) / sqrt((tp+fp) * (tp+fn) * (tn+fp) * (tn+fn))
    # +1 = perfect, 0 = no better than chance, -1 = perfectly wrong.
    # If the denominator is 0 (e.g. the model never predicts positive), return 0.0.
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denom == 0:
        return 0.0
    return (tp * tn - fp * fn) / denom


def best_f1_threshold(y_true, probs):
    # Try every distinct value in probs as a threshold (predict 1 if prob >= threshold).
    # Return (threshold, f1) with the highest F1. Ties: the HIGHER threshold.
    # Call this on a VALIDATION set, never the test set.
    for prob in probs:
        y_pred = (probs >= prob).astype(int)
        tp, fp, fn, tn = counts(y_true, y_pred)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        # Keep track of the best threshold and F1 score.
        if 'best_f1' not in locals() or f1 > best_f1 or (f1 == best_f1 and prob > best_threshold):
            best_f1 = f1
            best_threshold = prob
    return best_threshold, best_f1


# ---------------------------------------------------------------------------
# Self-checks. Don't edit below this line; make them pass.
# ---------------------------------------------------------------------------

def counts(y_true, y_pred):
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    return tp, fp, fn, tn


def recall_of(y_true, y_pred):
    tp, _, fn, _ = counts(y_true, y_pred)
    return tp / (tp + fn) if tp + fn else 0.0


def make_data(n_maj=950, n_min=50, seed=42):
    rng = np.random.RandomState(seed)
    X = np.vstack([rng.randn(n_maj, 2), rng.randn(n_min, 2) * 0.8 + 2.5])
    y = np.concatenate([np.zeros(n_maj), np.ones(n_min)])
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


def check(name, fn):
    try:
        ok, detail = fn()
    except NotImplementedError:
        ok, detail = False, "not implemented yet"
    except Exception as e:
        ok, detail = False, f"{type(e).__name__}: {e}"
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    return ok


def t_weights():
    y = np.array([0] * 9950 + [1] * 50)
    w = compute_class_weights(y)
    ok = w.shape == y.shape and abs(w[0] - 0.502513) < 1e-4 and abs(w[-1] - 100.0) < 1e-9 \
        and abs(w[y == 0].sum() - w[y == 1].sum()) < 1e-6
    return ok, f"w_safe={w[0]:.4f}, w_inj={w[-1]:.1f}, class totals {w[y == 0].sum():.0f} / {w[y == 1].sum():.0f}"


def t_weighted_training():
    X, y = make_data()
    plain = train_logreg(X, y, np.ones(len(y)))
    weighted = train_logreg(X, y, compute_class_weights(y))
    r_plain = recall_of(y, (sigmoid(X @ plain[0] + plain[1]) >= 0.5).astype(int))
    r_w = recall_of(y, (sigmoid(X @ weighted[0] + weighted[1]) >= 0.5).astype(int))
    return r_w > r_plain + 0.2, f"recall at 0.5: plain={r_plain:.2f}, class-weighted={r_w:.2f}"


def t_smote_shape():
    X_min = np.random.default_rng(1).normal(size=(10, 3))
    S = smote(X_min, k=3, n_synthetic=25, rng=np.random.default_rng(0))
    copies = any(np.allclose(s, x) for s in S for x in X_min)
    return S.shape == (25, 3) and not copies, f"shape={S.shape}, exact copies={copies}"


def t_smote_segment():
    a, b = np.array([0.0, 0.0]), np.array([4.0, 2.0])
    S = smote(np.vstack([a, b]), k=5, n_synthetic=50, rng=np.random.default_rng(0))
    on_line = np.allclose(S[:, 1], S[:, 0] / 2) and S[:, 0].min() >= 0 and S[:, 0].max() <= 4
    return on_line, "2 points, k clamped to 1: every new point lies on the segment between them"


def t_smote_neighbours():
    left = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]])
    right = left + 100.0
    S = smote(np.vstack([left, right]), k=2, n_synthetic=200, rng=np.random.default_rng(0))
    stray = int(np.sum((S[:, 0] > 1) & (S[:, 0] < 99)))
    return stray == 0, f"{stray} points created in the empty gap between clusters (k=2 must use only same-cluster neighbours)"


def t_mcc():
    ok = abs(mcc(3, 2, 1, 4) - 10 / np.sqrt(600)) < 1e-9 and mcc(5, 0, 0, 5) == 1.0 \
        and mcc(0, 5, 5, 0) == -1.0 and mcc(0, 0, 10, 990) == 0.0
    return ok, f"mcc(3,2,1,4)={mcc(3, 2, 1, 4):.4f}, always-negative={mcc(0, 0, 10, 990)}"


def t_threshold():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    p = np.array([0.95, 0.80, 0.60, 0.30, 0.70, 0.40, 0.20, 0.10, 0.05, 0.50])
    t, f = best_f1_threshold(y, p)
    return abs(t - 0.6) < 1e-9 and abs(f - 0.75) < 1e-9, f"threshold={t}, F1={f:.4f}"


def t_pipeline():
    X, y = make_data()
    X_tr, y_tr = X[:600], y[:600]
    X_val, y_val = X[600:800], y[600:800]
    X_te, y_te = X[800:], y[800:]
    w, b = train_logreg(X_tr, y_tr, compute_class_weights(y_tr))
    t, _ = best_f1_threshold(y_val, sigmoid(X_val @ w + b))
    default = mcc(*counts(y_te, (sigmoid(X_te @ w + b) >= 0.5).astype(int)))
    tuned = mcc(*counts(y_te, (sigmoid(X_te @ w + b) >= t).astype(int)))
    return tuned > default, f"test MCC: weighted@0.5={default:.3f} -> weighted@{t:.2f} (tuned on val)={tuned:.3f}"


if __name__ == "__main__":
    tests = [
        ("class weights: 0.5025 / 100, each class carries half the loss", t_weights),
        ("weighted loss raises minority recall at threshold 0.5", t_weighted_training),
        ("SMOTE: right shape, no exact copies", t_smote_shape),
        ("SMOTE: interpolates on the segment between neighbours", t_smote_segment),
        ("SMOTE: neighbours are the k NEAREST minority points", t_smote_neighbours),
        ("MCC: known value, +1, -1, and 0 for always-negative", t_mcc),
        ("best_f1_threshold picks 0.6 (F1 0.75) on last lesson's data", t_threshold),
        ("pipeline: weights + val-tuned threshold beats 0.5 on test MCC", t_pipeline),
    ]
    passed = sum(check(name, fn) for name, fn in tests)
    print(f"\n  {passed}/{len(tests)} checks passed")
