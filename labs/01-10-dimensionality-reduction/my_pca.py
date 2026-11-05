# Bridge lesson 2: Dimensionality Reduction. Your task is to write inverse_transform.
# Run: /usr/local/bin/python3 labs/01-10-dimensionality-reduction/my_pca.py
import numpy as np


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        cov = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
        self.components = eigenvectors[:, :self.n_components].T   # shape (k, d)
        self.explained_variance_ratio_ = eigenvalues[:self.n_components] / eigenvalues.sum()
        return self

    def transform(self, X):
        return (X - self.mean) @ self.components.T                # (n, d) @ (d, k) -> (n, k)

    def inverse_transform(self, Z):
        # Z has shape (n, k): points expressed in principal-component coordinates.
        # Return shape (n, d): the same points mapped back into the original feature space.
        # TODO: write this. Think about what transform() did, and undo it in reverse order.
        return Z @ self.components + self.mean


# ---------------------------------------------------------------------------
# Self-checks. Don't edit below this line; make them pass.
# ---------------------------------------------------------------------------

def make_low_rank_data(n=300, d=20, true_rank=5, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(n, true_rank))
    mixing = rng.normal(size=(true_rank, d))
    return latent @ mixing + 10.0 + noise * rng.normal(size=(n, d))   # offset of 10 so the mean matters


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    return ok


if __name__ == "__main__":
    X = make_low_rank_data()
    results = []

    pca = PCA(n_components=5).fit(X)
    Z = pca.transform(X)
    X_hat = pca.inverse_transform(Z)
    results.append(check("output shape is (n, d)", X_hat.shape == X.shape, f"got {X_hat.shape}"))

    full = PCA(n_components=20).fit(X)
    err_full = np.mean((X - full.inverse_transform(full.transform(X))) ** 2)
    results.append(check("all components -> near-perfect reconstruction", err_full < 1e-10, f"MSE={err_full:.2e}"))

    err_5 = np.mean((X - X_hat) ** 2)
    results.append(check("k = true rank (5) -> tiny error", err_5 < 0.01, f"MSE={err_5:.4f}"))

    errors = []
    for k in [1, 2, 3, 5, 10, 20]:
        p = PCA(n_components=k).fit(X)
        errors.append(np.mean((X - p.inverse_transform(p.transform(X))) ** 2))
    print("  k:    " + "  ".join(f"{k:>8d}" for k in [1, 2, 3, 5, 10, 20]))
    print("  MSE:  " + "  ".join(f"{e:8.4f}" for e in errors))
    results.append(check("error never increases as k grows", all(a >= b - 1e-12 for a, b in zip(errors, errors[1:]))))

    results.append(check("reconstruction is in original units (mean preserved)",
                         np.allclose(X_hat.mean(axis=0), X.mean(axis=0), atol=1e-8)))

    print(f"\n  {sum(results)}/{len(results)} checks passed")
