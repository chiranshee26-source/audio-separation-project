from preprocessing import center, whiten
from fastica import fastica


def separate(X, seed=0):
    """Full pipeline: center -> PCA whiten -> FastICA.

    X: mixed signals, shape (n_signals, n_samples).
    Returns a dict with the intermediate results and the estimated sources.
    """
    Xc, _ = center(X)
    Z, V, eigvals = whiten(Xc)
    W, n_iters = fastica(Z, seed=seed)
    return {"Xc": Xc, "Z": Z, "V": V, "eigvals": eigvals,
            "W": W, "n_iters": n_iters, "S_est": W @ Z}