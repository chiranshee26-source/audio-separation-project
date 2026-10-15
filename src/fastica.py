import numpy as np


def fastica(Z, n_components=None, max_iter=500, tol=1e-6, alpha=1.0, seed=0):
    """FastICA with deflation, using g(u) = tanh(alpha * u).

    Z: whitened data, shape (n_signals, n_samples).
    Returns (W, n_iters): W has shape (n_components, n_signals), one
    unmixing direction per row; n_iters lists iterations used per component.
    """
    n, T = Z.shape
    if n_components is None:
        n_components = n
    rng = np.random.default_rng(seed)

    W = np.zeros((n_components, n))
    n_iters = []

    for i in range(n_components):
        # random unit-length starting direction
        w = rng.standard_normal(n)
        w /= np.linalg.norm(w)

        for it in range(max_iter):
            wz = w @ Z                          # current projection, shape (T,)
            g = np.tanh(alpha * wz)             # g(u)
            g_prime = alpha * (1.0 - g ** 2)    # g'(u)

            # fixed-point update: E[z g(w'z)] - E[g'(w'z)] w
            w_new = (Z * g).mean(axis=1) - g_prime.mean() * w

            # deflation: remove components along already-found directions
            w_new -= W[:i].T @ (W[:i] @ w_new)

            w_new /= np.linalg.norm(w_new)

            # converged when direction stops changing (sign-insensitive)
            converged = abs(abs(w_new @ w) - 1.0) < tol
            w = w_new
            if converged:
                break

        W[i] = w
        n_iters.append(it + 1)

    return W, n_iters