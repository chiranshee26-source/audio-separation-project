import numpy as np


def center(X):
    """Subtract each row's mean so every signal has zero mean.

    X: array of shape (n_signals, n_samples), one signal per row.
    Returns (X_centered, mean), where mean has shape (n_signals, 1).
    """
    mean = X.mean(axis=1, keepdims=True)
    X_centered = X - mean
    return X_centered, mean

def whiten(Xc, eps=1e-10):
    """PCA-whiten centered data using covariance + eigendecomposition.

    Xc: centered array, shape (n_signals, n_samples).
    Returns (Z, V, eigvals): whitened data, whitening matrix, eigenvalues.
    """
    n, T = Xc.shape

    # 1. Covariance matrix (n x n)
    C = (Xc @ Xc.T) / (T - 1)

    # 2. Eigendecomposition (eigh is for symmetric matrices)
    eigvals, E = np.linalg.eigh(C)

    # 3. Guard against tiny/negative eigenvalues from rounding
    eigvals = np.maximum(eigvals, eps)

    # 4. Whitening matrix V = D^(-1/2) E^T
    V = np.diag(1.0 / np.sqrt(eigvals)) @ E.T

    # 5. Apply it
    Z = V @ Xc
    return Z, V, eigvals