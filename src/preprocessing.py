import numpy as np


def center(X):
    """Subtract each row's mean so every signal has zero mean.

    X: array of shape (n_signals, n_samples), one signal per row.
    Returns (X_centered, mean), where mean has shape (n_signals, 1).
    """
    mean = X.mean(axis=1, keepdims=True)
    X_centered = X - mean
    return X_centered, mean