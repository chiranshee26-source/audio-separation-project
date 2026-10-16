import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fastica import fastica  # noqa: E402
from preprocessing import center, whiten  # noqa: E402


def make_mixture(seed, n=3, T=20000):
    rng = np.random.default_rng(seed)
    S = rng.laplace(size=(n, T))             # non-Gaussian sources
    A = rng.uniform(0.2, 1.0, (n, n))
    return S, A @ S


def test_centering_gives_zero_mean():
    _, X = make_mixture(0)
    Xc, _ = center(X + 5.0)
    assert np.allclose(Xc.mean(axis=1), 0, atol=1e-12)


def test_whitening_gives_identity_covariance():
    _, X = make_mixture(1)
    Z, _, _ = whiten(center(X)[0])
    assert np.allclose(np.cov(Z), np.eye(3), atol=1e-6)


def test_ica_recovers_sources():
    S, X = make_mixture(2)
    Z, _, _ = whiten(center(X)[0])
    W, _ = fastica(Z)
    est = W @ Z
    C = np.abs(np.corrcoef(est, S)[:3, 3:])  # rows: estimates, cols: sources
    assert (C.max(axis=1) > 0.99).all()      # each estimate matches a source
    assert len(set(C.argmax(axis=1))) == 3   # ...and each source is used once