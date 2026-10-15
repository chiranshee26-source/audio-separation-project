import numpy as np
from preprocessing import whiten

Xc = np.load("data/mixed/X_centered.npy")
A = np.load("data/mixed/A_true.npy")
S = np.load("data/mixed/S_true.npy")

Z, V, eigvals = whiten(Xc)

# Check 1: covariance of Z should be the identity
C_before = np.cov(Xc)
C_after = np.cov(Z)
np.set_printoptions(precision=4, suppress=True)
print("Eigenvalues:", eigvals)
print("\nCovariance BEFORE whitening:\n", C_before)
print("\nCovariance AFTER whitening:\n", C_after)
print("\nIs cov(Z) ~ identity?", np.allclose(C_after, np.eye(3), atol=1e-6))

# Check 2: V @ A (with unit-variance sources) should be ~ orthogonal
std = (S - S.mean(axis=1, keepdims=True)).std(axis=1, ddof=1)
M = V @ A @ np.diag(std)
print("\nM @ M.T (should be close to identity):\n", M @ M.T)

np.save("data/mixed/Z_white.npy", Z)
np.save("data/mixed/V_white.npy", V)