import numpy as np
from scipy.stats import kurtosis
from fastica import fastica

Z = np.load("data/mixed/Z_white.npy")
V = np.load("data/mixed/V_white.npy")
A = np.load("data/mixed/A_true.npy")
S = np.load("data/mixed/S_true.npy")
X = np.load("data/mixed/X_mixed.npy")

W, n_iters = fastica(Z)
print("Iterations per component:", n_iters)

S_est = W @ Z

# Check 1: W V A should be a scaled permutation matrix
P = W @ V @ A
P_norm = P / np.abs(P).max(axis=1, keepdims=True)
np.set_printoptions(precision=3, suppress=True)
print("\nW @ V @ A (normalized per row):\n", P_norm)

# Check 2: CLT in action - mixtures are more Gaussian than sources
# (excess kurtosis: 0 = Gaussian, larger = more non-Gaussian)
print("\nExcess kurtosis of true sources  :", kurtosis(S, axis=1))
print("Excess kurtosis of mixtures      :", kurtosis(X, axis=1))
print("Excess kurtosis of ICA estimates :", kurtosis(S_est, axis=1))

np.save("data/mixed/W_ica.npy", W)
np.save("outputs/separated/S_est.npy", S_est)