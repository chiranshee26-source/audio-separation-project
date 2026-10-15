import numpy as np
from audio_utils import find_source, load_source, save_wav

NAMES = ["speech", "music", "ambient"]

rng = np.random.default_rng(42)          # fixed seed -> reproducible results

# 1. Load sources into S, shape (n_sources, n_samples)
S = np.vstack([load_source(find_source(n)) for n in NAMES])
n_sources = S.shape[0]
print("S shape:", S.shape)

# 2. Random positive mixing matrix; redraw until well-conditioned
while True:
    A = rng.uniform(0.2, 1.0, size=(n_sources, n_sources))
    if np.linalg.cond(A) < 20:
        break
print("Mixing matrix A:\n", A)
print("Condition number:", np.linalg.cond(A))

# 3. Mix: each row of X is one "microphone" recording
X = A @ S

# 4. Save the answer key and the mixtures
np.save("data/mixed/S_true.npy", S)
np.save("data/mixed/A_true.npy", A)
np.save("data/mixed/X_mixed.npy", X)
for i, x in enumerate(X):
    save_wav(f"data/mixed/mix_{i+1}.wav", x)

print("Done. Listen to data/mixed/mix_*.wav")