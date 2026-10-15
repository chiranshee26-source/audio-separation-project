import numpy as np
from audio_utils import save_wav

S = np.load("data/mixed/S_true.npy")
S_est = np.load("outputs/separated/S_est.npy")
names = ["speech", "music", "ambient"]

# Correlation between every estimate (rows) and every true source (columns)
corr = np.corrcoef(S_est, S)[:3, 3:]
np.set_printoptions(precision=3, suppress=True)
print("Correlation matrix (rows = estimates, cols = true sources):\n", corr)

for i in range(3):
    j = np.argmax(np.abs(corr[i]))          # which true source matches best
    sign = np.sign(corr[i, j])              # fix inverted sign
    signal = sign * S_est[i]
    save_wav(f"outputs/separated/separated_{names[j]}.wav", signal)
    print(f"Estimate {i+1} -> {names[j]}  (|corr| = {abs(corr[i, j]):.4f}, sign {int(sign):+d})")

print("Done. Listen to outputs/separated/*.wav")