import csv
import numpy as np
from metrics import si_sdr, match_sources

S = np.load("data/mixed/S_true.npy")
X = np.load("data/mixed/X_mixed.npy")
S_est = np.load("outputs/separated/S_est.npy")
names = ["speech", "music", "ambient"]

order = match_sources(S_est, S)
rows = []
print(f"{'source':<9}{'est#':>5}{'|corr|':>10}{'SI-SDR ICA':>12}{'SI-SDR mix':>12}{'gain':>8}")

for j, name in enumerate(names):
    est = S_est[order[j]]
    corr = abs(np.corrcoef(est, S[j])[0, 1])
    sdr_ica = si_sdr(est, S[j])
    sdr_mix = max(si_sdr(x, S[j]) for x in X)     # best raw microphone
    rows.append([name, order[j] + 1, corr, sdr_ica, sdr_mix, sdr_ica - sdr_mix])
    print(f"{name:<9}{order[j]+1:>5}{corr:>10.4f}{sdr_ica:>12.2f}{sdr_mix:>12.2f}{sdr_ica - sdr_mix:>8.2f}")

with open("outputs/results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["source", "estimate_index", "abs_corr", "si_sdr_ica_db", "si_sdr_best_mixture_db", "gain_db"])
    w.writerows(rows)
print("Saved outputs/results.csv")