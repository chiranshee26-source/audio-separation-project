import numpy as np
from scipy.optimize import linear_sum_assignment


def _center(x):
    return x - x.mean()


def si_sdr(est, ref):
    """Scale-invariant SDR in dB between an estimate and a reference."""
    est, ref = _center(est), _center(ref)
    alpha = (est @ ref) / (ref @ ref)       # best amplitude (sign included)
    target = alpha * ref                    # the part of est that is the source
    noise = est - target                    # everything else: leakage + distortion
    return 10 * np.log10((target @ target) / (noise @ noise + 1e-20))


def match_sources(S_est, S):
    """Resolve permutation ambiguity.

    Returns order, where order[j] = index of the estimate matching true source j.
    """
    n = S.shape[0]
    corr = np.abs(np.corrcoef(S_est, S)[:n, n:])     # rows: estimates, cols: sources
    est_idx, src_idx = linear_sum_assignment(-corr)  # minimize -corr = maximize corr
    order = np.zeros(n, dtype=int)
    order[src_idx] = est_idx
    return order