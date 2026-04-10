"""Shared helpers for symnmf and analysis scripts."""
import numpy as np


def load_points(file_name):
    """This functions is for loading the points via a file""" 
    points = []
    with open(file_name, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            coords = [float(x) for x in line.split(',')]
            points.append(coords)
    return np.array(points, dtype=np.float64)


def run_symnmf(points, k, epsilon=1e-4, max_iter=300):
    """
    This function is for running the SymNMF algorithm: compute W from points, init H, then iterative update.
    it returns the final H as (n, k) numpy array, or None on failure.
    """
    import _symnmf as symnmf_mod
    points_list = points.tolist()
    W = symnmf_mod.norm(points_list)
    if W is None:
        return None
    W_arr = np.array(W, dtype=np.float64)
    m = float(np.mean(W_arr))
    n = points.shape[0]
    H_init = np.random.uniform(0, 2 * np.sqrt(m / k), (n, k)).astype(np.float64)
    H_list = H_init.tolist()
    H_final = symnmf_mod.symnmf(H_list, W, epsilon, max_iter)
    if H_final is None:
        return None
    return np.array(H_final, dtype=np.float64)
