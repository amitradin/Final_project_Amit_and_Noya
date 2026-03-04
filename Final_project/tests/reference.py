"""
NumPy reference implementations for similarity matrix A, degree matrix D,
and normalized similarity W = D^{-1/2} A D^{-1/2}. Used as ground truth for tests.
"""
import numpy as np


def similarity_matrix(points):
    """
    A_ij = exp(-||x_i - x_j||_2^2 / 2) for i != j, A_ii = 0.
    points: (n, d) array.
    """
    points = np.asarray(points, dtype=np.float32)
    n = points.shape[0]
    A = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            d_sq = np.sum((points[i] - points[j]) ** 2)
            A[i, j] = np.exp(-d_sq / 2.0)
            A[j, i] = A[i, j]
    return A


def degree_matrix(points):
    """D is diagonal with d_i = sum_j A_ij."""
    A = similarity_matrix(points)
    d = np.sum(A, axis=1)
    return np.diag(d)


def normalized_similarity(points):
    """W = D^{-1/2} A D^{-1/2}. Use 1.0 where degree is 0 to avoid division by zero."""
    A = similarity_matrix(points)
    d = np.sum(A, axis=1)
    sqrt_d = np.where(d > 0, np.sqrt(d), 1.0)
    # W[i,j] = A[i,j] / (sqrt_d[i] * sqrt_d[j])
    W = A / (sqrt_d[:, np.newaxis] * sqrt_d[np.newaxis, :])
    return W
