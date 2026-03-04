"""Compare SymNMF vs K-means on a dataset and print silhouette scores."""
import sys
import numpy as np
from sklearn.metrics import silhouette_score

from utils import load_points, run_symnmf

np.random.seed(1234)

EPSILON = 1e-4
MAX_ITER = 300


def kmeans_hw1(points, k):
    """Run K-means using HW1 helpers returns cluster index per point."""
    import HW1
    points_list = [row.tolist() for row in points]
    centroids = [points_list[i][:] for i in range(k)]
    for _ in range(MAX_ITER):
        conn = HW1.calc_length_from_centroids(points_list, centroids, k)
        new_cen = HW1.calc_new_centroids(points_list, centroids, conn)
        converged = True
        for i in range(k):
            if HW1.calc_len(centroids[i], new_cen[i]) > EPSILON:
                converged = False
                break
        centroids = new_cen
        if converged:
            break
    return np.array(conn, dtype=np.int32)

def symnmf_labels(points, k):
    """Run SymNMF and return cluster label per point (argmax on H rows)."""
    H = run_symnmf(points, k, epsilon=EPSILON, max_iter=MAX_ITER)
    if H is None:
        return None
    return np.argmax(H, axis=1)


def main():
    # Args: k, file_name
    if len(sys.argv) != 3:
        print("An Error Has Occurred")
        sys.exit(1)
    try:
        k = int(sys.argv[1])
    except ValueError:
        print("An Error Has Occurred")
        sys.exit(1)
    file_name = sys.argv[2]
    try:
        points = load_points(file_name)
    except (IOError, OSError, ValueError):
        print("An Error Has Occurred")
        sys.exit(1)
    if points.size == 0:
        print("An Error Has Occurred")
        sys.exit(1)
    n = points.shape[0]
    if k >= n or k < 1:
        print("An Error Has Occurred")
        sys.exit(1)
    labels_nmf = symnmf_labels(points, k)
    if labels_nmf is None:
        print("An Error Has Occurred")
        sys.exit(1)
    labels_kmeans = kmeans_hw1(points, k)
    score_nmf = silhouette_score(points, labels_nmf)
    score_kmeans = silhouette_score(points, labels_kmeans)
    print("nmf: %.4f" % score_nmf)
    print("kmeans: %.4f" % score_kmeans)


if __name__ == '__main__':
    main()
