"""SymNMF CLI: compute sym/ddg/norm matrices or run full symnmf and print result."""
import sys
import numpy as np

from utils import load_points, run_symnmf

np.random.seed(1234)


def print_matrix(M):
    """Print matrix with 4 decimals, comma-separated, one row per line."""
    n, m = M.shape
    for i in range(n):
        row = ','.join('%.4f' % M[i, j] for j in range(m))
        print(row)


def main():
    """first conditions checking"""
    if len(sys.argv) != 4:
        print("An Error Has Occurred")
        sys.exit(1)
    try:
        k = int(sys.argv[1])
    except ValueError:
        print("An Error Has Occurred")
        sys.exit(1)
    goal = sys.argv[2]
    file_name = sys.argv[3]
    if goal not in ('symnmf', 'sym', 'ddg', 'norm'):
        print("An Error Has Occurred")
        sys.exit(1)
    try:
        points = load_points(file_name)
    except (IOError, OSError, ValueError):
        print("An Error Has Occurred")
        sys.exit(1)
    if points.size == 0:
        print("An Error Has Occurred")
        sys.exit(1)
    n, d = points.shape
    # For symnmf we need 1 <= k < n
    if goal == 'symnmf' and (k >= n or k < 1):
        print("An Error Has Occurred")
        sys.exit(1)
    try:
        import _symnmf as symnmf_mod
    except ImportError:
        print("An Error Has Occurred")
        sys.exit(1)

    """now we can start the actual computation"""
    points_list = points.tolist()
    if goal == 'sym':
        result = symnmf_mod.sym(points_list)
        if result is None:
            print("An Error Has Occurred")
            sys.exit(1)
        print_matrix(np.array(result))
    elif goal == 'ddg':
        result = symnmf_mod.ddg(points_list)
        if result is None:
            print("An Error Has Occurred")
            sys.exit(1)
        print_matrix(np.array(result))
    elif goal == 'norm':
        result = symnmf_mod.norm(points_list)
        if result is None:
            print("An Error Has Occurred")
            sys.exit(1)
        print_matrix(np.array(result))
    else:
        """now we can run the SymNMF algorithm"""
        H_final = run_symnmf(points, k)
        if H_final is None:
            print("An Error Has Occurred")
            sys.exit(1)
        print_matrix(H_final)


if __name__ == '__main__':
    main()
