"""
Python CLI (symnmf.py) and C extension (_symnmf) tests.
"""
from pathlib import Path

import numpy as np

from tests.conftest import (
    DATA_DIR,
    PROJECT_ROOT,
    load_points_from_file,
    parse_matrix_output,
    run_python_symnmf,
)
from tests.reference import degree_matrix, normalized_similarity, similarity_matrix

try:
    import unittest
except ImportError:
    import unittest

RTOL = 1e-4
ATOL = 1e-5
POINTS_5X2 = DATA_DIR / "points_5x2.txt"
POINTS_3X2 = DATA_DIR / "points_3x2.txt"


class TestPythonCLIExtension(unittest.TestCase):
    def test_python_symnmf_goal(self):
        stdout, stderr, rc = run_python_symnmf(2, "symnmf", POINTS_5X2)
        self.assertEqual(rc, 0, stderr)
        H = parse_matrix_output(stdout)
        self.assertEqual(H.shape, (5, 2))
        self.assertTrue(np.all(H >= -1e-10))

    def test_symnmf_cluster_labels(self):
        stdout, _, rc = run_python_symnmf(2, "symnmf", POINTS_5X2)
        self.assertEqual(rc, 0)
        H = parse_matrix_output(stdout)
        labels = np.argmax(H, axis=1)
        self.assertTrue(np.all((labels >= 0) & (labels < 2)))

    def test_extension_sym(self):
        try:
            import _symnmf as ext
        except ImportError:
            self.skipTest("_symnmf extension not built (run setup.py build_ext --inplace)")
        points = load_points_from_file(POINTS_5X2)
        ref = similarity_matrix(points)
        result = ext.sym(points.tolist())
        self.assertIsNotNone(result)
        arr = np.array(result, dtype=np.float64)
        self.assertEqual(arr.shape, ref.shape)
        np.testing.assert_allclose(arr, ref, rtol=RTOL, atol=ATOL)

    def test_extension_ddg(self):
        try:
            import _symnmf as ext
        except ImportError:
            self.skipTest("_symnmf extension not built (run setup.py build_ext --inplace)")
        points = load_points_from_file(POINTS_5X2)
        ref = degree_matrix(points)
        result = ext.ddg(points.tolist())
        self.assertIsNotNone(result)
        arr = np.array(result, dtype=np.float64)
        self.assertEqual(arr.shape, ref.shape)
        np.testing.assert_allclose(arr, ref, rtol=RTOL, atol=ATOL)

    def test_extension_norm(self):
        try:
            import _symnmf as ext
        except ImportError:
            self.skipTest("_symnmf extension not built (run setup.py build_ext --inplace)")
        points = load_points_from_file(POINTS_5X2)
        ref = normalized_similarity(points)
        result = ext.norm(points.tolist())
        self.assertIsNotNone(result)
        arr = np.array(result, dtype=np.float64)
        self.assertEqual(arr.shape, ref.shape)
        np.testing.assert_allclose(arr, ref, rtol=RTOL, atol=ATOL)

    def test_extension_symnmf_returns_h(self):
        try:
            import _symnmf as ext
        except ImportError:
            self.skipTest("_symnmf extension not built (run setup.py build_ext --inplace)")
        points = load_points_from_file(POINTS_5X2)
        points_list = points.tolist()
        W = ext.norm(points_list)
        self.assertIsNotNone(W)
        W_arr = np.array(W, dtype=np.float64)
        n, k = points.shape[0], 2
        m = float(np.mean(W_arr))
        np.random.seed(1234)
        H_init = np.random.uniform(0, 2 * np.sqrt(m / k), (n, k)).astype(np.float64).tolist()
        H_final = ext.symnmf(H_init, W, 1e-4, 300)
        self.assertIsNotNone(H_final)
        H_arr = np.array(H_final, dtype=np.float64)
        self.assertEqual(H_arr.shape, (n, k))
        self.assertTrue(np.all(H_arr >= -1e-10))


if __name__ == "__main__":
    unittest.main()
