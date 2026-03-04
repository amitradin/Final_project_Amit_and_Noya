"""
Unit-style tests: matrix math (sym, ddg, norm) vs NumPy reference.
"""
from pathlib import Path

import numpy as np

from tests.conftest import (
    DATA_DIR,
    PROJECT_ROOT,
    load_points_from_file,
    parse_matrix_output,
    run_c_symnmf,
    run_python_symnmf,
)
from tests.reference import degree_matrix, normalized_similarity, similarity_matrix

try:
    import unittest
except ImportError:
    import unittest

# Output is %.4f so parsed values have limited precision; use atol to allow rounding and tiny values as 0
RTOL = 1e-3
ATOL = 5e-4
POINTS_5X2 = DATA_DIR / "points_5x2.txt"


def _ref_and_c_parsed(data_path, goal):
    """Load points, get reference matrix, run C, parse output. Raises SkipTest if C binary missing."""
    if not (PROJECT_ROOT / "symnmf").exists():
        raise unittest.SkipTest("C binary not built (run make)")
    points = load_points_from_file(data_path)
    if goal == "sym":
        ref = similarity_matrix(points)
    elif goal == "ddg":
        ref = degree_matrix(points)
    else:
        ref = normalized_similarity(points)
    stdout, stderr, rc = run_c_symnmf(goal, data_path)
    if rc != 0:
        raise AssertionError(f"C symnmf {goal} failed: {stderr!r}")
    parsed = parse_matrix_output(stdout)
    return ref, parsed


def _ref_and_python_parsed(data_path, goal, k=2):
    """Load points, get reference, run Python CLI, parse output."""
    points = load_points_from_file(data_path)
    if goal == "sym":
        ref = similarity_matrix(points)
    elif goal == "ddg":
        ref = degree_matrix(points)
    else:
        ref = normalized_similarity(points)
    stdout, stderr, rc = run_python_symnmf(k, goal, data_path)
    if rc != 0:
        raise AssertionError(f"Python symnmf {goal} failed: {stderr!r}")
    parsed = parse_matrix_output(stdout)
    return ref, parsed


class TestMatrixMath(unittest.TestCase):
    def test_c_output_vs_reference_sym(self):
        ref, parsed = _ref_and_c_parsed(POINTS_5X2, "sym")
        self.assertEqual(parsed.shape, ref.shape)
        np.testing.assert_allclose(parsed, ref, rtol=RTOL, atol=ATOL)

    def test_c_output_vs_reference_ddg(self):
        ref, parsed = _ref_and_c_parsed(POINTS_5X2, "ddg")
        self.assertEqual(parsed.shape, ref.shape)
        np.testing.assert_allclose(parsed, ref, rtol=RTOL, atol=ATOL)

    def test_c_output_vs_reference_norm(self):
        ref, parsed = _ref_and_c_parsed(POINTS_5X2, "norm")
        self.assertEqual(parsed.shape, ref.shape)
        np.testing.assert_allclose(parsed, ref, rtol=RTOL, atol=ATOL)

    def test_python_output_vs_reference_sym(self):
        ref, parsed = _ref_and_python_parsed(POINTS_5X2, "sym")
        self.assertEqual(parsed.shape, ref.shape)
        np.testing.assert_allclose(parsed, ref, rtol=RTOL, atol=ATOL)

    def test_python_output_vs_reference_ddg(self):
        ref, parsed = _ref_and_python_parsed(POINTS_5X2, "ddg")
        self.assertEqual(parsed.shape, ref.shape)
        np.testing.assert_allclose(parsed, ref, rtol=RTOL, atol=ATOL)

    def test_python_output_vs_reference_norm(self):
        ref, parsed = _ref_and_python_parsed(POINTS_5X2, "norm")
        self.assertEqual(parsed.shape, ref.shape)
        np.testing.assert_allclose(parsed, ref, rtol=RTOL, atol=ATOL)

    def test_sym_symmetry(self):
        _, parsed = _ref_and_c_parsed(POINTS_5X2, "sym")
        np.testing.assert_allclose(parsed, parsed.T, rtol=0, atol=1e-10)

    def test_sym_diagonal_zero(self):
        _, parsed = _ref_and_c_parsed(POINTS_5X2, "sym")
        np.testing.assert_allclose(np.diag(parsed), 0.0, rtol=0, atol=1e-10)

    def test_ddg_diagonal(self):
        _, parsed = _ref_and_c_parsed(POINTS_5X2, "ddg")
        n = parsed.shape[0]
        for i in range(n):
            for j in range(n):
                if i != j:
                    self.assertLess(abs(parsed[i, j]), 1e-10)
        self.assertTrue(np.all(np.diag(parsed) >= -1e-10))

    def test_norm_symmetry(self):
        _, parsed = _ref_and_c_parsed(POINTS_5X2, "norm")
        np.testing.assert_allclose(parsed, parsed.T, rtol=0, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
