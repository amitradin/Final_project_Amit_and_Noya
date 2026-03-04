"""
Tests using edge-case data files: minimal n/d, 1D, high d, far/close points, decimals, large n, empty/single.
"""
from pathlib import Path

import numpy as np

from tests.conftest import (
    DATA_DIR,
    PROJECT_ROOT,
    load_points_from_file,
    parse_matrix_output,
    run_analysis,
    run_c_symnmf,
    run_python_symnmf,
)

try:
    import unittest
except ImportError:
    import unittest

POINTS_2X1 = DATA_DIR / "points_2x1.txt"
POINTS_4X1 = DATA_DIR / "points_4x1.txt"
POINTS_4X5 = DATA_DIR / "points_4x5.txt"
POINTS_FAR = DATA_DIR / "points_far.txt"
POINTS_CLOSE = DATA_DIR / "points_close.txt"
POINTS_DECIMALS = DATA_DIR / "points_decimals.txt"
POINTS_20X2 = DATA_DIR / "points_20x2.txt"
POINTS_EMPTY = DATA_DIR / "points_empty.txt"
POINTS_ONE_LINE = DATA_DIR / "points_one_line.txt"
INPUT_1 = DATA_DIR / "input_1.txt"
INPUT_2 = DATA_DIR / "input_2.txt"
INPUT_3 = DATA_DIR / "input_3.txt"
ERROR_MSG = "An Error Has Occurred"


class TestEdgeData(unittest.TestCase):
    """Tests for edge-case data files."""

    def test_points_2x1_sym_ddg_norm(self):
        """Smallest valid: 2 points 1D; sym/ddg/norm yield 2x2."""
        if not (PROJECT_ROOT / "symnmf").exists():
            self.skipTest("C binary not built")
        for goal in ["sym", "ddg", "norm"]:
            stdout, stderr, rc = run_c_symnmf(goal, POINTS_2X1)
            self.assertEqual(rc, 0, f"{goal}: {stderr}")
            mat = parse_matrix_output(stdout)
            self.assertEqual(mat.shape, (2, 2))

    def test_points_2x1_symnmf_k1(self):
        """2 points, k=1: symnmf runs and returns 2x1 H."""
        stdout, stderr, rc = run_python_symnmf(1, "symnmf", POINTS_2X1)
        self.assertEqual(rc, 0, stderr)
        H = parse_matrix_output(stdout)
        self.assertEqual(H.shape, (2, 1))
        self.assertTrue(np.all(H >= -1e-10))

    def test_points_4x1_sym_ddg_norm(self):
        """4 points 1D: sym/ddg/norm yield 4x4."""
        if not (PROJECT_ROOT / "symnmf").exists():
            self.skipTest("C binary not built")
        for goal in ["sym", "ddg", "norm"]:
            stdout, stderr, rc = run_c_symnmf(goal, POINTS_4X1)
            self.assertEqual(rc, 0, f"{goal}: {stderr}")
            mat = parse_matrix_output(stdout)
            self.assertEqual(mat.shape, (4, 4))

    def test_points_4x1_analysis_k2(self):
        """4 points 1D, k=2: analysis runs and silhouette in [-1,1]."""
        stdout, stderr, rc = run_analysis(2, POINTS_4X1)
        self.assertEqual(rc, 0, stderr)
        lines = [ln.strip() for ln in stdout.strip().splitlines() if ln.strip()]
        self.assertEqual(len(lines), 2)
        for line in lines:
            score = float(line.split(":")[1].strip())
            self.assertGreaterEqual(score, -1)
            self.assertLessEqual(score, 1)

    def test_points_4x5_sym_ddg_norm(self):
        """4 points 5D: sym/ddg/norm yield 4x4."""
        if not (PROJECT_ROOT / "symnmf").exists():
            self.skipTest("C binary not built")
        for goal in ["sym", "ddg", "norm"]:
            stdout, stderr, rc = run_c_symnmf(goal, POINTS_4X5)
            self.assertEqual(rc, 0, f"{goal}: {stderr}")
            mat = parse_matrix_output(stdout)
            self.assertEqual(mat.shape, (4, 4))

    def test_points_far_no_crash(self):
        """Very far points: norm and symnmf run without crash; no inf/nan."""
        if not (PROJECT_ROOT / "symnmf").exists():
            self.skipTest("C binary not built")
        stdout, stderr, rc = run_c_symnmf("norm", POINTS_FAR)
        self.assertEqual(rc, 0, stderr)
        mat = parse_matrix_output(stdout)
        self.assertTrue(np.all(np.isfinite(mat)))
        stdout_py, stderr_py, rc_py = run_python_symnmf(2, "symnmf", POINTS_FAR)
        self.assertEqual(rc_py, 0, stderr_py)
        H = parse_matrix_output(stdout_py)
        self.assertTrue(np.all(np.isfinite(H)))
        self.assertTrue(np.all(H >= -1e-10))

    def test_points_close_smoke(self):
        """All points close: sym/ddg/norm and symnmf run; W in [0,1]."""
        if not (PROJECT_ROOT / "symnmf").exists():
            self.skipTest("C binary not built")
        stdout, stderr, rc = run_c_symnmf("norm", POINTS_CLOSE)
        self.assertEqual(rc, 0, stderr)
        W = parse_matrix_output(stdout)
        self.assertTrue(np.all((W >= -1e-10) & (W <= 1.0 + 1e-10)))
        stdout_py, stderr_py, rc_py = run_python_symnmf(2, "symnmf", POINTS_CLOSE)
        self.assertEqual(rc_py, 0, stderr_py)

    def test_points_decimals_four_decimal_output(self):
        """Spec decimals file: output values re-format to same with %.4f."""
        if not (PROJECT_ROOT / "symnmf").exists():
            self.skipTest("C binary not built")
        stdout, stderr, rc = run_c_symnmf("sym", POINTS_DECIMALS)
        self.assertEqual(rc, 0, stderr)
        lines = [ln.strip() for ln in stdout.strip().splitlines() if ln.strip()]
        for line in lines:
            for part in line.split(","):
                val = float(part)
                self.assertEqual(part, "%.4f" % val)

    def test_points_20x2_smoke(self):
        """Larger n=20: sym/ddg/norm and symnmf run without crash."""
        if not (PROJECT_ROOT / "symnmf").exists():
            self.skipTest("C binary not built")
        for goal in ["sym", "ddg", "norm"]:
            stdout, stderr, rc = run_c_symnmf(goal, POINTS_20X2)
            self.assertEqual(rc, 0, f"{goal}: {stderr}")
            mat = parse_matrix_output(stdout)
            self.assertEqual(mat.shape, (20, 20))
        stdout_py, stderr_py, rc_py = run_python_symnmf(2, "symnmf", POINTS_20X2)
        self.assertEqual(rc_py, 0, stderr_py)
        H = parse_matrix_output(stdout_py)
        self.assertEqual(H.shape, (20, 2))

    def test_points_empty_c_errors(self):
        """Empty file: C prints error and non-zero exit."""
        if not (PROJECT_ROOT / "symnmf").exists():
            self.skipTest("C binary not built")
        stdout, stderr, rc = run_c_symnmf("sym", POINTS_EMPTY)
        self.assertNotEqual(rc, 0)
        self.assertIn(ERROR_MSG, stdout + stderr)

    def test_points_empty_python_errors(self):
        """Empty file: Python prints error and exit 1."""
        stdout, stderr, rc = run_python_symnmf(2, "sym", POINTS_EMPTY)
        self.assertEqual(rc, 1)
        self.assertIn(ERROR_MSG, stdout + stderr)

    def test_points_one_line_sym_ddg_norm(self):
        """Single point: sym/ddg/norm yield 1x1 (if implementation accepts n=1)."""
        if not (PROJECT_ROOT / "symnmf").exists():
            self.skipTest("C binary not built")
        stdout, stderr, rc = run_c_symnmf("sym", POINTS_ONE_LINE)
        if rc == 0:
            mat = parse_matrix_output(stdout)
            self.assertEqual(mat.shape, (1, 1))
        else:
            self.assertIn(ERROR_MSG, stdout + stderr)

    def test_points_one_line_symnmf_k1_errors(self):
        """Single point, k=1: symnmf requires k < n; expect error."""
        stdout, stderr, rc = run_python_symnmf(1, "symnmf", POINTS_ONE_LINE)
        self.assertEqual(rc, 1)
        self.assertIn(ERROR_MSG, stdout + stderr)

    def test_input_1_smoke(self):
        """User input_1.txt: C norm and Python symnmf run; correct shapes."""
        if not INPUT_1.exists():
            self.skipTest("tests/data/input_1.txt not found")
        points = load_points_from_file(INPUT_1)
        n = points.shape[0]
        k = 5
        if k >= n:
            k = max(1, n - 1)
        if not (PROJECT_ROOT / "symnmf").exists():
            self.skipTest("C binary not built")
        stdout, stderr, rc = run_c_symnmf("norm", INPUT_1)
        self.assertEqual(rc, 0, stderr)
        W = parse_matrix_output(stdout)
        self.assertEqual(W.shape, (n, n))
        stdout_py, stderr_py, rc_py = run_python_symnmf(k, "symnmf", INPUT_1)
        self.assertEqual(rc_py, 0, stderr_py)
        H = parse_matrix_output(stdout_py)
        self.assertEqual(H.shape, (n, k))
        self.assertTrue(np.all(H >= -1e-10))

    def test_input_2_smoke(self):
        """User input_2.txt: C norm and Python symnmf run; correct shapes."""
        if not INPUT_2.exists():
            self.skipTest("tests/data/input_2.txt not found")
        points = load_points_from_file(INPUT_2)
        n = points.shape[0]
        k = 5
        if k >= n:
            k = max(1, n - 1)
        if not (PROJECT_ROOT / "symnmf").exists():
            self.skipTest("C binary not built")
        stdout, stderr, rc = run_c_symnmf("norm", INPUT_2)
        self.assertEqual(rc, 0, stderr)
        W = parse_matrix_output(stdout)
        self.assertEqual(W.shape, (n, n))
        stdout_py, stderr_py, rc_py = run_python_symnmf(k, "symnmf", INPUT_2)
        self.assertEqual(rc_py, 0, stderr_py)
        H = parse_matrix_output(stdout_py)
        self.assertEqual(H.shape, (n, k))
        self.assertTrue(np.all(H >= -1e-10))

    def test_input_3_smoke(self):
        """User input_3.txt: Python symnmf runs (smoke); H shape and non-negative."""
        if not INPUT_3.exists():
            self.skipTest("tests/data/input_3.txt not found")
        points = load_points_from_file(INPUT_3)
        n = points.shape[0]
        k = 3
        if k >= n:
            k = max(1, n - 1)
        stdout_py, stderr_py, rc_py = run_python_symnmf(k, "symnmf", INPUT_3)
        self.assertEqual(rc_py, 0, stderr_py)
        H = parse_matrix_output(stdout_py)
        self.assertEqual(H.shape, (n, k))
        self.assertTrue(np.all(H >= -1e-10))


if __name__ == "__main__":
    unittest.main()
