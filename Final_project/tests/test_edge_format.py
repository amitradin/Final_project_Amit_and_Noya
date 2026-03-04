"""
Edge cases and output format: minimal input, four-decimal format.
"""
from pathlib import Path

import numpy as np

from tests.conftest import (
    DATA_DIR,
    PROJECT_ROOT,
    parse_matrix_output,
    run_c_symnmf,
    run_python_symnmf,
)

try:
    import unittest
except ImportError:
    import unittest

POINTS_3X2 = DATA_DIR / "points_3x2.txt"


class TestEdgeFormat(unittest.TestCase):
    def test_minimal_input_sym(self):
        if not (PROJECT_ROOT / "symnmf").exists():
            self.skipTest("C binary not built")
        stdout, stderr, rc = run_c_symnmf("sym", POINTS_3X2)
        self.assertEqual(rc, 0, stderr)
        mat = parse_matrix_output(stdout)
        self.assertEqual(mat.shape, (3, 3))

    def test_minimal_input_ddg(self):
        if not (PROJECT_ROOT / "symnmf").exists():
            self.skipTest("C binary not built")
        stdout, stderr, rc = run_c_symnmf("ddg", POINTS_3X2)
        self.assertEqual(rc, 0, stderr)
        mat = parse_matrix_output(stdout)
        self.assertEqual(mat.shape, (3, 3))

    def test_minimal_input_norm(self):
        if not (PROJECT_ROOT / "symnmf").exists():
            self.skipTest("C binary not built")
        stdout, stderr, rc = run_c_symnmf("norm", POINTS_3X2)
        self.assertEqual(rc, 0, stderr)
        mat = parse_matrix_output(stdout)
        self.assertEqual(mat.shape, (3, 3))

    def test_minimal_input_symnmf(self):
        stdout, stderr, rc = run_python_symnmf(2, "symnmf", POINTS_3X2)
        self.assertEqual(rc, 0, stderr)
        H = parse_matrix_output(stdout)
        self.assertEqual(H.shape, (3, 2))
        self.assertTrue(np.all(H >= -1e-10))

    def test_output_four_decimals(self):
        if not (PROJECT_ROOT / "symnmf").exists():
            self.skipTest("C binary not built")
        stdout, _, rc = run_c_symnmf("sym", POINTS_3X2)
        self.assertEqual(rc, 0)
        lines = [ln.strip() for ln in stdout.strip().splitlines() if ln.strip()]
        for line in lines:
            for part in line.split(","):
                val = float(part)
                formatted = "%.4f" % val
                self.assertTrue(part == formatted or abs(float(part) - val) < 1e-10)


if __name__ == "__main__":
    unittest.main()
