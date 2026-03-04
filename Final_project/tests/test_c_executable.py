"""
C executable integration tests: ./symnmf sym/ddg/norm file.
"""
import re
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

FLOAT_LINE_RE = re.compile(r"^-?\d+\.\d{4}(,-?\d+\.\d{4})*$")
POINTS_5X2 = DATA_DIR / "points_5x2.txt"


class TestCExecutable(unittest.TestCase):
    def test_c_sym_output_shape(self):
        if not (PROJECT_ROOT / "symnmf").exists():
            self.skipTest("C binary not built (run make)")
        stdout, stderr, rc = run_c_symnmf("sym", POINTS_5X2)
        self.assertEqual(rc, 0, stderr)
        mat = parse_matrix_output(stdout)
        self.assertEqual(mat.shape, (5, 5))

    def test_c_ddg_output_shape(self):
        if not (PROJECT_ROOT / "symnmf").exists():
            self.skipTest("C binary not built (run make)")
        stdout, stderr, rc = run_c_symnmf("ddg", POINTS_5X2)
        self.assertEqual(rc, 0, stderr)
        mat = parse_matrix_output(stdout)
        self.assertEqual(mat.shape, (5, 5))

    def test_c_norm_output_shape(self):
        if not (PROJECT_ROOT / "symnmf").exists():
            self.skipTest("C binary not built (run make)")
        stdout, stderr, rc = run_c_symnmf("norm", POINTS_5X2)
        self.assertEqual(rc, 0, stderr)
        mat = parse_matrix_output(stdout)
        self.assertEqual(mat.shape, (5, 5))

    def test_c_sym_equals_python_sym(self):
        if not (PROJECT_ROOT / "symnmf").exists():
            self.skipTest("C binary not built (run make)")
        stdout_c, _, rc_c = run_c_symnmf("sym", POINTS_5X2)
        self.assertEqual(rc_c, 0)
        stdout_py, _, rc_py = run_python_symnmf(2, "sym", POINTS_5X2)
        self.assertEqual(rc_py, 0)
        mat_c = parse_matrix_output(stdout_c)
        mat_py = parse_matrix_output(stdout_py)
        np.testing.assert_allclose(mat_c, mat_py, rtol=1e-5, atol=1e-6)

    def test_c_ddg_equals_python(self):
        if not (PROJECT_ROOT / "symnmf").exists():
            self.skipTest("C binary not built (run make)")
        stdout_c, _, rc_c = run_c_symnmf("ddg", POINTS_5X2)
        self.assertEqual(rc_c, 0)
        stdout_py, _, rc_py = run_python_symnmf(2, "ddg", POINTS_5X2)
        self.assertEqual(rc_py, 0)
        mat_c = parse_matrix_output(stdout_c)
        mat_py = parse_matrix_output(stdout_py)
        np.testing.assert_allclose(mat_c, mat_py, rtol=1e-5, atol=1e-6)

    def test_c_norm_equals_python(self):
        if not (PROJECT_ROOT / "symnmf").exists():
            self.skipTest("C binary not built (run make)")
        stdout_c, _, rc_c = run_c_symnmf("norm", POINTS_5X2)
        self.assertEqual(rc_c, 0)
        stdout_py, _, rc_py = run_python_symnmf(2, "norm", POINTS_5X2)
        self.assertEqual(rc_py, 0)
        mat_c = parse_matrix_output(stdout_c)
        mat_py = parse_matrix_output(stdout_py)
        np.testing.assert_allclose(mat_c, mat_py, rtol=1e-5, atol=1e-6)

    def test_c_output_format(self):
        if not (PROJECT_ROOT / "symnmf").exists():
            self.skipTest("C binary not built (run make)")
        stdout, _, rc = run_c_symnmf("sym", POINTS_5X2)
        self.assertEqual(rc, 0)
        lines = [ln.strip() for ln in stdout.strip().splitlines() if ln.strip()]
        for line in lines:
            self.assertTrue(FLOAT_LINE_RE.match(line), f"Line does not match format: {line!r}")


if __name__ == "__main__":
    unittest.main()
