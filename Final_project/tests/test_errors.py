"""
Error handling tests: wrong args, invalid goal, missing file, bad k.
"""
import subprocess
import sys
from pathlib import Path

from tests.conftest import DATA_DIR, PROJECT_ROOT, run_analysis, run_c_symnmf, run_python_symnmf

try:
    import unittest
except ImportError:
    import unittest

ERROR_MSG = "An Error Has Occurred"
POINTS_5X2 = DATA_DIR / "points_5x2.txt"


class TestErrors(unittest.TestCase):
    def test_c_wrong_args_count(self):
        if not (PROJECT_ROOT / "symnmf").exists():
            self.skipTest("C binary not built")
        result = subprocess.run(
            [str(PROJECT_ROOT / "symnmf"), "sym"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(ERROR_MSG, result.stdout + result.stderr)

    def test_c_invalid_goal(self):
        if not (PROJECT_ROOT / "symnmf").exists():
            self.skipTest("C binary not built")
        stdout, stderr, rc = run_c_symnmf("symnmf", POINTS_5X2)
        self.assertNotEqual(rc, 0)
        self.assertIn(ERROR_MSG, stdout + stderr)

    def test_c_missing_file(self):
        if not (PROJECT_ROOT / "symnmf").exists():
            self.skipTest("C binary not built")
        stdout, stderr, rc = run_c_symnmf("sym", str(PROJECT_ROOT / "nonexistent_12345.txt"))
        self.assertNotEqual(rc, 0)
        self.assertIn(ERROR_MSG, stdout + stderr)

    def test_python_wrong_args(self):
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "symnmf.py"), "2", "sym"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn(ERROR_MSG, result.stdout + result.stderr)

    def test_python_invalid_goal(self):
        stdout, stderr, rc = run_python_symnmf(2, "invalid", POINTS_5X2)
        self.assertEqual(rc, 1)
        self.assertIn(ERROR_MSG, stdout + stderr)

    def test_python_non_integer_k(self):
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "symnmf.py"), "abc", "sym", str(POINTS_5X2)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn(ERROR_MSG, result.stdout + result.stderr)

    def test_python_missing_file(self):
        stdout, stderr, rc = run_python_symnmf(2, "sym", str(PROJECT_ROOT / "nonexistent_99999.txt"))
        self.assertEqual(rc, 1)
        self.assertIn(ERROR_MSG, stdout + stderr)

    def test_python_symnmf_k_ge_n(self):
        stdout, stderr, rc = run_python_symnmf(10, "symnmf", POINTS_5X2)
        self.assertEqual(rc, 1)
        self.assertIn(ERROR_MSG, stdout + stderr)

    def test_analysis_wrong_args(self):
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "analysis.py"), "2"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn(ERROR_MSG, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
