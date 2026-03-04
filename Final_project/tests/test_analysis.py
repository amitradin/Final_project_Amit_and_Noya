"""
analysis.py tests: output format and silhouette score range.
"""
import re
from pathlib import Path

from tests.conftest import DATA_DIR, run_analysis

try:
    import unittest
except ImportError:
    import unittest

NMF_LINE_RE = re.compile(r"^nmf:\s*(-?\d+\.\d{4})\s*$")
KMEANS_LINE_RE = re.compile(r"^kmeans:\s*(-?\d+\.\d{4})\s*$")
POINTS_5X2 = DATA_DIR / "points_5x2.txt"


class TestAnalysis(unittest.TestCase):
    def test_analysis_output_format(self):
        stdout, stderr, rc = run_analysis(2, POINTS_5X2)
        self.assertEqual(rc, 0, stderr)
        lines = [ln.strip() for ln in stdout.strip().splitlines() if ln.strip()]
        self.assertEqual(len(lines), 2)
        self.assertTrue(NMF_LINE_RE.match(lines[0]), f"First line: {lines[0]!r}")
        self.assertTrue(KMEANS_LINE_RE.match(lines[1]), f"Second line: {lines[1]!r}")

    def test_analysis_silhouette_range(self):
        stdout, _, rc = run_analysis(2, POINTS_5X2)
        self.assertEqual(rc, 0)
        lines = [ln.strip() for ln in stdout.strip().splitlines() if ln.strip()]
        m1 = NMF_LINE_RE.match(lines[0])
        m2 = KMEANS_LINE_RE.match(lines[1])
        self.assertTrue(m1 and m2)
        score_nmf = float(m1.group(1))
        score_kmeans = float(m2.group(1))
        self.assertGreaterEqual(score_nmf, -1)
        self.assertLessEqual(score_nmf, 1)
        self.assertGreaterEqual(score_kmeans, -1)
        self.assertLessEqual(score_kmeans, 1)

    def test_analysis_exit_zero(self):
        stdout, stderr, rc = run_analysis(2, POINTS_5X2)
        self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
