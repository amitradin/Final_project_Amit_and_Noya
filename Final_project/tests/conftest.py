"""
Shared helpers for SymNMF test suite.
Assumes C binary (./symnmf) and extension (_symnmf) are built from project root.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np

# Project root: parent of directory containing conftest.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(__file__).resolve().parent / "data"


def load_points_from_file(path):
    """Load points from file (one point per line, comma-separated). Same format as symnmf.py."""
    path = Path(path)
    points = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            coords = [float(x) for x in line.split(",")]
            points.append(coords)
    return np.array(points, dtype=np.float64)


def parse_matrix_output(stdout):
    """Parse matrix from stdout: lines of comma-separated floats. Returns 2D numpy array."""
    lines = [ln.strip() for ln in stdout.strip().splitlines() if ln.strip()]
    if not lines:
        raise ValueError("Empty output")
    rows = []
    for line in lines:
        parts = line.split(",")
        rows.append([float(x) for x in parts])
    n = len(rows)
    m = len(rows[0]) if rows else 0
    for row in rows:
        if len(row) != m:
            raise ValueError("Inconsistent column count")
    return np.array(rows, dtype=np.float64)


def run_c_symnmf(goal, filepath, cwd=None):
    """Run ./symnmf goal filepath. Returns (stdout, stderr, returncode)."""
    cwd = cwd or PROJECT_ROOT
    exe = cwd / "symnmf"
    filepath = Path(filepath)
    if not filepath.is_absolute():
        filepath = PROJECT_ROOT / filepath
    result = subprocess.run(
        [str(exe), goal, str(filepath)],
        cwd=str(cwd),
        capture_output=True,
        text=True,
    )
    return result.stdout, result.stderr, result.returncode


def run_python_symnmf(k, goal, filepath, cwd=None):
    """Run python3 symnmf.py k goal filepath. Returns (stdout, stderr, returncode)."""
    cwd = cwd or PROJECT_ROOT
    filepath = Path(filepath)
    if not filepath.is_absolute():
        filepath = PROJECT_ROOT / filepath
    script = cwd / "symnmf.py"
    result = subprocess.run(
        [sys.executable, str(script), str(k), goal, str(filepath)],
        cwd=str(cwd),
        capture_output=True,
        text=True,
    )
    return result.stdout, result.stderr, result.returncode


def run_analysis(k, filepath, cwd=None):
    """Run python3 analysis.py k filepath. Returns (stdout, stderr, returncode)."""
    cwd = cwd or PROJECT_ROOT
    filepath = Path(filepath)
    if not filepath.is_absolute():
        filepath = PROJECT_ROOT / filepath
    script = cwd / "analysis.py"
    result = subprocess.run(
        [sys.executable, str(script), str(k), str(filepath)],
        cwd=str(cwd),
        capture_output=True,
        text=True,
    )
    return result.stdout, result.stderr, result.returncode
