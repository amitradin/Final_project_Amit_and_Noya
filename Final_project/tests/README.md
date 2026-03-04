# SymNMF test suite

Run all tests from the **project root** after building:

1. **C executable**: `make`
2. **Python extension**: `python3 setup.py build_ext --inplace`

Then:

```bash
python3 -m unittest discover -s tests -p "test_*.py" -v
```

Tests cover: matrix math (sym/ddg/norm) vs NumPy reference, C executable and Python CLI output, extension API, analysis.py output format and silhouette range, error handling, and edge/format checks.

## Memory check (Valgrind)

From the project root, with [Valgrind](https://valgrind.org/) installed:

```bash
make valgrind
```

This runs the C executable under Valgrind for goals `sym`, `ddg`, and `norm` on `tests/data/points_3x2.txt` and reports leaks or invalid access. To use another data file or Valgrind options:

```bash
make valgrind VALGRIND_DATA=tests/data/points_5x2.txt
make valgrind VALGRIND_OPTS="--leak-check=full --track-origins=yes"
```
