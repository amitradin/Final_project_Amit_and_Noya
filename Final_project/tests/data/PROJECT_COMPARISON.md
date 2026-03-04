# Comparison: Final_project vs second_final (on tests/data)

**Updated after Final_project was changed to use exp(-‖·‖²/2) in the similarity formula (per PDF).**

Run from **Final_project** root. second_final was built with `python3 setup.py build_ext --inplace` and run with absolute paths to these data files.

---

## 1. Similarity matrix formula (sym)

Both projects now use the **same formula** (per PDF §1.1):

- **A_ij = exp(−‖x_i − x_j‖² / 2)** for i ≠ j, A_ii = 0.

---

## 2. Outputs on test data (after formula alignment)

### points_3x2.txt (3 points, 2 dims)

| Goal     | Final_project | second_final | Match? |
|----------|----------------|--------------|--------|
| **sym**  | 0.0000,0.3679,0.0001 / 0.3679,0.0000,0.0000 / 0.0001,0.0000,0.0000 | Same | ✓ |
| **ddg**  | 0.3680,0.3679,0.0002 on diagonal | Same | ✓ |
| **norm** | 0.0000,0.9998,0.0157 / 0.9998,0.0000,0.0058 / 0.0157,0.0058,0.0000 | Same | ✓ |
| **symnmf** (H) | 0.1939,0.6799 / 0.1953,0.6794 / 0.0303,0.0117 | 0.4968,0.5029 / 0.4980,0.5016 / 0.0128,0.0196 | ✗ (different seeds) |

### points_5x2.txt

| Goal   | Final_project | second_final | Match? |
|--------|----------------|--------------|--------|
| **sym** (first 2 rows) | 0.0000,0.3679,0.0000,... / 0.3679,0.0000,... | Same | ✓ |
| **analysis** k=2 | nmf: 0.5203, kmeans: 0.5203 | nmf: 0.1448, kmeans: 0.5203 | kmeans ✓, nmf ✗ (seed) |

### points_2x1.txt (2 points, 1 dim)

- **Both:** sym = `0.0000,0.6065` / `0.6065,0.0000` (exp(-1/2) = 0.6065). **Match ✓**

### points_empty.txt

- **Both:** "An Error Has Occurred", non-zero exit. **Match ✓**

---

## 3. Summary

| Aspect | Final_project | second_final | Match? |
|--------|----------------|--------------|--------|
| **Similarity formula** | exp(-‖·‖²/2) | exp(-‖·‖²/2) | ✓ |
| **sym / ddg / norm** | Same numeric output on tested files | Same | ✓ |
| **symnmf H** | seed 1234 | seed 0 | ✗ (different H init) |
| **analysis nmf score** | Depends on H init | Depends on H init | ✗ (different seeds) |
| **analysis kmeans score** | Same on tested files | Same | ✓ |
| **Empty file** | Error | Error | ✓ |
| **Output format** | 4 decimals, comma-separated | 4 decimals, comma-separated | ✓ |

**Conclusion:** With the same similarity formula (exp(-‖·‖²/2)), **sym, ddg, and norm outputs now match** between the two projects on all tested data files. The only remaining differences are **symnmf H** and **analysis nmf score**, due to different random seeds (Final_project: 1234, second_final: 0) for H initialization.
