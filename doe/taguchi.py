"""
taguchi.py — Standard Taguchi orthogonal arrays, adapted to this experiment.

--------------------------------------------------------------------
WHAT IS AN ORTHOGONAL ARRAY?

A pre-computed table of runs with a beautiful property: for EVERY
pair of columns, every combination of levels appears equally often.
Consequence: when you average the results per level of one factor,
the influence of all the other factors cancels out exactly — each
factor can be judged independently ("orthogonally").

That is also exactly what ANOVA needs to separate the factors
cleanly, so Taguchi arrays are simply very balanced designs.

THE CATCH for our experiment: arrays only exist for specific level
patterns. Ours is 5 x 3 x 3 x 3, which no standard array matches
exactly. The two closest are adapted below, each with a standard
trick that costs some balance:

  * L18 (18 runs, one 2-level + seven 3-level columns)
      -> "column merging": columns 1 and 2 are combined into one
         6-level column for the coil. Since we have only 5 coils,
         the 6th slot is a DUMMY: it repeats one coil (giving that
         coil 6 runs while the others get 3).

  * L25 (25 runs, six 5-level columns)
      -> coil fits a 5-level column perfectly (fully balanced!).
         Each 3-level factor is squeezed into a 5-level column by
         the DUMMY-LEVEL technique: levels 4,5 are mapped back to
         levels 1,2. Result: within that factor, two levels get 10
         runs and one gets 5 (unbalanced but still a valid design).

Both remain usable designs — the price of the tricks shows up as
mild correlation between effect estimates, which metrics.py
quantifies (VIF / correlation), so you can judge the damage.
--------------------------------------------------------------------
"""

import numpy as np
import pandas as pd

from factors import FACTORS

# ------------------------------------------------------------------
# L18 (2^1 x 3^7): the standard published array, levels coded 0,1,2.
# Rows = 18 runs, columns = 8 factors slots.
# ------------------------------------------------------------------
L18 = np.array([
    [0,0,0,0,0,0,0,0],
    [0,0,1,1,1,1,1,1],
    [0,0,2,2,2,2,2,2],
    [0,1,0,0,1,1,2,2],
    [0,1,1,1,2,2,0,0],
    [0,1,2,2,0,0,1,1],
    [0,2,0,1,0,2,1,2],
    [0,2,1,2,1,0,2,0],
    [0,2,2,0,2,1,0,1],
    [1,0,0,2,2,1,1,0],
    [1,0,1,0,0,2,2,1],
    [1,0,2,1,1,0,0,2],
    [1,1,0,1,2,0,2,1],
    [1,1,1,2,0,1,0,2],
    [1,1,2,0,1,2,1,0],
    [1,2,0,2,1,2,0,1],
    [1,2,1,0,2,0,1,2],
    [1,2,2,1,0,1,2,0],
])


def l25_array() -> np.ndarray:
    """
    L25 (5^6), built with the standard finite-field construction:
    rows are indexed by two base-5 digits (i, j); the six columns are
    i, j, i+j, i+2j, i+3j, i+4j  (all modulo 5).
    This guarantees the orthogonality property (checked below).
    """
    rows = []
    for i in range(5):
        for j in range(5):
            rows.append([i, j, (i + j) % 5, (i + 2 * j) % 5,
                         (i + 3 * j) % 5, (i + 4 * j) % 5])
    return np.array(rows)


def check_orthogonality(array: np.ndarray) -> bool:
    """
    Verify the defining property: for every pair of columns, every
    pair of levels appears equally often.
    """
    n, m = array.shape
    for a in range(m):
        for b in range(a + 1, m):
            pairs, counts = np.unique(array[:, [a, b]], axis=0, return_counts=True)
            la, lb = len(np.unique(array[:, a])), len(np.unique(array[:, b]))
            if len(pairs) != la * lb or len(set(counts)) != 1:
                return False
    return True


def design_l18() -> pd.DataFrame:
    """
    Adapt L18 to our factors:
      coil   <- columns 0 & 1 merged into a 6-level column,
                6th level = dummy repeat of C1
      foam_material, foam_thickness, metal <- columns 2, 3, 4
    """
    merged = L18[:, 0] * 3 + L18[:, 1]          # 6 levels: 0..5
    coil_map = FACTORS["coil"] + [FACTORS["coil"][0]]   # level 5 -> repeat C1
    return pd.DataFrame({
        "coil":           [coil_map[v] for v in merged],
        "foam_material":  [FACTORS["foam_material"][v] for v in L18[:, 2]],
        "foam_thickness": [FACTORS["foam_thickness"][v] for v in L18[:, 3]],
        "metal":          [FACTORS["metal"][v] for v in L18[:, 4]],
    })


def design_l25() -> pd.DataFrame:
    """
    Adapt L25 to our factors:
      coil <- column 0 (native 5 levels, perfectly balanced: 5 runs each)
      each 3-level factor <- its own 5-level column with the
      dummy-level map [0,1,2,0,1]: levels 0,1 appear 10x, level 2 5x.
    """
    arr = l25_array()
    dummy = [0, 1, 2, 0, 1]   # 5-level code -> 3-level index
    return pd.DataFrame({
        "coil":           [FACTORS["coil"][v] for v in arr[:, 0]],
        "foam_material":  [FACTORS["foam_material"][dummy[v]] for v in arr[:, 1]],
        "foam_thickness": [FACTORS["foam_thickness"][dummy[v]] for v in arr[:, 2]],
        "metal":          [FACTORS["metal"][dummy[v]] for v in arr[:, 3]],
    })


# Sanity checks: the raw arrays must be genuinely orthogonal.
assert check_orthogonality(L18[:, 1:]), "L18 3-level columns lost orthogonality!"
assert check_orthogonality(l25_array()), "L25 construction lost orthogonality!"


if __name__ == "__main__":
    print("L18 adapted design (18 runs):")
    print(design_l18().to_string())
    print("\nL25 adapted design (25 runs):")
    print(design_l25().to_string())
    print("\nRaw arrays pass the orthogonality check "
          "(every level-pair equally frequent in every column-pair).")
