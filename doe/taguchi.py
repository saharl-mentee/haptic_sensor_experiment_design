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

THE CATCH: arrays only exist for specific level patterns, and ours
(coil x 3 x 3 x 3) never matches one exactly. Three arrays are
adapted below; which is "natural" depends on how many coils are in
factors.py:

  * L16 (16 runs, five 4-level columns) — natural for a 4-LEVEL coil.
      Coil gets a 4-level column (perfectly balanced, 4 runs each);
      each 3-level factor is squeezed into a 4-level column by the
      DUMMY-LEVEL technique (level 4 repeats level 1 -> that factor
      becomes unbalanced: 8/4/4 runs).

  * L18 (18 runs, one 2-level + seven 3-level columns) — cheap option
      for 4-6 coils. "Column merging": the 2-level and a 3-level
      column combine into one 6-level column for the coil. Coil slots
      beyond your actual coils are DUMMIES that repeat coils
      cyclically (5 coils -> one coil built 6x; 4 coils -> two coils
      built 6x while the rest get 3x).

  * L25 (25 runs, six 5-level columns) — natural for a 5-LEVEL coil.
      Coil gets a 5-level column (balanced, 5 runs each; with fewer
      coils the extra slots repeat coils cyclically). Each 3-level
      factor is dummy-leveled into a 5-level column (10/10/5 runs).

All remain valid designs — the price of each trick shows up as
imbalance and mild correlation between effect estimates, which
metrics.py quantifies (balance / VIF / correlation), so you can
judge the damage instead of guessing.
--------------------------------------------------------------------
"""

import numpy as np
import pandas as pd

from factors import FACTORS


def _cyclic_map(levels: list, n_slots: int, array_name: str,
                what: str = "a factor") -> list:
    """
    Map an array column's n_slots codes onto len(levels) real levels by
    cycling: slot i -> levels[i % k]. This is the general dummy-level
    rule and it works for ANY level count k <= n_slots:
      * k == n_slots -> every level once (perfectly balanced).
      * k <  n_slots -> the first (n_slots - k) levels repeat (built
        more often) — cycling guarantees EVERY real level still appears.
      * k >  n_slots -> impossible: the column cannot carry that many
        levels, so we refuse loudly instead of silently dropping one.
    """
    k = len(levels)
    if k > n_slots:
        raise ValueError(
            f"{array_name}: {what} has {k} levels but this array only "
            f"offers {n_slots} column-slots for it — use a larger array "
            f"(L16<L18<L25) or the D-optimal designs, which handle any "
            f"level count.")
    return [levels[i % k] for i in range(n_slots)]


# ------------------------------------------------------------------
# L18 (2^1 x 3^7): the standard published array, levels coded 0,1,2.
# Rows = 18 runs, columns = 8 factor slots.
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
    Works because 5 is PRIME: mod-5 multiplication by 1..4 is
    invertible, which is what guarantees orthogonality (checked below).
    """
    rows = []
    for i in range(5):
        for j in range(5):
            rows.append([i, j, (i + j) % 5, (i + 2 * j) % 5,
                         (i + 3 * j) % 5, (i + 4 * j) % 5])
    return np.array(rows)


# GF(4) multiplication table. The mod-5 trick above does NOT work for
# 4 levels because 4 is not prime (2*2 = 0 mod 4 — information gets
# destroyed). The fix is the finite field GF(4), where addition is
# bitwise XOR and multiplication follows this table:
_GF4_MUL = np.array([
    [0, 0, 0, 0],
    [0, 1, 2, 3],
    [0, 2, 3, 1],
    [0, 3, 1, 2],
])


def l16_array() -> np.ndarray:
    """
    L16 (4^5): rows indexed by two GF(4) digits (i, j); the five
    columns are  i,  j,  i(+)j,  i(+)2(*)j,  i(+)3(*)j,  where (+) is
    XOR and (*) is GF(4) multiplication. Same construction idea as
    L25, just with field arithmetic instead of mod-5.
    """
    rows = []
    for i in range(4):
        for j in range(4):
            rows.append([i, j,
                         i ^ j,
                         i ^ _GF4_MUL[2][j],
                         i ^ _GF4_MUL[3][j]])
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


def _build_design(columns: dict, slots: dict, array_name: str) -> pd.DataFrame:
    """
    Turn array code-columns into a design table. `columns[factor]` is
    that factor's code vector from the array; `slots[factor]` is how
    many levels that array column carries. Each factor is mapped with
    _cyclic_map, so ANY level count in factors.py works (or fails loudly
    if a factor has more levels than its column can hold).
    """
    maps = {f: _cyclic_map(FACTORS[f], slots[f], array_name,
                           "the coil" if f == "coil" else f)
            for f in columns}
    return pd.DataFrame({f: [maps[f][v] for v in codes]
                         for f, codes in columns.items()})


def design_l16() -> pd.DataFrame:
    """
    Adapt L16 (four 4-level columns) to our factors. Coil is the natural
    4-level fit; foam/thickness/metal are dummy-leveled into 4-level
    columns (with 3 levels: one level built 2x more). Any level count
    <= 4 per factor is handled by the cyclic dummy rule.
    """
    arr = l16_array()
    cols = {"coil": arr[:, 0], "foam_material": arr[:, 1],
            "foam_thickness": arr[:, 2], "metal": arr[:, 3]}
    return _build_design(cols, {f: 4 for f in cols}, "L16")


def design_l18() -> pd.DataFrame:
    """
    Adapt L18 to our factors: coil <- columns 0 & 1 merged into one
    6-level column (dummy repeats beyond the real coils); foam/thickness/
    metal <- the 3-level columns 2, 3, 4. Coil takes <= 6 levels, the
    others <= 3.
    """
    merged = L18[:, 0] * 3 + L18[:, 1]          # 6 levels: 0..5
    cols = {"coil": merged, "foam_material": L18[:, 2],
            "foam_thickness": L18[:, 3], "metal": L18[:, 4]}
    slots = {"coil": 6, "foam_material": 3, "foam_thickness": 3, "metal": 3}
    return _build_design(cols, slots, "L18")


def design_l25() -> pd.DataFrame:
    """
    Adapt L25 (six 5-level columns) to our factors. Coil is the natural
    5-level fit; foam/thickness/metal are dummy-leveled into 5-level
    columns (with 3 levels: two levels built 2x more). Any level count
    <= 5 per factor is handled by the cyclic dummy rule.
    """
    arr = l25_array()
    cols = {"coil": arr[:, 0], "foam_material": arr[:, 1],
            "foam_thickness": arr[:, 2], "metal": arr[:, 3]}
    return _build_design(cols, {f: 5 for f in cols}, "L25")


# The adapted designs, keyed by name, with their run counts — used by
# compare_designs.py to offer whatever fits the current coil count.
ADAPTED = {"Taguchi L16": design_l16,
           "Taguchi L18": design_l18,
           "Taguchi L25": design_l25}

# Sanity checks: the raw arrays must be genuinely orthogonal.
assert check_orthogonality(l16_array()), "L16 construction lost orthogonality!"
assert check_orthogonality(L18[:, 1:]), "L18 3-level columns lost orthogonality!"
assert check_orthogonality(l25_array()), "L25 construction lost orthogonality!"


if __name__ == "__main__":
    print(f"factors.py currently defines {len(FACTORS['coil'])} coils\n")
    for name, builder in ADAPTED.items():
        try:
            d = builder()
            print(f"{name} ({len(d)} runs), runs per coil: "
                  f"{d['coil'].value_counts().to_dict()}")
            print(d.to_string(), "\n")
        except ValueError as e:
            print(f"{name}: not applicable — {e}\n")
    print("Raw arrays pass the orthogonality check "
          "(every level-pair equally frequent in every column-pair).")
