"""
optimal.py — D-optimal design selection (the algorithm behind R's
AlgDesign::optFederov and Design-Expert's "optimal" designs).

--------------------------------------------------------------------
THE PROBLEM

Out of all 135 possible factor combinations, choose N (say 20) to
actually build — such that the ANOVA can separate all the effects as
cleanly and precisely as possible.

THE IDEA

For a candidate set of runs, form the model matrix X (model_matrix.py).
Statistics tells us the uncertainty of ALL estimated effects together
is proportional to 1 / det(X'X):

  * det(X'X) LARGE  -> effects estimated precisely and independently
  * det(X'X) small  -> some effects entangled (correlated columns)
  * det(X'X) = 0    -> some effect literally cannot be estimated
                       (perfect confounding)

So we search for the N runs that MAXIMIZE det(X'X). That is all
"D-optimal" means (D = determinant).

THE ALGORITHM (Fedorov exchange — simple but effective)

  1. Start from N runs chosen at random.
  2. For every run currently in the design, try replacing it with
     every candidate not in the design; keep the swap that increases
     det(X'X) the most.
  3. Repeat until no swap helps.
  4. Because this can get stuck in a local optimum, restart from
     several random starts and keep the overall best.

We work with log(det) for numerical stability.
--------------------------------------------------------------------
"""

import numpy as np
import pandas as pd
from itertools import product

from factors import FACTORS
from model_matrix import build_model_matrix


def candidate_set() -> pd.DataFrame:
    """All possible factor combinations (the full factorial, 135 rows)."""
    names = list(FACTORS)
    rows = list(product(*FACTORS.values()))
    return pd.DataFrame(rows, columns=names)


def _logdet(X: np.ndarray) -> float:
    """log(det(X'X)); -inf if singular (some effect not estimable)."""
    sign, val = np.linalg.slogdet(X.T @ X)
    return val if sign > 0 else -np.inf


def d_optimal(n_runs: int, include_interaction: bool = False,
              n_starts: int = 30, seed: int = 42) -> pd.DataFrame:
    """
    Pick n_runs DISTINCT combinations from the candidate set that
    maximize det(X'X) for the requested model.
    """
    cand = candidate_set()
    Xall, _ = build_model_matrix(cand, include_interaction)
    n_cand, rng = len(cand), np.random.default_rng(seed)

    best_idx, best_val = None, -np.inf
    for _ in range(n_starts):
        # 1. random starting design
        idx = list(rng.choice(n_cand, size=n_runs, replace=False))
        val = _logdet(Xall[idx])

        # 2-3. exchange until no swap improves det
        improved = True
        while improved:
            improved = False
            outside = [i for i in range(n_cand) if i not in set(idx)]
            for pos in range(n_runs):
                current, best_swap, best_swap_val = idx[pos], None, val
                for j in outside:
                    idx[pos] = j
                    v = _logdet(Xall[idx])
                    if v > best_swap_val + 1e-10:
                        best_swap, best_swap_val = j, v
                idx[pos] = current
                if best_swap is not None:
                    outside.remove(best_swap)
                    outside.append(idx[pos])
                    idx[pos], val = best_swap, best_swap_val
                    improved = True

        # 4. keep the best restart. The `best_idx is None` clause makes
        #    the FIRST start always stick, so that even when every design
        #    is singular (val = -inf, e.g. more model params than runs)
        #    we still return a design instead of crashing on None. The
        #    scorecard then flags it as not-estimable.
        if best_idx is None or val > best_val:
            best_val, best_idx = val, list(idx)

    return cand.iloc[sorted(best_idx)].reset_index(drop=True)


def add_replicates(design: pd.DataFrame, n_replicates: int,
                   include_interaction: bool = False) -> pd.DataFrame:
    """
    Add replicate runs = EXACT DUPLICATES of already-chosen runs.

    Why waste runs repeating a build? Because two sensors built to
    the same spec will NOT give identical results (glue, foam cutting,
    coil tolerances...). Replicates measure that build-to-build noise
    directly ("pure error"), with no modeling assumption — the 3
    measurement cycles you do per sensor only capture measurement
    noise, not fabrication scatter.

    Which runs to duplicate? Greedily, the ones that keep det(X'X)
    highest — i.e. the most informative points.
    """
    out = design.copy()
    for _ in range(n_replicates):
        best_row, best_val = None, -np.inf
        for i in range(len(design)):          # only duplicate original points
            Xrow, _ = build_model_matrix(
                pd.concat([out, design.iloc[[i]]], ignore_index=True),
                include_interaction)
            v = _logdet(Xrow)
            if v > best_val:
                best_val, best_row = v, i
        out = pd.concat([out, design.iloc[[best_row]]], ignore_index=True)
    out["is_replicate"] = out.duplicated(subset=list(FACTORS), keep="first")
    return out


def augment_design(base: pd.DataFrame, n_add: int,
                   include_interaction: bool = False,
                   allow_replicates: bool = False,
                   n_starts: int = 30, seed: int = 42) -> pd.DataFrame:
    """
    Grow an EXISTING design by n_add runs, keeping every base run.

    This is "sequential" / "augmented" D-optimal design: you have
    already committed to (or already built) the `base` runs, and now
    want the best runs to ADD — the design that maximizes det(X'X) for
    the combined set, subject to the base being fixed.

    Why not just re-run d_optimal at the bigger N? Because that gives a
    DIFFERENT set of runs (D-optimal designs are not nested), so your
    already-built sensors might not be in it. Augmenting guarantees the
    result CONTAINS the base -> you can build in stages, stop early if
    the answer is already clear, or extend later without waste. The
    price is a sliver of D-efficiency vs the from-scratch design.

    allow_replicates=False -> the added runs are new, distinct combos
    (also distinct from the base). True -> a base combo may be repeated
    if that is what best raises det(X'X).

    The math trick: the base contributes a fixed M0 = Xbase' Xbase to
    the information matrix, so the combined X'X is just M0 plus the
    added rows' contribution — we never re-touch the base while search.
    """
    cand = candidate_set()
    Xcand, _ = build_model_matrix(cand, include_interaction)
    Xbase, _ = build_model_matrix(base, include_interaction)
    M0 = Xbase.T @ Xbase
    rng = np.random.default_rng(seed)

    def combined_logdet(add_idx):
        Xa = Xcand[add_idx]
        sign, val = np.linalg.slogdet(M0 + Xa.T @ Xa)
        return val if sign > 0 else -np.inf

    # candidate pool for the ADDED runs
    if allow_replicates:
        pool = list(range(len(cand)))
    else:
        base_keys = set(map(tuple, base[list(FACTORS)].to_numpy()))
        pool = [i for i, row in enumerate(cand[list(FACTORS)].to_numpy())
                if tuple(row) not in base_keys]

    best_idx, best_val = None, -np.inf
    for _ in range(n_starts):
        idx = list(rng.choice(pool, size=n_add, replace=False))
        val = combined_logdet(idx)
        improved = True
        while improved:
            improved = False
            outside = [i for i in pool if i not in set(idx)]
            for pos in range(n_add):
                current, best_swap, best_swap_val = idx[pos], None, val
                for j in outside:
                    idx[pos] = j
                    v = combined_logdet(idx)
                    if v > best_swap_val + 1e-10:
                        best_swap, best_swap_val = j, v
                idx[pos] = current
                if best_swap is not None:
                    outside.remove(best_swap)
                    outside.append(idx[pos])
                    idx[pos], val = best_swap, best_swap_val
                    improved = True
        if best_idx is None or val > best_val:
            best_val, best_idx = val, list(idx)

    added = cand.iloc[best_idx].reset_index(drop=True)
    out = pd.concat([base.reset_index(drop=True), added], ignore_index=True)
    out["is_augment"] = [False] * len(base) + [True] * n_add
    return out


def _self_test():
    """
    Cross-check on a case with a KNOWN answer: for two 3-level factors
    and N=9, the D-optimal design must be the full 3x3 factorial
    (every combination once) — perfectly balanced.
    """
    # mutate FACTORS in place so every module sees the test factors
    saved = dict(FACTORS)
    FACTORS.clear()
    FACTORS.update({"a": ["a1", "a2", "a3"], "b": ["b1", "b2", "b3"]})
    try:
        d = d_optimal(9, include_interaction=False, n_starts=10, seed=1)
        counts = d.value_counts().max()
        assert counts == 1 and len(d) == 9, "Fedorov failed to find the 3x3 factorial"
    finally:
        FACTORS.clear()
        FACTORS.update(saved)
    print("self-test OK: Fedorov recovers the 3x3 full factorial at N=9")


if __name__ == "__main__":
    _self_test()
    d = d_optimal(20)
    print("\nD-optimal 20-run design (main-effects model):")
    print(d.to_string())
    for f in FACTORS:
        print(f"{f}: level counts = {d[f].value_counts().to_dict()}")
