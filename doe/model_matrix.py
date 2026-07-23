"""
model_matrix.py — Turn a design table into the matrix ANOVA actually fits,
and do the "degrees of freedom" bookkeeping.

--------------------------------------------------------------------
THE KEY IDEA (plain language)

After the experiment, each run gives one response number (e.g. its
sensitivity). ANOVA fits a model:

    response = overall_mean
             + (effect of this run's coil)
             + (effect of its foam material)
             + (effect of its thickness)
             + (effect of its metal)
             [+ interaction terms]
             + noise

Every quantity the model must ESTIMATE costs one "degree of freedom"
(df) — think of df as the information budget:

  * A categorical factor with k levels costs k-1 df.
    (Why k-1 and not k? Because effects are measured RELATIVE to a
    reference level. With 3 foams, once you know how FoamB and FoamC
    differ from FoamA, you know everything.)

  * An interaction between a (k-level) and an (m-level) factor costs
    (k-1)*(m-1) df.

  * Whatever is left over —  N_runs - model_df — is the ERROR df:
    the information used to estimate the NOISE level. Every p-value
    is computed against this noise estimate, so with too few error
    df (< ~8) the whole analysis becomes unreliable.

Technically: each run becomes a row of the "model matrix" X. A
categorical factor with k levels becomes k-1 indicator (0/1)
columns ("dummy coding"). Fitting = ordinary least squares on X.
--------------------------------------------------------------------
"""

import numpy as np
import pandas as pd

from factors import FACTORS, INTERACTIONS


def _effect_col(design: pd.DataFrame, factor: str, level: str, ref: str) -> np.ndarray:
    """
    One EFFECT-CODED (a.k.a. sum-to-zero / deviation) column for a
    categorical factor:
        +1  where the run is at `level`
        -1  where the run is at the reference level `ref` (factor's first)
         0  otherwise
    Unlike 0/1 dummy coding, these columns sum to zero, so main effects
    and their products (interactions) are nearly uncorrelated — which is
    why the interaction-model VIFs report near 1-2 instead of ~10. The
    design, its D-efficiency and its power are UNCHANGED (this is just a
    different, invertible way to write the same model); only the VIF /
    correlation numbers become honest instead of inflated by a coding
    artifact.
    """
    col = np.zeros(len(design))
    col[(design[factor] == level).to_numpy()] = 1.0
    col[(design[factor] == ref).to_numpy()] = -1.0
    return col


def build_model_matrix(design: pd.DataFrame, include_interaction: bool = False):
    """
    design: DataFrame with one row per run and one column per factor
            (values are level names, e.g. 'C3', 'FoamB', ...).
    Returns (X, column_names): X is the (n_runs x n_params) matrix.

    Categorical factors use EFFECT coding (+1/-1/0, see _effect_col), so
    an interaction column is the PRODUCT of its two parents' effect
    columns.
    """
    n = len(design)
    columns = [np.ones(n)]           # the intercept = overall mean
    names = ["intercept"]

    # Main effects: k-1 effect-coded columns per factor (first level = reference)
    for factor, levels in FACTORS.items():
        ref = levels[0]
        for level in levels[1:]:
            columns.append(_effect_col(design, factor, level, ref))
            names.append(f"{factor}[{level}]")

    # Optional interactions: the PRODUCT of the two factors' effect-coded
    # columns. include_interaction=True adds EVERY pair in INTERACTIONS.
    if include_interaction:
        for fa, fb in INTERACTIONS:
            refa, refb = FACTORS[fa][0], FACTORS[fb][0]
            for la in FACTORS[fa][1:]:
                col_a = _effect_col(design, fa, la, refa)
                for lb in FACTORS[fb][1:]:
                    col_b = _effect_col(design, fb, lb, refb)
                    columns.append(col_a * col_b)
                    names.append(f"{fa}[{la}]:{fb}[{lb}]")

    return np.column_stack(columns), names


def model_df(include_interaction: bool = False) -> int:
    """How many parameters (df) the model spends, before error df."""
    df = 1  # intercept (the overall mean)
    for levels in FACTORS.values():
        df += len(levels) - 1
    if include_interaction:
        for fa, fb in INTERACTIONS:
            df += (len(FACTORS[fa]) - 1) * (len(FACTORS[fb]) - 1)
    return df


def print_df_accounting(n_runs: int, include_interaction: bool = False):
    """Print the information-budget bookkeeping for a given run count."""
    print(f"Degrees-of-freedom budget for N = {n_runs} runs")
    print(f"  {'overall mean (intercept)':38s} costs  1 df")
    total = 1
    for factor, levels in FACTORS.items():
        k = len(levels)
        print(f"  {factor + f' ({k} levels)':38s} costs {k-1:2d} df")
        total += k - 1
    if include_interaction:
        for fa, fb in INTERACTIONS:
            cost = (len(FACTORS[fa]) - 1) * (len(FACTORS[fb]) - 1)
            print(f"  {fa + ' x ' + fb:38s} costs {cost:2d} df")
            total += cost
    err = n_runs - total
    print(f"  {'-'*52}")
    print(f"  model total = {total} df  ->  error df = {n_runs} - {total} = {err}")
    if err < 0:
        print("  *** NOT ESTIMABLE: more parameters than runs! ***")
    elif err < 8:
        print("  (warning: fewer than ~8 error df -> noise estimate is shaky,")
        print("   p-values unreliable, one failed build could sink the analysis)")
    else:
        print("  (>= 8 error df: a solid basis for the noise estimate)")


if __name__ == "__main__":
    from factors import INTERACTIONS
    print("Main-effects model:")
    print_df_accounting(20, include_interaction=False)
    pairs = ", ".join(f"{a} x {b}" for a, b in INTERACTIONS)
    print(f"\nMain effects + interactions ({pairs}):")
    print_df_accounting(25, include_interaction=True)
