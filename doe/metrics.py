"""
metrics.py — A common scorecard so ANY design (Taguchi array,
D-optimal, full factorial...) can be judged on the same criteria.

--------------------------------------------------------------------
WHAT EACH METRIC MEANS

* error df       N_runs - model_df. The information left for
                 estimating noise. Rule of thumb: aim for >= 8-10.
                 Below that, p-values rest on a shaky noise estimate.

* replicates     Runs that are exact duplicates of another run.
                 They measure build-to-build noise with no model
                 assumptions ("pure error").

* balance        For each factor: how many runs each level gets
                 (min-max). Balanced designs estimate every level
                 equally well; unbalanced ones know some levels
                 better than others.

* D-efficiency   Information per run, relative to the full factorial
                 (which is the fairest, most balanced design money
                 could buy). 100% = "each of my N runs works as hard
                 as a full-factorial run". 80%+ is very good for a
                 constrained design.

* worst VIF      Variance Inflation Factor. VIF = 1 means an effect
                 is estimated as precisely as if all other factors
                 didn't exist (perfect separation). VIF = 2 means the
                 CONFOUNDING with other effects doubles the variance
                 (uncertainty^2) of that estimate — you'd need 2x the
                 runs to get the same precision. > 5 is trouble;
                 infinite = perfectly confounded, not estimable.

* max |corr|     Largest correlation between two effect columns.
                 0 = orthogonal (Taguchi's ideal); close to 1 = the
                 design can barely tell those two effects apart.
--------------------------------------------------------------------
"""

import numpy as np
import pandas as pd

from factors import FACTORS, full_factorial_size
from model_matrix import build_model_matrix, model_df


def _dispersion(X: np.ndarray) -> float:
    """Scaled information per run: det(X'X / N)^(1/p)."""
    n, p = X.shape
    sign, logdet = np.linalg.slogdet(X.T @ X / n)
    return np.exp(logdet / p) if sign > 0 else 0.0


def d_efficiency(design: pd.DataFrame, include_interaction: bool = False) -> float:
    """Design's information-per-run as % of the full factorial's."""
    from optimal import candidate_set
    X, _ = build_model_matrix(design, include_interaction)
    Xf, _ = build_model_matrix(candidate_set(), include_interaction)
    ref = _dispersion(Xf)
    return 100.0 * _dispersion(X) / ref if ref > 0 else np.nan


def vif_and_corr(design: pd.DataFrame, include_interaction: bool = False):
    """
    Worst VIF and largest |correlation| among effect columns.
    Computed on centered columns (the intercept is excluded — it is
    'confounded with the mean' by definition and that is fine).
    """
    X, names = build_model_matrix(design, include_interaction)
    Xc = X[:, 1:] - X[:, 1:].mean(axis=0)     # drop intercept, center
    # correlation matrix of the effect columns
    norms = np.linalg.norm(Xc, axis=0)
    if np.any(norms == 0):
        return np.inf, 1.0                     # a column never varies
    R = (Xc / norms).T @ (Xc / norms)
    try:
        vifs = np.diag(np.linalg.inv(R))
        worst_vif = float(np.max(vifs))
    except np.linalg.LinAlgError:
        worst_vif = np.inf
    off = R - np.eye(len(R))
    return worst_vif, float(np.max(np.abs(off)))


def count_replicates(design: pd.DataFrame) -> int:
    return int(design.duplicated(subset=list(FACTORS), keep="first").sum())


def balance_summary(design: pd.DataFrame) -> str:
    """e.g. 'coil 3-6, foam 5-8, thick 5-8, metal 6-7' (runs per level)."""
    short = {"coil": "coil", "foam_material": "foamMat",
             "foam_thickness": "thick", "metal": "metal"}
    parts = []
    for f in FACTORS:
        counts = design[f].value_counts()
        # include levels that never appear at all
        lo = min(counts.reindex(FACTORS[f]).fillna(0).astype(int))
        parts.append(f"{short.get(f, f)}:{lo}-{counts.max()}")
    return "  ".join(parts)


def scorecard(name: str, design: pd.DataFrame,
              include_interaction: bool = False) -> dict:
    """All metrics for one design, as a dict (one comparison-table row)."""
    n = len(design)
    p = model_df(include_interaction)
    err_df = n - p
    estimable = err_df >= 0 and _dispersion(
        build_model_matrix(design, include_interaction)[0]) > 0
    vif, corr = vif_and_corr(design, include_interaction) if estimable else (np.inf, 1.0)
    return {
        "design": name,
        "runs": n,
        "model": "ME+interaction" if include_interaction else "main effects",
        "model df": p,
        "error df": err_df,
        "replicates": count_replicates(design),
        "estimable": "yes" if estimable else "NO",
        "D-eff %": round(d_efficiency(design, include_interaction), 1) if estimable else 0.0,
        "worst VIF": round(vif, 2) if np.isfinite(vif) else np.inf,
        "max |corr|": round(corr, 2),
        "balance (runs/level)": balance_summary(design),
    }


if __name__ == "__main__":
    from taguchi import design_l18, design_l25
    for nm, d in [("L18", design_l18()), ("L25", design_l25())]:
        print(pd.Series(scorecard(nm, d)).to_string(), "\n")
