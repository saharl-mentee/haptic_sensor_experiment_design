"""
power_sim.py — Monte-Carlo statistical power: "if a factor REALLY
matters, what is the chance my experiment will notice?"

--------------------------------------------------------------------
WHAT IS POWER?

Suppose FoamB truly gives 20% higher sensitivity than the others
(EFFECT_SIZE in factors.py), and identical builds scatter by 10%
(NOISE_CV). Run the whole experiment, do the ANOVA: will the foam
factor come out significant (p < 0.05)?

Not always! Noise can mask a real effect. POWER = the probability
of detecting it. 80% power is the conventional "well-designed
experiment" target. 50% power means a coin flip — you may spend 20
builds and *still miss a real 20% improvement*.

HOW WE ESTIMATE IT (simulation — no formulas to trust blindly):

  repeat ~400 times:
    1. invent a "true world": one level of the tested factor shifts
       the response by EFFECT_SIZE, everything else flat
    2. generate fake results for every run in the design:
       y = truth + random noise (NOISE_CV)
    3. fit the ANOVA model and F-test the factor
  power = fraction of repetitions where p < ALPHA

The F-test asks: "does the model fit get significantly worse when I
remove this factor?" — the standard ANOVA question.
--------------------------------------------------------------------
"""

import numpy as np
import pandas as pd
from scipy import stats

from factors import (FACTORS, NOISE_CV, NOISE_CV_BY_LEVEL, EFFECT_SIZE,
                     BASELINE, ALPHA)
from model_matrix import build_model_matrix


def _fit_rss(X: np.ndarray, y: np.ndarray) -> float:
    """Ordinary least squares -> residual sum of squares."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ beta
    return float(r @ r)


def run_noise_sigma(design: pd.DataFrame) -> np.ndarray:
    """
    Per-run response-noise SD (absolute, in BASELINE units), one value
    per run. Combines the baseline NOISE_CV with any level-dependent
    extra noise from factors.NOISE_CV_BY_LEVEL, in quadrature:

        sigma_run = BASELINE * sqrt( NOISE_CV^2 + sum_f extra(level_f)^2 )

    An empty NOISE_CV_BY_LEVEL -> constant NOISE_CV (original behaviour).
    Only differences BETWEEN levels of the same factor make the noise
    vary from run to run; equal extra-CVs across a factor's levels just
    raise every run's noise by the same amount.
    """
    n = len(design)
    var = np.full(n, NOISE_CV ** 2, dtype=float)
    for factor, level_map in NOISE_CV_BY_LEVEL.items():
        if factor not in design.columns:
            continue
        extra = design[factor].map(lambda lv: level_map.get(lv, 0.0)).to_numpy(float)
        var = var + extra ** 2
    return BASELINE * np.sqrt(var)


def power_for_factor(design: pd.DataFrame, factor: str,
                     include_interaction: bool = False,
                     n_sims: int = 400, seed: int = 0) -> float:
    """Probability that `factor`'s F-test reaches p < ALPHA."""
    X_full, names = build_model_matrix(design, include_interaction)
    n, p_full = X_full.shape
    if n <= p_full:
        return 0.0                       # model not even estimable

    # the "reduced" model = same model WITHOUT this factor's columns
    keep = [i for i, nm in enumerate(names)
            if not nm.startswith(f"{factor}[") and f":{factor}[" not in nm]
    X_red = X_full[:, keep]
    df_factor = p_full - len(keep)       # df the factor occupies
    df_err = n - p_full

    # The "true world": ONE level of the factor shifts the response by
    # EFFECT_SIZE. Since we don't know in advance WHICH level is the
    # good one (which coil, which foam...), we rotate the shifted level
    # across all of them and average — this also treats unbalanced
    # designs fairly (a level with fewer runs is harder to detect).
    levels = FACTORS[factor]
    rng = np.random.default_rng(seed)
    f_crit = stats.f.ppf(1 - ALPHA, df_factor, df_err)
    sigma = run_noise_sigma(design)          # per-run noise SD (may vary)
    sims_per_level = max(50, n_sims // len(levels))
    hits = total = 0
    for shifted in levels:
        truth = np.where(design[factor] == shifted,
                         BASELINE * (1 + EFFECT_SIZE), BASELINE)
        for _ in range(sims_per_level):
            y = truth + rng.normal(0, 1, size=n) * sigma
            rss_full = _fit_rss(X_full, y)
            rss_red = _fit_rss(X_red, y)
            F = ((rss_red - rss_full) / df_factor) / (rss_full / df_err)
            hits += F > f_crit
            total += 1
    return hits / total


def power_table(design: pd.DataFrame, include_interaction: bool = False,
                n_sims: int = 400) -> dict:
    """Power for every factor, as {factor: power}."""
    return {f: power_for_factor(design, f, include_interaction, n_sims)
            for f in FACTORS}


if __name__ == "__main__":
    from taguchi import design_l25
    d = design_l25()
    print(f"Assumptions: a real {EFFECT_SIZE:.0%} effect, {NOISE_CV:.0%} "
          f"build-to-build noise, significance level {ALPHA}")
    print("\nPower of the L25 design (probability of detecting the effect):")
    for f, pw in power_table(d).items():
        print(f"  {f:15s} {pw:.0%}")
