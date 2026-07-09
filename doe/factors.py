"""
factors.py — The single place where the experiment is defined.

Edit THIS file when you know the real material names / thicknesses,
or when you want to test different assumptions about noise and effect
size. Every other script imports from here.

--------------------------------------------------------------------
Vocabulary used throughout the project (plain language):

* FACTOR   = a design parameter you control (e.g. foam material).
* LEVEL    = one of the values a factor can take (e.g. FoamA).
* RUN      = one complete sensor BUILD, tested with the full
             mass-sweep protocol. The DOE decides which factor-level
             combinations to build.
* RESPONSE = the number(s) you extract from each run's measurements
             (sensitivity, dynamic range, repeatability). The ANOVA
             is computed on these, one value per run.
--------------------------------------------------------------------
"""

# ---------------------------------------------------------------
# The four design factors and their levels (placeholders for now).
# Order of levels does not matter for the statistics.
# ---------------------------------------------------------------
FACTORS = {
    # 5 truly different coil geometries. The 2 "mirror winding"
    # variants are NOT here on purpose: they will be tested later as
    # a small paired comparison against the winning configuration.
    # "coil": ["C1", "C2", "C3", "C4", "C5"],
    "coil": ["C1", "C2", "C3", "C4"],
    # "coil": ["C1", "C2", "C3"],

    # 3 candidate foam materials.
    "foam_material": ["FoamA", "FoamB", "FoamC"],

    # 3 foam thicknesses. Physically this is a continuous quantity;
    # for CHOOSING which runs to build, we treat it as 3 levels
    # (same "cost" either way: 3 levels = 2 degrees of freedom).
    # In the final ANALYSIS we can fit it as a continuous trend.
    "foam_thickness": ["T1", "T2", "T3"],

    # 3 metal sheet options. NOTE: material and thickness change
    # TOGETHER here (they are bundled into one factor), so the
    # experiment cannot say whether an effect came from the metal's
    # material or its thickness — only which sheet option is best.
    "metal": ["MetalA", "MetalB", "MetalC"],
}

# ---------------------------------------------------------------
# Which interaction do we consider physically plausible enough to
# spend runs on?
#
# An INTERACTION means: "the effect of one factor depends on the
# level of another". Example: maybe thin foam is best for a stiff
# foam material but thick foam is best for a soft one. If that is
# true and we only fit "main effects", we would pick a wrong combo.
#
# foam_material x foam_thickness is the natural candidate: both set
# the stiffness of the compliant layer that converts force into
# deflection. Estimating it costs (3-1)*(3-1) = 4 extra runs' worth
# of information.
# ---------------------------------------------------------------
PLAUSIBLE_INTERACTION = ("foam_material", "foam_thickness")

# ---------------------------------------------------------------
# Assumptions used ONLY by the power simulation (power_sim.py).
# Change them to match your intuition / first measurements, and
# re-run compare_designs.py to see how the conclusions shift.
# ---------------------------------------------------------------

# How much does the response (e.g. sensitivity) vary between two
# sensors built to the SAME spec, as a fraction of its value?
# This is build-to-build + measurement noise combined.
# 0.10 = 10% coefficient of variation. If your fabrication is very
# repeatable use 0.05; if it is crude, 0.15-0.20.
NOISE_CV = 0.10

# The smallest difference between factor levels that you would care
# about detecting, as a fraction of the baseline response.
# 0.20 = "I want to notice when one foam gives 20% more sensitivity
# than another". Smaller values need more runs to detect.
EFFECT_SIZE = 0.20

# Arbitrary baseline response value (units cancel out everywhere —
# only the RATIOS above matter).
BASELINE = 1.0

# Significance level for all statistical tests (the standard 5%:
# we accept a 5% chance of crying "effect!" when there is none).
ALPHA = 0.05


def n_levels(factor: str) -> int:
    return len(FACTORS[factor])


def full_factorial_size() -> int:
    """Number of runs if every combination were built (the reference)."""
    n = 1
    for levels in FACTORS.values():
        n *= len(levels)
    return n


if __name__ == "__main__":
    print("Factors and levels:")
    for name, levels in FACTORS.items():
        print(f"  {name:15s} {len(levels)} levels: {', '.join(levels)}")
    print(f"\nFull factorial (every combination) = {full_factorial_size()} builds")
    print(f"Assumed noise between identical builds : {NOISE_CV:.0%}")
    print(f"Smallest effect worth detecting        : {EFFECT_SIZE:.0%}")
