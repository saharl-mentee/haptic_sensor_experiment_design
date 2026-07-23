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
    # "coil": ["6", "8", "9", "10", "11"],
    # "coil": ["6", "8", "9", "10"],
    "coil": ["6", "8", "10"],

    # 3 candidate foam materials.
    "foam_material": ["H800", "H870", "Poron"],

    # 3 foam thicknesses. Physically this is a continuous quantity;
    # for CHOOSING which runs to build, we treat it as 3 levels
    # (same "cost" either way: 3 levels = 2 degrees of freedom).
    # In the final ANALYSIS we can fit it as a continuous trend.
    # "foam_thickness": ["1.58", "2.39", "3.18"],
    "foam_thickness": ["1.58", "2.39"],

    # 3 metal sheet options. NOTE: material and thickness change
    # TOGETHER here (they are bundled into one factor), so the
    # experiment cannot say whether an effect came from the metal's
    # material or its thickness — only which sheet option is best.
    "metal": ["copper", "aluminum", "ss mesh"],
}

# ---------------------------------------------------------------
# Which interactions do we consider physically plausible enough to
# spend runs on?
#
# An INTERACTION means: "the effect of one factor depends on the
# level of another". Example: maybe thin foam is best for a stiff
# foam material but thick foam is best for a soft one. If that is
# true and we only fit "main effects", we would pick a wrong combo.
#
# This is a LIST — uncomment/add the pairs you believe in, and re-run
# compare_designs.py to watch the df cost and power price of each.
# When include_interaction=True is passed anywhere, EVERY pair listed
# here is added to the model. Each pair (a x b) costs
# (len(a)-1)*(len(b)-1) extra df.
#
# Physics notes for THIS sensor (frequency shifts ~1/d^3 with the
# coil-to-metal gap d, which the foam sets):
#  * foam_material x foam_thickness — both set the compliant layer's
#    stiffness (how far the metal moves per unit force). Costs 4 df.
#  * foam_thickness x metal — the gap distance (set by thickness)
#    steeply modulates how much the metal couples to the coil, so the
#    metal's effect may be much larger at small gaps. Costs 4 df.
#  * foam_thickness x coil — same gap argument, per coil geometry.
#    Costs (3-1)*(n_coils-1) df.
# ---------------------------------------------------------------
INTERACTIONS = [
    ("foam_material", "foam_thickness"),
    ("foam_thickness", "metal"),
    ("foam_thickness", "coil"),
]

# ---------------------------------------------------------------
# Assumptions used ONLY by the power simulation (power_sim.py).
# Change them to match your intuition / first measurements, and
# re-run compare_designs.py to see how the conclusions shift.
# ---------------------------------------------------------------

# How much does the response (e.g. sensitivity) vary between two
# sensors built to the SAME spec, as a fraction of its value?
# This is the BASELINE build-to-build + measurement noise that every
# run has, whatever its factor levels.
# 0.10 = 10% coefficient of variation. If your fabrication is very
# repeatable use 0.05; if it is crude, 0.15-0.20.
NOISE_CV = 0.1

# ---------------------------------------------------------------
# OPTIONAL: selective (level-dependent) noise.
#
# Some parts of the build are more reproducible than others: a coil
# PCB comes out nearly identical every time (~0 extra noise), while a
# hand-cut soft foam may scatter a lot. List that EXTRA noise per
# level here; it is ADDED (in quadrature) on top of NOISE_CV:
#
#   sigma_run = BASELINE * sqrt( NOISE_CV^2 + sum_f extra_cv(level_f)^2 )
#
# Rules:
#  * A factor you don't list, or a level you don't list, adds 0 extra
#    (so an empty dict {} reproduces the plain single-NOISE_CV behavior).
#  * Only LEVEL-dependent differences change the statistics. A factor
#    that adds the same extra noise to all its levels is mathematically
#    just a bigger global NOISE_CV and won't change any comparison.
#
# Example (uncomment / edit):
# NOISE_CV_BY_LEVEL = {
#     "foam_material": {"FoamA": 0.03, "FoamC": 0.15},  # soft foam noisy
#     "coil":          {"C1": 0.0, "C2": 0.0},          # PCB: no extra
# }
# NOISE_CV_BY_LEVEL = {"coil":{"6": 0.01, "8": 0.01, "10":0.01},
#                      "foam_material":{"H800": 0.1, "H870": 0.1, "Poron":0.1},
#                      "foam_thickness": {"1.58":0.3, "2.39":0.2, "3.18":0.15},
#                      "metal": {"copper": 0.3, "aluminum":0.3, "ss mesh":0.2}
# }
NOISE_CV_BY_LEVEL = {}

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
