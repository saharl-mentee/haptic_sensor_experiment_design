"""
augment_study.py — STUDY 2: grow a chosen design, add replicates, and decide
whether the run count is ENOUGH to see the physical effect statistically.

The workflow this supports:
  1. You picked a base design/size in Study 1 (compare_designs.py).
  2. Build it. Maybe it is not quite enough — so ADD runs by AUGMENTATION,
     which keeps every base run (the base is nested in the bigger design,
     so nothing you already built is wasted).
  3. Optionally ADD REPLICATE builds — exact duplicates that measure
     build-to-build noise directly ("pure error").
  4. Compare base vs augmented vs augmented+replicates on the same scorecard
     and power table as Study 1, and read a SUFFICIENCY VERDICT: does every
     factor clear the target power at your assumed effect size?

Edit the CONFIG block, then run:   python augment_study.py
Results -> ../results/augmented/.
"""

import os
import pandas as pd

from factors import FACTORS, NOISE_CV, EFFECT_SIZE, ALPHA
from model_matrix import model_df, print_df_accounting
from optimal import d_optimal, augment_design, add_replicates
from metrics import d_efficiency
import report

# ======================== CONFIG (edit me) ========================
BASE_N           = 20      # runs you commit to / build first
ADD_RUNS         = 5       # extra runs added by augmentation (0 = skip)
N_REPLICATES     = 3       # replicate builds added on top      (0 = skip)
WITH_INTERACTION = True    # fit the interaction(s) listed in factors.py?
TARGET_POWER     = 0.80    # a factor counts as "detectable" at/above this
SEED             = 42
# ==================================================================

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "results", "augmented")
os.makedirs(OUT, exist_ok=True)

KEYS = list(FACTORS)
model_label = "main effects + interactions" if WITH_INTERACTION else "main effects"

# ------------------------------------------------------------------
# 1. The model we are trying to support
# ------------------------------------------------------------------
report.banner("STEP 1 — The model, and the base design's df budget")
print(f"Fitting: {model_label}.  Target power per factor: {TARGET_POWER:.0%}.\n")
print_df_accounting(BASE_N, include_interaction=WITH_INTERACTION)

# ------------------------------------------------------------------
# 2. Build the staged designs: base -> augmented -> +replicates
# ------------------------------------------------------------------
report.banner("STEP 2 — Building base -> augmented -> +replicates")

base = d_optimal(BASE_N, include_interaction=WITH_INTERACTION, seed=SEED)
designs = {}                       # name -> (DataFrame, include_interaction)
name_base = f"base N={BASE_N}"
designs[name_base] = (base, WITH_INTERACTION)
print(f"  built base: {BASE_N} runs")

current, name_current = base, name_base
if ADD_RUNS > 0:
    aug = augment_design(base, ADD_RUNS, include_interaction=WITH_INTERACTION,
                         seed=SEED)
    name_aug = f"augmented N={BASE_N + ADD_RUNS}"
    designs[name_aug] = (aug, WITH_INTERACTION)
    current, name_current = aug, name_aug
    print(f"  augmented: +{ADD_RUNS} runs -> {len(aug)} (all {BASE_N} base runs kept)")

if N_REPLICATES > 0:
    final = add_replicates(current[KEYS], N_REPLICATES,
                           include_interaction=WITH_INTERACTION)
    name_final = f"augmented+{N_REPLICATES}rep N={len(final)}"
    designs[name_final] = (final, WITH_INTERACTION)
    current, name_current = final, name_final
    print(f"  +replicates: +{N_REPLICATES} duplicate builds -> {len(final)}")

final_N = len(current)

# from-scratch design at the final size — the yardstick for what augmentation costs
scratch = d_optimal(final_N, include_interaction=WITH_INTERACTION, seed=SEED)
name_scratch = f"from-scratch N={final_N}"
designs[name_scratch] = (scratch, WITH_INTERACTION)

report.save_designs(designs, OUT)
print(f"\nDesign run-lists saved to {os.path.abspath(OUT)}/designs/")

# ------------------------------------------------------------------
# 3. Scorecard
# ------------------------------------------------------------------
report.banner("STEP 3 — Scorecard (base vs augmented vs +replicates)")
table = report.scorecard_table(designs, OUT)
print()
print(table.to_string(index=False))
print(report.SCORECARD_LEGEND)

# ------------------------------------------------------------------
# 4. What augmentation costs in D-efficiency
# ------------------------------------------------------------------
report.banner("STEP 4 — Price of staging (augmented vs from-scratch)")
eff_aug = d_efficiency(current[KEYS], WITH_INTERACTION)
eff_scr = d_efficiency(scratch, WITH_INTERACTION)
print(f"""At the final size N={final_N}:
  from-scratch D-efficiency : {eff_scr:.1f}%
  augmented   D-efficiency : {eff_aug:.1f}%
  decline (price of keeping every base run nested): {eff_scr - eff_aug:.1f} points

A few points of decline is the normal, cheap price for being able to build
in stages and stop early. A large decline means the base was too small or
awkward for this model.""")

# ------------------------------------------------------------------
# 5. Power — can we SEE the physical effect?
# ------------------------------------------------------------------
report.banner("STEP 5 — Statistical power (simulated)")
print(f"""Assumptions (edit factors.py): a real {EFFECT_SIZE:.0%} effect on one
level of a factor, {NOISE_CV:.0%} noise between identical builds, significance
{ALPHA}. Power = chance this design DETECTS that effect.\n""")
pf = report.power_frame(designs, OUT, n_sims=600)
report.print_power(pf)

# ------------------------------------------------------------------
# 6. Sufficiency verdict for the final (augmented + replicates) design
# ------------------------------------------------------------------
report.banner("STEP 6 — Is the run count SUFFICIENT?")
final_power = pf.loc[name_current]
print(f"Final design: {name_current}  ({final_N} builds, {model_label})\n")
print(f"  {'factor':16s} {'power':>7s}   verdict (target {TARGET_POWER:.0%})")
all_ok = True
for factor, pw in final_power.items():
    ok = pw >= TARGET_POWER
    all_ok &= ok
    print(f"  {factor:16s} {pw:>6.0%}   {'OK' if ok else 'TOO LOW — add runs'}")

print()
if all_ok:
    print(f"VERDICT: sufficient. Every factor clears {TARGET_POWER:.0%} power — this")
    print(f"design can see a real {EFFECT_SIZE:.0%} effect on all factors.")
else:
    weak = final_power.idxmin()
    print(f"VERDICT: not yet sufficient. Weakest factor '{weak}' is at "
          f"{final_power.min():.0%} (< {TARGET_POWER:.0%}).")
    print("Raise ADD_RUNS in the CONFIG block and re-run — augmentation keeps")
    print("the runs you already have, so re-running only adds to them.")

print(f"\nAll outputs: {os.path.abspath(OUT)}")
