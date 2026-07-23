"""
augment_study.py — STUDY 2: grow a chosen design and decide whether the run
count is ENOUGH to see the physical effect statistically — and how best to
SPEND the extra builds.

The workflow this supports (sequential / staged DOE):
  1. You picked a base design/size in Study 1 (compare_designs.py). Typically
     the base is a lean SCREENING design that only fits MAIN EFFECTS.
  2. Build it. To learn more you ADD extra builds by augmentation, which keeps
     every base run (the base is nested, so nothing already built is wasted) —
     and the added runs are chosen to support a RICHER model, e.g. one that now
     includes INTERACTIONS. So the model can grow with the run count:
        base      : BASE_INTERACTION  (usually main effects only)
        augmented : ADDED_INTERACTION (usually main effects + interactions)
  3. Given a fixed budget of extra builds, is it better to spend them all on NEW
     distinct runs, or to mix in some REPLICATE builds (exact duplicates that
     measure build-to-build "pure error")? Both are built to the SAME total size
     so they cost the same, and are compared head-to-head:
        A "all-new"  : ADD_TOTAL new distinct runs
        B "with-rep" : (ADD_TOTAL - N_REPLICATES) new runs + N_REPLICATES reps
  4. Scorecard + power table + a SUFFICIENCY VERDICT for each option.

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
BASE_N            = 18     # runs you commit to / build first
BASE_INTERACTION  = False  # base MODEL: False = main effects only (screening)
ADD_TOTAL         = 10     # total EXTRA builds beyond the base (same budget
                           # for BOTH options below, so they cost the same)
ADDED_INTERACTION = True   # model AFTER adding runs: True = + interactions
                           # (the added runs are chosen to support THIS model)
N_REPLICATES      = 3      # in the "with replicates" option, how many of the
                           # ADD_TOTAL extra builds are replicate duplicates
                           # instead of new distinct runs. So:
                           #   option A "all-new"  : ADD_TOTAL new distinct runs
                           #   option B "with-rep" : (ADD_TOTAL - N_REPLICATES)
                           #                         new runs + N_REPLICATES reps
TARGET_POWER      = 0.80   # a factor counts as "detectable" at/above this
SEED              = 50
# ==================================================================

assert 0 <= N_REPLICATES <= ADD_TOTAL, \
    "N_REPLICATES must be 0..ADD_TOTAL (replicates come OUT of the added budget)"
FINAL_N    = BASE_N + ADD_TOTAL
NEW_IN_REP = ADD_TOTAL - N_REPLICATES   # new distinct runs in the with-rep option

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "results", "augmented")
os.makedirs(OUT, exist_ok=True)

KEYS = list(FACTORS)
def _label(inter):
    return "main effects + interactions" if inter else "main effects"
base_label, added_label = _label(BASE_INTERACTION), _label(ADDED_INTERACTION)

# ------------------------------------------------------------------
# 1. The models we are trying to support, and the df budget
# ------------------------------------------------------------------
report.banner("STEP 1 — Models and df budget: base vs augmented")
print(f"Base model     : {base_label}   at N={BASE_N}")
print(f"Augmented model: {added_label}   at N={FINAL_N}  (+{ADD_TOTAL})")
print(f"Target power per factor: {TARGET_POWER:.0%}.\n")
print(f"[base N={BASE_N}, {base_label}]")
print_df_accounting(BASE_N, include_interaction=BASE_INTERACTION)
print(f"\n[augmented N={FINAL_N}, {added_label}]")
print_df_accounting(FINAL_N, include_interaction=ADDED_INTERACTION)
if BASE_INTERACTION != ADDED_INTERACTION:
    print("\nNote: the model GROWS when you add runs — the base only has to")
    print("support main effects; the added builds pay for the interaction terms.")

# ------------------------------------------------------------------
# 2. Build: main-effects base, then augment to the richer model
# ------------------------------------------------------------------
report.banner("STEP 2 — Base (screening), then two ways to spend the extra builds")

base = d_optimal(BASE_N, include_interaction=BASE_INTERACTION, seed=SEED)
print(f"  base: {BASE_N} runs, fits {base_label}")

# Option A: all extra builds are NEW distinct runs, chosen for the ADDED model
all_new = augment_design(base, ADD_TOTAL, include_interaction=ADDED_INTERACTION,
                         seed=SEED)
print(f"  option A 'all-new' : +{ADD_TOTAL} new distinct runs -> {len(all_new)} "
      f"({added_label})")

# Option B: fewer new runs, remainder as replicate builds -> same total
interm = (base if NEW_IN_REP == 0
          else augment_design(base, NEW_IN_REP,
                              include_interaction=ADDED_INTERACTION, seed=SEED))
with_rep = add_replicates(interm[KEYS], N_REPLICATES,
                          include_interaction=ADDED_INTERACTION)
print(f"  option B 'with-rep': +{NEW_IN_REP} new runs +{N_REPLICATES} replicates "
      f"-> {len(with_rep)} ({added_label})")

# From-scratch design at the final size & model — yardstick for staging cost
scratch = d_optimal(FINAL_N, include_interaction=ADDED_INTERACTION, seed=SEED)

name_base    = f"base N={BASE_N} [ME]"
name_allnew  = f"all-new (+{ADD_TOTAL}) N={FINAL_N} [+int]"
name_withrep = f"{NEW_IN_REP}new+{N_REPLICATES}rep N={FINAL_N} [+int]"
name_scratch = f"from-scratch N={FINAL_N} [+int]"

designs = {                                   # name -> (DataFrame, include_interaction)
    name_base:    (base,     BASE_INTERACTION),
    name_allnew:  (all_new,  ADDED_INTERACTION),
    name_withrep: (with_rep, ADDED_INTERACTION),
    name_scratch: (scratch,  ADDED_INTERACTION),
}

report.save_designs(designs, OUT)
print(f"\nDesign run-lists saved to {os.path.abspath(OUT)}/designs/")

# ------------------------------------------------------------------
# 3. Scorecard (each design scored under ITS OWN model)
# ------------------------------------------------------------------
report.banner("STEP 3 — Scorecard (base under ME; augmented under +int)")
table = report.scorecard_table(designs, OUT)
print()
print(table.to_string(index=False))
print(report.SCORECARD_LEGEND)
print("\nThe 'model' column shows the base is scored as main-effects and the")
print("augmented designs as +interactions — the staged model this study is about.")

# ------------------------------------------------------------------
# 4. What the choices cost in D-efficiency (at the augmented model)
# ------------------------------------------------------------------
report.banner("STEP 4 — D-efficiency of the augmented options (interaction model)")
eff_scr = d_efficiency(scratch, ADDED_INTERACTION)
eff_new = d_efficiency(all_new[KEYS], ADDED_INTERACTION)
eff_rep = d_efficiency(with_rep[KEYS], ADDED_INTERACTION)
print(f"""All three are N={FINAL_N} builds, scored on the {added_label} model:
  from-scratch          : {eff_scr:5.1f}%   (ideal, but doesn't reuse the base)
  all-new (augmented)   : {eff_new:5.1f}%   (staging cost vs scratch: {eff_scr-eff_new:.1f} pt)
  with-replicates       : {eff_rep:5.1f}%   (cost of {N_REPLICATES} replicates
                                     instead of new runs: {eff_new-eff_rep:.1f} pt)

Because the base was built for main effects only, the augmented design has
to 'catch up' to support the interaction terms — any staging cost here is
the price of having screened first and committed to interactions later.""")

# ------------------------------------------------------------------
# 5. Power — can each stage SEE the physical effect?
# ------------------------------------------------------------------
report.banner("STEP 5 — Statistical power (simulated)")
print(f"""Assumptions (edit factors.py): a real {EFFECT_SIZE:.0%} effect on one
level of a factor, {NOISE_CV:.0%} baseline noise between identical builds,
significance {ALPHA}. Power = chance the design DETECTS that effect.
(Power shown is for the MAIN-EFFECT factors, each design under its own model;
detecting the interaction terms themselves is a separate question.)\n""")
pf = report.power_frame(designs, OUT, n_sims=600)
report.print_power(pf)

# ------------------------------------------------------------------
# 6. Sufficiency verdict — the two augmented (interaction-model) options
# ------------------------------------------------------------------
report.banner("STEP 6 — Is the augmented run count SUFFICIENT?")
print(f"Both options: {FINAL_N} builds, {added_label}, target {TARGET_POWER:.0%}.")
print(f"(Base for reference: {BASE_N} builds, {base_label}.)\n")

verdicts = {}
for tag, nm in [("A all-new", name_allnew), ("B with-rep", name_withrep)]:
    power = pf.loc[nm]
    print(f"  {tag}  ({nm})")
    print(f"    {'factor':16s} {'power':>7s}   verdict")
    all_ok = True
    for factor, pw in power.items():
        ok = pw >= TARGET_POWER
        all_ok &= ok
        print(f"    {factor:16s} {pw:>6.0%}   {'OK' if ok else 'TOO LOW'}")
    verdicts[tag] = all_ok
    print(f"    -> {'SUFFICIENT' if all_ok else 'NOT yet sufficient'}\n")

if verdicts["A all-new"] and verdicts["B with-rep"]:
    print("Both options clear the target on the interaction model. Prefer")
    print("'with-rep': same power AND a real build-to-build noise estimate.")
elif verdicts["A all-new"] and not verdicts["B with-rep"]:
    weak = pf.loc[name_withrep].idxmin()
    print(f"Only 'all-new' is sufficient. Spending {N_REPLICATES} builds on replicates")
    print(f"drops '{weak}' below target — you need the distinct runs more than the")
    print("pure-error estimate. Lower N_REPLICATES, or raise ADD_TOTAL.")
else:
    weak = pf.loc[name_allnew].idxmin()
    print(f"Neither is sufficient yet — even all-new leaves '{weak}' below target.")
    print("Raise ADD_TOTAL and re-run (augmentation keeps every run already built).")

print(f"\nAll outputs: {os.path.abspath(OUT)}")
