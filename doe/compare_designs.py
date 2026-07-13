"""
compare_designs.py — STUDY 1: which DOE TYPE, and how many RUNS?

Compares the two families of designs on equal footing (the main-effects
model), across a range of run counts:
  * Taguchi orthogonal arrays (whichever of L16/L18/L25 fit the factors)
  * D-optimal designs (Fedorov search) at N = 16, 18, 20, 22, 25, 28
  * the full-factorial reference

It prints the degrees-of-freedom budget, a scorecard, a statistical-power
table, plain-language pros & cons, and a power-vs-N chart. Everything is
written to ../results/comparison/.

    python compare_designs.py

This script does NOT do augmentation or replicates or interactions — that
is the job of augment_study.py, where you decide whether a chosen run count
is enough to see the physical effect.
"""

import os

from factors import FACTORS, NOISE_CV, EFFECT_SIZE, ALPHA, full_factorial_size
from model_matrix import print_df_accounting
from taguchi import ADAPTED
from optimal import d_optimal, candidate_set
from power_sim import power_table
import report

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "results", "comparison")
os.makedirs(OUT, exist_ok=True)

N_SWEEP = [16, 18, 20, 22, 25, 28]        # D-optimal run counts to compare
pattern = " x ".join(str(len(v)) for v in FACTORS.values())

# ------------------------------------------------------------------
# 1. The information budget
# ------------------------------------------------------------------
report.banner("STEP 1 — The information budget (degrees of freedom)")
print(f"""Factor pattern: {pattern}  ->  full factorial = {full_factorial_size()} builds.
Each build gives ONE data point per response (its sensitivity, etc.).
The ANOVA model 'spends' these points; whatever is left estimates the
noise. This study fits the MAIN-EFFECTS model (no interactions):\n""")
print_df_accounting(20, include_interaction=False)

# ------------------------------------------------------------------
# 2. Build the candidate designs (main-effects model, fair comparison)
# ------------------------------------------------------------------
report.banner("STEP 2 — Building the candidate designs")
print(f"Taguchi: pre-tabulated orthogonal arrays adapted to the {pattern}")
print("pattern. D-optimal: Fedorov search, run fresh per N. ~seconds each...\n")

designs = {}   # name -> (DataFrame, include_interaction)

for name, builder in ADAPTED.items():
    try:
        designs[name] = (builder(), False)
    except ValueError as e:
        print(f"  {name}: skipped — {e}")

for n in N_SWEEP:
    designs[f"D-opt {n}"] = (d_optimal(n, include_interaction=False), False)

designs["Full factorial (ref)"] = (candidate_set(), False)

report.save_designs(designs, OUT)
print(f"Design run-lists saved to {os.path.abspath(OUT)}/designs/")

# ------------------------------------------------------------------
# 3. Scorecard
# ------------------------------------------------------------------
report.banner("STEP 3 — Scorecard (see metrics.py for column meanings)")
table = report.scorecard_table(designs, OUT)
print()
print(table.to_string(index=False))
print(report.SCORECARD_LEGEND)

# ------------------------------------------------------------------
# 4. Statistical power
# ------------------------------------------------------------------
report.banner("STEP 4 — Statistical power (simulated)")
print(f"""Assumptions (edit factors.py): a real {EFFECT_SIZE:.0%} effect on one
level of a factor, {NOISE_CV:.0%} noise between identical builds, significance
{ALPHA}. Power = chance the experiment DETECTS that effect; 80% is the target.\n""")
pf = report.power_frame(designs, OUT, skip=lambda name: name.startswith("Full"))
report.print_power(pf)

# ------------------------------------------------------------------
# 5. Power-vs-N chart (reuses the power numbers above — no extra sims)
# ------------------------------------------------------------------
report.banner("STEP 5 — Chart: how power grows with N")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#898781"
GRID, BASELINE, BLUE = "#e1e0d9", "#c3c2b7", "#2a78d6"

dopt = sorted((len(designs[n][0]), pf.loc[n].min() * 100)
              for n in pf.index if n.startswith("D-opt"))
taguchi = [(n, len(designs[n][0]), pf.loc[n].min() * 100)
           for n in pf.index if n.startswith("Taguchi")]

fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)
ax.axhline(80, color=MUTED, lw=1, ls=(0, (4, 4)))
ax.annotate("80% power target", (dopt[0][0], 80), color=MUTED, fontsize=9,
            va="center", ha="left", xytext=(0, 7), textcoords="offset points")

ax.plot([x for x, _ in dopt], [y for _, y in dopt], color=BLUE, lw=2,
        marker="o", ms=5, label="D-optimal")
for name, n, pw in taguchi:
    ax.scatter([n], [pw], s=90, facecolor=SURFACE, edgecolor=INK,
               zorder=5, linewidth=1.6)
    ax.annotate(name.replace("Taguchi ", ""), (n, pw), color=INK, fontsize=9,
                xytext=(-8, 10), textcoords="offset points", ha="right")

ax.set_xlabel("Number of builds (N)", color=INK)
ax.set_ylabel("Power of the weakest factor  (%)", color=INK)
ax.set_title(f"Detecting a real {EFFECT_SIZE:.0%} effect "
             f"(noise {NOISE_CV:.0%}) — Taguchi vs D-optimal",
             color=INK, fontsize=11, loc="left")
ax.set_ylim(0, 102)
ax.grid(axis="y", color=GRID, lw=0.8)
ax.tick_params(colors=MUTED)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color(BASELINE)
leg = ax.legend(loc="lower right", frameon=False, fontsize=9)
for t in leg.get_texts():
    t.set_color(INK)
fig.tight_layout()
chart = os.path.join(OUT, "power_vs_N.png")
fig.savefig(chart, facecolor=SURFACE)
print(f"Chart saved: {os.path.abspath(chart)}")

# ------------------------------------------------------------------
# 6. Pros & cons — the two TYPES
# ------------------------------------------------------------------
report.banner("STEP 6 — Taguchi vs D-optimal, in plain language")

# smallest swept N where D-optimal clears the 80% target on every factor
crossing = next((n for x, y in dopt for n in [x] if y >= 80), None)
best_tag = max(taguchi, key=lambda t: t[2]) if taguchi else None

print(f"""TAGUCHI orthogonal arrays
  + Classical, citable, perfectly balanced core; the array whose native
    level count matches your coil count is the natural fit.
  + Zero setup: pre-tabulated, no search.
  - Run count is FIXED by the array (16/18/25 here) — no free choice of N,
    and non-fitting factors need dummy levels that unbalance them.
  - No built-in room for replicates or a chosen interaction.""")
if best_tag:
    print(f"  Best Taguchi here: {best_tag[0]} — weakest-factor power "
          f"{best_tag[2]:.0f}% at {best_tag[1]} builds.")

print(f"""
D-OPTIMAL designs
  + You choose N freely, so you can dial power up/down and hit a budget.
  + Near-balanced at any N; can honor build constraints and be AUGMENTED
    later (see augment_study.py).
  - Not a 'named' textbook design — justify it by the metrics above.""")
if crossing:
    print(f"  D-optimal clears 80% power on every factor from N = {crossing}.")
else:
    print("  D-optimal does not clear 80% on every factor within the swept N —"
          "\n  raise N, shrink the model, or accept the weakest factor.")

print(f"""
NOTE  The weakest factor is whichever has the most levels (its information
is spread thinnest). That factor sets the run count you need. To test
whether N is enough to see a real PHYSICAL effect — including interactions
and with replicate builds — use augment_study.py.

All outputs: {os.path.abspath(OUT)}
Edit NOISE_CV / EFFECT_SIZE / FACTORS in factors.py and re-run to explore.""")
