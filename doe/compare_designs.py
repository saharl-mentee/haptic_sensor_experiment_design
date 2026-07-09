"""
compare_designs.py — PHASE 0 DRIVER. Run this one.

    python compare_designs.py

It builds every realistic design option for the haptic-sensor study,
scores them all on one scorecard, simulates their statistical power,
prints plain-language pros & cons, and saves everything to ../results/.

The goal: let you CHOOSE the number of builds (and the design type)
with the trade-offs in front of you, instead of guessing.

Options compared
  L18            Taguchi orthogonal array, 18 runs (coil via column-merge)
  L25            Taguchi orthogonal array, 25 runs (coil perfectly balanced)
  D-opt 20       computer-optimized 20 runs, main-effects model
  D-opt 22+int   22 runs, also estimates the foam x thickness interaction
  D-opt 25+int+r 22 optimal runs + 3 replicate builds (pure-error check)
  Full factorial all 135 combinations — reference only
"""

import os
import pandas as pd

from factors import NOISE_CV, EFFECT_SIZE, ALPHA
from model_matrix import print_df_accounting
from taguchi import design_l18, design_l25
from optimal import d_optimal, add_replicates, candidate_set
from metrics import scorecard
from power_sim import power_table

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(RESULTS, exist_ok=True)

SEP = "=" * 78


def banner(title):
    print(f"\n{SEP}\n{title}\n{SEP}")


# ------------------------------------------------------------------
# 1. The information budget, explained
# ------------------------------------------------------------------
banner("STEP 1 — The information budget (degrees of freedom)")
print("""Each build gives ONE data point per response (its sensitivity, etc.).
The ANOVA model 'spends' these points; whatever is left estimates the
noise, and every p-value depends on that noise estimate.\n""")
print_df_accounting(20, include_interaction=False)
print()
print_df_accounting(25, include_interaction=True)

# ------------------------------------------------------------------
# 2. Build all candidate designs
# ------------------------------------------------------------------
banner("STEP 2 — Building the candidate designs")

print("Taguchi L18 / L25: pre-tabulated orthogonal arrays, adapted to the")
print("5x3x3x3 factor pattern (see taguchi.py for the two adaptation tricks).")
print("D-optimal: Fedorov exchange search over all 135 combinations, run")
print("fresh for each N and model (see optimal.py). ~seconds each...\n")

designs = {}   # name -> (design_df, include_interaction)

designs["Taguchi L18"] = (design_l18(), False)
designs["Taguchi L25"] = (design_l25(), False)
designs["Taguchi L25 +int"] = (design_l25(), True)
designs["D-opt 20"] = (d_optimal(20, include_interaction=False), False)
d22_me = d_optimal(22, include_interaction=False)
designs["D-opt 25 ME +3rep"] = (add_replicates(d22_me, 3, include_interaction=False), False)
d22_int = d_optimal(22, include_interaction=True)
designs["D-opt 22 +int"] = (d22_int, True)
designs["D-opt 25 +int +3rep"] = (add_replicates(d22_int, 3, include_interaction=True), True)
designs["Full factorial (ref)"] = (candidate_set(), True)

for name, (d, _) in designs.items():
    fn = os.path.join(RESULTS, f"design_{name.replace(' ', '_').replace('+','')}.csv")
    d.to_csv(fn, index=False)
print(f"All design matrices saved to {os.path.abspath(RESULTS)}/design_*.csv")

# ------------------------------------------------------------------
# 3. The scorecard
# ------------------------------------------------------------------
banner("STEP 3 — Scorecard (see metrics.py for what each column means)")

rows = [scorecard(name, d, inter) for name, (d, inter) in designs.items()]
table = pd.DataFrame(rows)
print()
print(table.to_string(index=False))
table.to_csv(os.path.join(RESULTS, "comparison.csv"), index=False)

print("""
How to read this:
  error df    >= 8-10 is comfortable; below that p-values get shaky.
  replicates  duplicate builds -> direct measure of build-to-build noise.
  D-eff %     information per run vs the full factorial (higher = better).
  worst VIF   1.0 = effects perfectly separated; 2 = one effect's
              uncertainty doubled by entanglement; >5 = trouble.
              CAVEAT for '+interaction' rows: interaction terms always
              correlate with their parent main effects (a coding artifact),
              so even the PERFECT design (full factorial row) shows VIF 4.
              Judge '+int' designs against that reference, not against 1.
  max |corr|  0 = fully orthogonal design (the Taguchi ideal).""")

# ------------------------------------------------------------------
# 4. Statistical power (simulation)
# ------------------------------------------------------------------
banner("STEP 4 — Statistical power (simulated, see power_sim.py)")
print(f"""Assumptions (edit factors.py to change them):
  a real effect of {EFFECT_SIZE:.0%} on one level of a factor,
  {NOISE_CV:.0%} noise between identical builds, significance {ALPHA}.
Power = probability the experiment DETECTS that real effect.
80% is the conventional target; 50% is a coin flip.\n""")

power_rows = {}
for name, (d, inter) in designs.items():
    if name.startswith("Full"):
        continue
    power_rows[name] = power_table(d, include_interaction=inter, n_sims=400)
ptab = pd.DataFrame(power_rows).T
print((ptab * 100).round(0).astype(int).astype(str).add("%").to_string())
ptab.to_csv(os.path.join(RESULTS, "power.csv"))

# ------------------------------------------------------------------
# 5. Power-vs-N sweep for the chart
# ------------------------------------------------------------------
banner("STEP 5 — How power grows with N (chart)")
print("Sweeping D-optimal designs across N and simulating each...")

sweep_me, sweep_int = {}, {}
for n in range(14, 29):
    d = d_optimal(n, include_interaction=False, n_starts=25, seed=7)
    sweep_me[n] = min(power_table(d, False, n_sims=600).values())
for n in range(18, 29):
    d = d_optimal(n, include_interaction=True, n_starts=25, seed=7)
    sweep_int[n] = min(power_table(d, True, n_sims=600).values())

# power of the Taguchi options (worst factor), for the chart
pw_l18 = min(power_rows["Taguchi L18"].values())
pw_l25 = min(power_rows["Taguchi L25"].values())

# ---- chart (palette values from the project's data-viz standard) ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
BLUE, VIOLET = "#2a78d6", "#4a3aa7"

fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)

ax.axhline(80, color=MUTED, lw=1, ls=(0, (4, 4)))
ax.annotate("80% power target", (14, 80), color=MUTED, fontsize=9,
            va="center", ha="left", xytext=(0, 7), textcoords="offset points")

xs = sorted(sweep_me)
ax.plot(xs, [sweep_me[n] * 100 for n in xs], color=BLUE, lw=2,
        marker="o", ms=4, label="D-optimal, main effects")
ax.annotate("main effects", (xs[-1], sweep_me[xs[-1]] * 100), color=BLUE,
            fontsize=9, xytext=(6, 4), textcoords="offset points")

xi = sorted(sweep_int)
ax.plot(xi, [sweep_int[n] * 100 for n in xi], color=VIOLET, lw=2,
        marker="o", ms=4, label="D-optimal, + foam×thickness")
ax.annotate("+ interaction", (xi[-1], sweep_int[xi[-1]] * 100), color=VIOLET,
            fontsize=9, xytext=(6, -12), textcoords="offset points")

for (n, pw, nm) in [(18, pw_l18, "L18"), (25, pw_l25, "L25")]:
    ax.scatter([n], [pw * 100], s=90, facecolor=SURFACE, edgecolor=INK,
               zorder=5, linewidth=1.6)
    ax.annotate(f"Taguchi {nm}", (n, pw * 100), color=INK, fontsize=9,
                xytext=(-8, 10), textcoords="offset points", ha="right")

ax.set_xlabel("Number of builds (N)", color=INK)
ax.set_ylabel("Power of the weakest factor  (%)", color=INK)
ax.set_title(f"Chance of detecting a real {EFFECT_SIZE:.0%} effect "
             f"(noise {NOISE_CV:.0%} between builds)",
             color=INK, fontsize=11, loc="left")
ax.set_ylim(0, 102)
ax.set_xticks(range(14, 29, 2))
ax.grid(axis="y", color=GRID, lw=0.8)
ax.tick_params(colors=MUTED)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color(BASELINE)
leg = ax.legend(loc="lower right", frameon=False, fontsize=9)
for t in leg.get_texts():
    t.set_color(INK)

chart = os.path.join(RESULTS, "power_vs_N.png")
fig.tight_layout()
fig.savefig(chart, facecolor=SURFACE)
print(f"Chart saved: {os.path.abspath(chart)}")

# ------------------------------------------------------------------
# 6. Plain-language pros & cons
# ------------------------------------------------------------------
banner("STEP 6 — Pros & cons, in plain language")

pw = lambda name: min(power_rows[name].values())
print(f"""
TAGUCHI L18 (18 builds)
  + Cheapest option; fully orthogonal core; foam/thickness/metal balanced.
  + Textbook-standard array — easy to cite/justify.
  - Coil unbalanced (one coil gets 6 runs, others 3) due to the merge trick.
  - Only 7 error df and no replicates: noise estimate is thin; one failed
    build hurts a lot.
  - Cannot estimate the foam x thickness interaction.
  - Worst-factor power: {pw('Taguchi L18'):.0%}

TAGUCHI L25 (25 builds)
  + Coil perfectly balanced (5 runs per coil) — the 5-level factor is its
    natural home. 14 error df. Orthogonal core, easy to justify.
  + Can even estimate the interaction if needed (see 'L25 +int' row).
  - The dummy-level trick makes foam/thickness/metal unbalanced (10/10/5):
    one level of each is known only half as precisely.
  - The 25 runs are FIXED by the array — no freedom to choose, no replicates.
  - Worst-factor power: {pw('Taguchi L25'):.0%}

D-OPTIMAL 20 (20 builds, main effects only)
  + Fewest runs that still give ~9 error df and near-balance everywhere.
  + Runs chosen freely — can accommodate constraints (e.g. drop a combo
    that cannot be manufactured) and later augmentation.
  - No interaction, no replicates.
  - Not a 'named' textbook design (justify it by the printed metrics).
  - Worst-factor power: {pw('D-opt 20'):.0%}

D-OPTIMAL 25 ME = 22 + 3 REPLICATES (main effects, 25 builds)
  + Highest power on foam material / thickness / metal of ALL options,
    AND 3 duplicate builds measure build-to-build noise directly —
    something no Taguchi array at this size gives you.
  + Coil test as strong as L25's, despite slightly less balance (4-6 vs
    exactly 5 runs/coil).
  - Cannot estimate the foam x thickness interaction.
  - Worst-factor power (coil): {pw('D-opt 25 ME +3rep'):.0%}

D-OPTIMAL 22 + INTERACTION (22 builds)
  + Adds the physically-plausible foam x thickness interaction — protects
    you from recommending a wrong thickness per foam.
  - Only 7 error df, no replicates.
  - Worst-factor power: {pw('D-opt 22 +int'):.0%}

D-OPTIMAL 25 +INT = 22 + 3 REPLICATES (interaction, 25 builds)
  + Interaction estimated AND 3 duplicate builds (10 error df total).
  - Worst-factor power: {pw('D-opt 25 +int +3rep'):.0%}

THE HONEST TRADE-OFFS THE POWER TABLE SHOWS
  1. The 5-level coil is always the WEAKEST factor (its 4 df spread the
     information thin, and each coil gets only ~4-5 runs). Detecting a
     20% coil effect reliably is what really pushes you toward N=25;
     L25 and D-opt-25-ME are tied as the best coil tests.
  2. Estimating the interaction is NOT free: interaction terms partly
     overlap the foam/thickness main effects, so at the same N their
     power DROPS (compare 'D-opt 25 ME' vs 'D-opt 25 +int'). You are
     buying insurance against a "wrong thickness for this foam" mistake,
     paying with sensitivity to the average effects.

RULE OF THUMB (all at 25 builds unless noted)
  Coil geometry is the question you care most about     -> Taguchi L25.
  Foam/metal stack is the focus + want replicates       -> D-opt 25 ME.
  Worried the best thickness depends on the foam        -> D-opt 25 +int.
  Need a citable classical design for a paper           -> Taguchi L25.
  Only ~20 builds truly affordable                      -> D-opt 20 (accept
                                                           a weak coil test).
""")

print(f"All outputs in: {os.path.abspath(RESULTS)}")
print("To test other assumptions, edit NOISE_CV / EFFECT_SIZE in factors.py "
      "and re-run.")
