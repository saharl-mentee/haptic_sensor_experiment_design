"""
compare_designs.py — PHASE 0 DRIVER. Run this one.

    python compare_designs.py

It builds every realistic design option for the haptic-sensor study,
scores them all on one scorecard, simulates their statistical power,
prints plain-language pros & cons, and saves everything to ../results/.

The goal: let you CHOOSE the number of builds (and the design type)
with the trade-offs in front of you, instead of guessing.

The option list adapts to factors.py: Taguchi arrays that fit the
current coil count (L16 suits 4 coils, L25 suits 5, L18 up to 6),
D-optimal designs at N = 20/22/25, and the full factorial reference.
"""

import os
import pandas as pd

from factors import (FACTORS, INTERACTIONS, NOISE_CV, EFFECT_SIZE, ALPHA,
                     full_factorial_size)
from model_matrix import print_df_accounting, model_df
from taguchi import ADAPTED
from optimal import d_optimal, add_replicates, augment_design, candidate_set
from metrics import scorecard, d_efficiency
from power_sim import power_table

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(RESULTS, exist_ok=True)

SEP = "=" * 78


def banner(title):
    print(f"\n{SEP}\n{title}\n{SEP}")


pattern = " x ".join(str(len(v)) for v in FACTORS.values())

# ------------------------------------------------------------------
# 1. The information budget, explained
# ------------------------------------------------------------------
banner("STEP 1 — The information budget (degrees of freedom)")
print(f"""Factor pattern: {pattern}  ->  full factorial = {full_factorial_size()} builds.
Each build gives ONE data point per response (its sensitivity, etc.).
The ANOVA model 'spends' these points; whatever is left estimates the
noise, and every p-value depends on that noise estimate.\n""")
print_df_accounting(20, include_interaction=False)
print()
print_df_accounting(25, include_interaction=True)

# ------------------------------------------------------------------
# 2. Build all candidate designs
# ------------------------------------------------------------------
banner("STEP 2 — Building the candidate designs")

print(f"Taguchi: pre-tabulated orthogonal arrays adapted to the {pattern}")
print("pattern (see taguchi.py for the adaptation tricks and their price).")
print("D-optimal: Fedorov exchange search over all candidates, run fresh")
print("for each N and model (see optimal.py). ~seconds each...\n")

designs = {}   # name -> (design_df, include_interaction)

for name, builder in ADAPTED.items():
    try:
        d = builder()
    except ValueError as e:
        print(f"  {name}: skipped — {e}")
        continue
    designs[name] = (d, False)
    # re-use the same array for the interaction model, but only when
    # it leaves enough error df for the test to mean anything
    if len(d) - model_df(True) >= 7:
        designs[f"{name} +int"] = (d, True)

designs["D-opt 20"] = (d_optimal(20, include_interaction=False), False)
designs["D-opt 20 +int"] = (d_optimal(20, include_interaction=True), True)
designs["D-opt 16"] = (d_optimal(16, include_interaction=False), False)
d22_me = d_optimal(22, include_interaction=False)
designs["D-opt 25"] = (d_optimal(25, include_interaction=False), False)
designs["D-opt 25 ME +3rep"] = (add_replicates(d22_me, 3, include_interaction=False), False)
d22_int = d_optimal(22, include_interaction=True)
d25_int = d_optimal(25, include_interaction=True)
designs["D-opt 22 +int"] = (d22_int, True)
designs["D-opt 25 +int +3rep"] = (add_replicates(d22_int, 3, include_interaction=True), True)
d28_int = d_optimal(28, include_interaction=True)
designs["D-opt 28"] = (d_optimal(28, include_interaction=False), False)
designs["D-opt 28 +int"] = (d28_int, True)
designs["D-opt 28 +int +3rep"] = (add_replicates(d25_int, 3, include_interaction=True), True)
d35_int = d_optimal(35, include_interaction=True)
designs["D-opt 35"] = (d35_int, False)
designs["D-opt 35 +int"] = (d35_int, True)
designs["Full factorial (ref)"] = (candidate_set(), True)

# ------------------------------------------------------------------
# 2b. Augmentation path — build in stages without wasting any run
# ------------------------------------------------------------------
banner("STEP 2b — Augmentation path (grow the design in stages)")
print("""If you are unsure of the final build count, build a small D-optimal
design now and ADD runs later WITHOUT discarding any. Fixing the earlier
runs and optimizing only the additions guarantees the small design is
NESTED inside the big one (so every sensor you already built still counts).
The only price is a small drop in D-efficiency vs designing the big one
from scratch — the 'decline' column below is exactly that price.\n""")

KEYS = list(FACTORS)
_AUG_STARTS = 12          # fewer restarts here to keep this section quick


def _rowset(df):
    return set(map(tuple, df[KEYS].to_numpy()))


def augmentation_path(inter, stages, tag):
    """Grow stages[0] -> stages[1] -> ... by augmentation; print the
    efficiency decline vs from-scratch at each N. Returns {N: design}."""
    print(f"  {tag}")
    print(f"    {'N':>3}  {'D-eff scratch':>13}  {'D-eff augmented':>15}  "
          f"{'decline':>8}  {'orig. runs kept':>15}")
    first = base = d_optimal(stages[0], include_interaction=inter,
                             n_starts=_AUG_STARTS)
    prev, out = base, {}
    for i, N in enumerate(stages):
        scratch = d_optimal(N, include_interaction=inter, n_starts=_AUG_STARTS)
        aug = base if i == 0 else augment_design(
            prev, N - len(prev), include_interaction=inter, n_starts=_AUG_STARTS)
        es, ea = d_efficiency(scratch, inter), d_efficiency(aug, inter)
        kept = len(_rowset(first) & _rowset(aug))
        print(f"    {N:>3}  {es:>12.1f}%  {ea:>14.1f}%  "
              f"{es - ea:>6.1f}pt  {kept:>7}/{len(first)}")
        out[N], prev = aug, aug
    return out

# Main-effects staged path — always affordable, always estimable.
augmentation_path(False, [16, 20, 25], "Main-effects model:  16 -> 20 -> 25")

# Interaction staged path — needs N > model params, so scale it to
# however many interactions are enabled in factors.py right now.
print()
p_int = model_df(True)
lo = max(20, p_int + 2)             # first stage: estimable, with error df
int_stages = [lo, lo + 4, lo + 8]
if int_stages[0] <= 28:
    aug_int = augmentation_path(
        True, int_stages,
        f"Interaction model:  {int_stages[0]} -> {int_stages[1]} -> {int_stages[2]}"
        f"  (needs {p_int} df for the model)")
    # surface the augmented endpoint in the main scorecard/power tables
    designs[f"D-opt {int_stages[-1]} aug +int"] = (aug_int[int_stages[-1]], True)
else:
    print(f"  Interaction model: skipped — the {len(INTERACTIONS)} interaction "
          f"pair(s) enabled in factors.py need {p_int} df, too rich to augment")
    print(f"  within ~28 builds. Enable fewer pairs to explore this path.")

for name, (d, _) in designs.items():
    fn = os.path.join(RESULTS, f"design_{name.replace(' ', '_').replace('+','')}.csv")
    d.to_csv(fn, index=False)
print(f"\nAll design matrices saved to {os.path.abspath(RESULTS)}/design_*.csv")

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
              so even the PERFECT design (full factorial row) shows an
              elevated VIF. Judge '+int' designs against that reference
              row, not against 1.
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

# the factor hardest to detect, on average across all options
weakest_factor = ptab.mean(axis=0).idxmin()

# ------------------------------------------------------------------
# 5. Power-vs-N sweep for the chart
# ------------------------------------------------------------------
banner("STEP 5 — How power grows with N (chart)")
print("Sweeping D-optimal designs across N and simulating each...")

sweep_me, sweep_int = {}, {}
for n in range(14, 29):
    d = d_optimal(n, include_interaction=False, n_starts=25, seed=7)
    sweep_me[n] = min(power_table(d, False, n_sims=600).values())
for n in range(model_df(True) + 3, 29):
    d = d_optimal(n, include_interaction=True, n_starts=25, seed=7)
    sweep_int[n] = min(power_table(d, True, n_sims=600).values())

# Taguchi options as points on the same axes (main-effects rows only)
taguchi_pts = {name: (len(designs[name][0]), min(pw.values()))
               for name, pw in power_rows.items()
               if name.startswith("Taguchi") and "+int" not in name}

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

for name, (n, pw) in taguchi_pts.items():
    ax.scatter([n], [pw * 100], s=90, facecolor=SURFACE, edgecolor=INK,
               zorder=5, linewidth=1.6)
    ax.annotate(name, (n, pw * 100), color=INK, fontsize=9,
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
# 6. Plain-language pros & cons (numbers pulled from the tables above)
# ------------------------------------------------------------------
banner("STEP 6 — Pros & cons, in plain language")

# What each design TYPE inherently is — facts that don't depend on N
# or the current factor levels. The numbers are filled in live.
WHY = {
    "Taguchi L16": "Classical 4-level array — natural home for a 4-coil study;\n"
                   "    coil perfectly balanced. Cheapest classical option, but the\n"
                   "    3-level factors carry a dummy level (one level built 2x more).",
    "Taguchi L18": "Classical array, cheap. Coil enters via column-merging with\n"
                   "    cyclic dummy repeats -> coil unbalanced; other factors balanced.",
    "Taguchi L25": "Classical 5-level array — natural home for a 5-coil study.\n"
                   "    3-level factors carry dummy levels (10/10/5 imbalance). Runs\n"
                   "    are FIXED by the array: no freedom, no replicates.",
    "D-opt 20": "Computer-chosen runs, near-balance everywhere; can honor\n"
                "    constraints (e.g. drop an unmanufacturable combo) and be\n"
                "    augmented later. Justify it by the printed metrics, not a name.",
    "D-opt 25 ME +3rep": "Computer-chosen + 3 replicate builds: a direct, assumption-free\n"
                         "    measure of build-to-build noise that no small Taguchi array\n"
                         "    offers. Usually the sharpest main-effects option per run.",
    "D-opt 22 +int": "Estimates the foam x thickness interaction — insurance against\n"
                     "    recommending a wrong thickness per foam. Thin error df.",
    "D-opt 25 +int +3rep": "Interaction insurance AND replicates — the most complete\n"
                           "    model your ~25-build budget can carry.",
}

score_by_name = {r["design"]: r for r in rows}
for name in designs:
    if name.startswith("Full") or name.endswith("+int") and name.startswith("Taguchi"):
        continue
    s = score_by_name[name]
    pw = power_rows[name]
    worst = min(pw, key=pw.get)
    why = WHY.get(name, "")
    print(f"\n{name.upper()}  ({s['runs']} builds, {s['model']})")
    if why:
        print(f"    {why}")
    print(f"    error df {s['error df']}, replicates {s['replicates']}, "
          f"D-eff {s['D-eff %']}%")
    print(f"    balance: {s['balance (runs/level)']}")
    print(f"    power: worst factor is '{worst}' at {pw[worst]:.0%}"
          f"  (others: " +
          ", ".join(f"{f} {p:.0%}" for f, p in pw.items() if f != worst) + ")")

print(f"""
THE TWO LESSONS THE TABLES KEEP TEACHING
  1. The weakest factor here is '{weakest_factor}' — the factor with the
     most levels always is: its df spread the information thin and each
     level gets the fewest builds. Reliably detecting an effect on it is
     what drives the total build count.
  2. Estimating the interaction is NOT free: its terms partly overlap
     the parent factors' main effects, so at the same N those factors'
     power DROPS (compare the ME and +int rows at equal N). You are
     buying insurance against a wrong combined choice, paying with
     sensitivity to the average effects.

CHOOSING
  Classical, citable design for a paper/report  -> the Taguchi row whose
                                                   native level count
                                                   matches your coil count.
  Sharpest factor screening + fabrication check -> D-opt ME + replicates.
  Worried the best thickness depends on foam    -> a '+int' option.
  Squeezed budget                               -> smallest row with
                                                   error df >= 8 and
                                                   acceptable power.
""")

print(f"All outputs in: {os.path.abspath(RESULTS)}")
print("To test other assumptions, edit NOISE_CV / EFFECT_SIZE / FACTORS in "
      "factors.py and re-run.")
