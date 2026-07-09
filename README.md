# Haptic LC-Sensor — Experiment Design (Phase 0)

Toolkit for choosing the experimental design of the haptic-sensor study:
**which factor combinations to build, and how many builds (N)**, before any
sensor is fabricated. It compares Taguchi orthogonal arrays against
computer-optimized (D-optimal) designs on equal footing.

## The experiment in one paragraph

Four design factors: **coil configuration** (5 geometries), **foam material**
(3), **foam thickness** (3), **metal sheet** (3 material+thickness options).
Building every combination = 135 sensors; the budget is ~20–25. Each build is
tested on the mass jig (mass sweep, 3 cycles) and collapses to a few response
numbers — sensitivity (Δf/ΔF), dynamic range, repeatability. ANOVA on those
responses then says which factor levels matter. **The design question is which
20–25 of the 135 combinations to build so the ANOVA can answer cleanly.**

## Quick start

```bash
source ~/.venvs/haptic_doe/bin/activate     # created for this project
cd doe
python compare_designs.py                   # ~40 s, prints everything
```

Outputs land in `results/`: the scorecard (`comparison.csv`), per-factor power
(`power.csv`), every candidate design as a CSV run list (`design_*.csv`), and
the power-vs-N chart (`power_vs_N.png`).

**To test your own assumptions** (different noise level, smaller effect you
want to detect, real material names): edit the constants in `doe/factors.py`
and re-run. **To score a design of your own** (e.g. a hand-tweaked CSV):
`metrics.scorecard(name, your_dataframe)` works on any design table.

## The files

| File | What it does |
|---|---|
| `doe/factors.py` | The experiment definition + power-simulation assumptions. **Edit this one.** |
| `doe/model_matrix.py` | Design table → ANOVA model matrix; degrees-of-freedom bookkeeping |
| `doe/taguchi.py` | Standard L18/L25 orthogonal arrays, adapted to the 5×3×3×3 pattern |
| `doe/optimal.py` | D-optimal (Fedorov exchange) design search, with replicate support |
| `doe/metrics.py` | The common scorecard: error df, balance, D-efficiency, VIF, correlation |
| `doe/power_sim.py` | Monte-Carlo statistical power for any design |
| `doe/compare_designs.py` | **The driver — run this** |

Every module has a long plain-language explanation at the top and can be run
alone (`python taguchi.py`) to explore just that piece.

## DOE primer (the concepts, using this sensor as the example)

**Run / build.** One fabricated sensor = one row of the design = one data point
per response. The mass sweep and 3 cycles *within* a build are the measurement
protocol, not extra runs.

**Degrees of freedom (df) — the information budget.** A factor with k levels
costs k−1 df to estimate (effects are relative to a reference level): coil 4,
foam 2, thickness 2, metal 2, +1 for the mean = **11**. What remains,
N − 11, is the **error df** — the basis of the noise estimate every p-value
leans on. Below ~8 error df, conclusions get fragile.

**Interaction.** "The best thickness depends on which foam" is a
foam×thickness interaction; estimating it costs (3−1)×(3−1) = 4 more df.
It is the one interaction here with clear physics (both set the compliant
layer's stiffness) — coil×metal etc. are sacrificed by every ≤25-run design.

**Orthogonality.** A design is orthogonal when every pair of factor levels
co-occurs equally often — then averaging over one factor cancels the others
exactly, and effects separate perfectly. Taguchi arrays are pre-tabulated
orthogonal designs; ours don't fit the 5×3×3×3 pattern exactly, so standard
adaptation tricks (column merge, dummy level) trade away a little balance.

**D-optimality.** When no orthogonal array fits, search all 135 candidates for
the N-run subset maximizing det(XᵀX) — equivalently, minimizing the combined
uncertainty of all effect estimates. That's what R's `AlgDesign::optFederov`
and Design-Expert do; `doe/optimal.py` is the same algorithm, readable.

**D-efficiency.** Information per run relative to the full factorial. 96–98%
means each of your 20 runs works nearly as hard as a full-factorial run.

**Confounding / VIF.** If two effects always change together in your subset,
ANOVA can't tell them apart (confounded). VIF measures the damage: 1 = clean,
2 = that effect needs 2× the runs for the same precision, ∞ = not estimable.
(With interaction terms, even the perfect design shows VIF ≈ 4 — a
dummy-coding artifact; compare against the full-factorial reference row.)

**Replicates vs repeat cycles.** Re-measuring one sensor 3× captures
*measurement* noise. Building the same spec twice captures *fabrication*
noise too — that's a replicate, and it's what "pure error" means. Your 3
cycles are not replicates.

**Power.** The probability of detecting an effect that truly exists, given
noise. 80% is the standard target; 50% is a coin flip. Simulated in
`power_sim.py` rather than taken from formulas, so you can see the machinery.

## Where this is heading (later phases)

1. **Phase 1** — you pick the design/N; generate the final randomized run
   sheet + data-collection template; build and measure.
2. **Phase 2** — analysis: extract sensitivity/range/repeatability per build,
   ANOVA, effect plots, pick the winning stack; confirmation build.
3. **Phase 3** — position sensing: all 5 coils × winning stack, pin-force at
   a grid of locations (repeated-measures); plus the mirror-winding coil
   paired mini-test.
