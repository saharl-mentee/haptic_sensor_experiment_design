# Phase 2 — Analysis Pipeline (run after the Phase 1 builds are measured)

Turns the raw frequency-vs-force measurements from the ~20 Phase 1 builds into
a decision: **which foam / thickness / metal / coil configuration wins, and
why**, with an ANOVA that quantifies each factor's effect.

Not built yet — this is the roadmap. Build it once Phase 1 has produced real
numbers. Everything stays in the educational style of the Phase 0 code
(plain-language docstrings, printed explanations, small readable functions).

## Input

The raw-data CSV template that Phase 1 will generate — one row per
(build, coil, cycle, mass) with the measured resonant frequency. **Each build
carries 3-4 coils, so the response is multivariate:** one frequency reading per
coil, per force, per cycle.

## Stage A — Extract a response per build (per coil)

Collapse each build's mass-sweep × 3 cycles into a few scalar responses, one
set **per coil**:

- **Sensitivity** = slope of the calibration curve over the working range.
  Fit both `f` vs force and `1/f²` vs force (the latter is linear in the
  inductance L for an LC circuit, so it often straightens the curve) and keep
  the more linear one. Report the slope and the fit R².
- **Dynamic range** = force span where the response stays usable (linear /
  above the noise floor) before saturation.
- **Repeatability / hysteresis** = SD of frequency across the 3 cycles at each
  mass; hysteresis = gap between the loading and unloading curves.
- **Separability check (feeds Phase 3):** confirm the force-response *shape*
  is consistent across builds/coils — i.e. `Δf ≈ S · g(force)` with a common
  `g`. If it holds, Phase 3 can skip force sweeps at most pin locations.

## Stage B — ANOVA per response

For each response (and each coil, or an aggregate):

- Fit `response ~ C(coil) + C(foam_material) + foam_thickness
  + I(foam_thickness**2) + C(metal) + <enabled interactions>`, Type-II ANOVA.
  (Thickness treated as a continuous trend here — 1-2 df — since it is
  physically continuous; the 1/d³ physics means keep at least a quadratic term.)
- Diagnostics: residual plots, normality (QQ), log-transform the response if
  variance grows with the mean.
- **Effect plots**: main-effect means ± CI per factor, plus interaction plots
  for whichever pairs were enabled in `factors.py` — this is where you *see*
  which levels matter.
- Reuse the replicate builds (if the chosen design has them) as the pure-error
  term, and run a **lack-of-fit test**: if residual scatter >> pure error, the
  model is missing something (likely an interaction).

## Stage C — Pick the winner + confirm

- **Desirability score**: combine the responses (weighted sensitivity /
  dynamic range / repeatability) into one number; weights are yours to set.
- Predict the best combination across ALL 135 possible builds — even one not in
  the ~20 measured — and recommend a **confirmation build**: physically build
  the predicted best and verify it performs as predicted. Match → the model
  generalizes; miss → a term is missing, augment and re-fit.
- **Multi-coil caveat:** the single "best" for a *localizer* is not just the
  most sensitive coil config. Uniform-pressure Phase 1 cannot see coil spatial
  *diversity* (all coils see the same load), so use Phase 1 to pick the
  foam/metal stack and overall sensitivity, and carry **2-3 coil configs**
  forward — the localization-relevant choice is judged in Phase 3.

## Deliverables (when built)

- `analysis/extract_responses.py` — Stage A, raw CSV → per-build/-coil responses.
- `analysis/anova.py` — Stage B, statsmodels OLS + Type-II ANOVA + diagnostics
  + effect/interaction plots.
- `analysis/rank_and_confirm.py` — Stage C, desirability ranking + predicted
  best + confirmation-build spec.
- Verify on **simulated data** first (inject known effects + noise, confirm the
  pipeline recovers the injected winner and flags the true effects) — the same
  self-test discipline as Phase 0.
