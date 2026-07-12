# Phase 3 — Position-Sensing Study (pin force at grid locations)

After Phase 1/2 pick the winning foam/metal stack, take the best **2-3 coil
configurations** forward and characterize their **spatial** behaviour: a 2mm
pin applies force through a **6×5 grid (30 holes)**. Goals: (a) map the spatial
sensitivity, and (b) **localize** where a force was applied. Each device has
**3-4 coils read simultaneously**.

Not built yet — roadmap only. Same educational code style as Phase 0.

## Why this is a different design from Phase 1

The "factor" here is **position** — a continuous 2D coordinate (x, y), not a
categorical level. So this is a **spatial sampling / response-surface** problem:
sample a subset of holes, reconstruct the whole field by smooth interpolation
(thin-plate spline / kriging), and validate on held-out holes. Assumed
**asymmetric / unknown** field → no symmetry shortcut → space-filling design.

## Which holes to measure — space-filling "leave-out" (≈20 of 30)

Two rules (field is asymmetric, so):
1. **Keep the boundary** (corners + edges): interpolation between measured holes
   is reliable, extrapolation beyond them is not.
2. **Disperse the skipped holes into the interior**, none adjacent to another
   skip → each skipped hole is bracketed by measured neighbours and doubles as a
   **held-out validation point**.

Recommended pattern (● measure, · skip) — 20 measured, 10 skipped:

```
col:   1   2   3   4   5   6
row1   ●   ●   ●   ●   ·   ●
row2   ●   ·   ●   ·   ●   ·
row3   ·   ●   ·   ●   ·   ●
row4   ●   ·   ●   ·   ●   ●
row5   ●   ●   ·   ●   ●   ●
```

Fit the map, predict the 10 skipped holes, then measure ~3-4 of them to check
error. Small error → done; large error → add holes where error is worst (same
augmentation logic as Phase 1). Can also start at ~12 holes and grow.

## Keeping the experiment count low — don't cross position × force

Separable model: **Δf(x,y,F) ≈ S(x,y) · g(F)**, where `g(F)` is the force-
response curve (a stack property, already seen in Phase 1) and `S(x,y)` is the
spatial map. Then you do NOT need a force sweep at every hole:

- **~4 anchor holes** (spread across the grid): full force sweep ×3 repeats.
  Nails `g(F)` AND checks separability — if the curve shape matches across all
  4 anchors, separability holds.
- **~16 other holes**: a **single reference pin force**, one push each (a 2nd
  repeat at a handful for a repeatability spot-check).

Because the 3-4 coils are read **at once**, a single push records the full
multi-coil signature — you don't multiply work by the coil count. This cuts a
naive ~20×5×3 ≈ 240 pushes down to ≈ 4×5×3 + 16 ≈ 75 per sensor. Randomize hole
order to guard against drift.

## Localization (well-posed with 3-4 coils)

For a single point force the unknowns are (x, y, F) — 3 numbers. With 3-4 coil
readings the system is exactly-/over-determined → least-squares position with an
uncertainty estimate. Using the **ratio** of two coils cancels `g(F)` and the
force magnitude:

    Δf_i / Δf_j = S_i(x,y) / S_j(x,y)   →   depends on position only

so you localize **without knowing how hard the press was**, using just the
single-reference-force maps `S_k`.

**Conditioning / coil-diversity criterion:** localization is well-posed only if
the coils' spatial maps `S_k(x,y)` are sufficiently *different* (well-conditioned
Jacobian). Two near-identical coil maps → ill-posed however sensitive they are.
This diversity is invisible under Phase 1's uniform pressure and is the real
reason to judge the coil-configuration winner **here**, and to carry 2-3 configs
into this phase rather than locking to one earlier.

## Also here: mirror-winding coil mini-test

The 2 mirror-winding coil variants excluded from the main DOE get a small
**paired comparison** against the winning configuration (same stack, same
protocol, a handful of builds) — appended to this phase.

## Deliverables (when built)

- `analysis/position_design.py` — space-filling/maximin measured set + held-out
  validation set for any grid & budget; anchor-hole selection; randomized pin
  run-sheet.
- `analysis/position_map.py` — fit S_k(x,y) per coil (spline/kriging), map plots,
  interpolation-error report on held-out holes.
- `analysis/localize.py` — invert the multi-coil readings to (x,y[,F]); report
  localization accuracy and the conditioning of each coil-config set.
