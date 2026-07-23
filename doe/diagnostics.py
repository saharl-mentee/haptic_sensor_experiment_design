"""
diagnostics.py — PRE-TEST RISK CHECK for a chosen design.

Two things you can inspect BEFORE building a single sensor — they depend only
on the RUN LIST, not on any measured data:

  * CORRELATION MAP  — how entangled the effects you WILL fit are with each
      other. 0 = cleanly separable; near +/-1 = the design can barely tell two
      effects apart. (Saved as a colour heat-map.)

  * ALIAS MATRIX     — the sneaky risk. Effects you are NOT fitting (e.g.
      interactions you left out of the model) do not vanish: if they are real,
      they LEAK into the coefficients you DO estimate and bias them. The alias
      matrix shows which fitted effect absorbs which left-out interaction:
          entry ~ 0  -> safe   (that interaction won't bias this effect)
          entry ~ 1  -> fully aliased (you literally cannot separate them)
          in between -> partial bias of that size, per unit of the true effect
      Formula: A = (X1' X1)^-1 X1' X2, with X1 = terms you fit, X2 = terms you
      left out. Row = a fitted effect, column = a left-out interaction.

Both use the same EFFECT coding as the rest of the toolkit, so the numbers
match the scorecard's VIF / max|corr|.

Run:  python diagnostics.py     (edit the CONFIG block first)
Outputs -> ../results/diagnostics/
"""

import os
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from factors import FACTORS, INTERACTIONS
from model_matrix import build_model_matrix, _effect_col
from optimal import d_optimal
import report

# ======================== CONFIG (edit me) ========================
# Which design to risk-check? Either point at a saved run-list CSV...
DESIGN_CSV      = "results/augmented/designs/base2rep_N20_[ME].csv"       # e.g. "../results/augmented/designs/all-new_(10)_N28_int.csv"
# ...or (if DESIGN_CSV is "") build one fresh here:
N               = 28
FIT_INTERACTION = False     # the model you INTEND to fit after the experiment
SEED            = 50
# ==================================================================

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "results", "diagnostics")
os.makedirs(OUT, exist_ok=True)

# data-viz palette (diverging blue <-> gray <-> red, per the project standard)
SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#898781"
_CMAP = LinearSegmentedColormap.from_list(
    "corr", ["#2a78d6", "#f0efec", "#e34948"])   # -1 blue . 0 gray . +1 red


def _interaction_columns(design: pd.DataFrame, pairs):
    """Effect-coded columns for a list of factor pairs (product of parents)."""
    cols, names = [], []
    for fa, fb in pairs:
        refa, refb = FACTORS[fa][0], FACTORS[fb][0]
        for la in FACTORS[fa][1:]:
            ca = _effect_col(design, fa, la, refa)
            for lb in FACTORS[fb][1:]:
                cb = _effect_col(design, fb, lb, refb)
                cols.append(ca * cb)
                names.append(f"{fa}[{la}]:{fb}[{lb}]")
    if cols:
        return np.column_stack(cols), names
    return np.empty((len(design), 0)), []


def correlation_matrix(design: pd.DataFrame, include_interaction: bool) -> pd.DataFrame:
    """Correlation among the fitted-model effect columns (intercept dropped)."""
    X, names = build_model_matrix(design, include_interaction)
    X, names = X[:, 1:], names[1:]                 # drop intercept
    Xc = X - X.mean(axis=0)
    norms = np.linalg.norm(Xc, axis=0)
    norms[norms == 0] = 1.0
    Z = Xc / norms
    R = Z.T @ Z
    return pd.DataFrame(R, index=names, columns=names)


def alias_matrix(design: pd.DataFrame, fit_interaction: bool) -> pd.DataFrame:
    """
    A = (X1'X1)^-1 X1'X2.  X1 = the model you FIT; X2 = every 2-factor
    interaction you do NOT fit (the effects that could secretly bias you).
    """
    X1, n1 = build_model_matrix(design, fit_interaction)
    fitted = {frozenset(p) for p in (INTERACTIONS if fit_interaction else [])}
    alias_pairs = [p for p in combinations(FACTORS, 2)
                   if frozenset(p) not in fitted]
    X2, n2 = _interaction_columns(design, alias_pairs)
    # lstsq(X1, X2) returns exactly (X1'X1)^-1 X1'X2, robust if near-singular
    A, *_ = np.linalg.lstsq(X1, X2, rcond=None)
    return pd.DataFrame(A, index=n1, columns=n2)


def heatmap(M: pd.DataFrame, title: str, path: str, annotate=True):
    """Diverging colour map of a matrix, centred at 0, saved to path."""
    data = M.to_numpy(float)
    vmax = max(np.abs(data).max(), 1e-9)
    nrow, ncol = data.shape
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * ncol + 3),
                                    max(4, 0.5 * nrow + 2)), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    im = ax.imshow(data, cmap=_CMAP, norm=norm, aspect="auto")

    ax.set_xticks(range(ncol)); ax.set_yticks(range(nrow))
    ax.set_xticklabels(M.columns, rotation=90, fontsize=7, color=INK)
    ax.set_yticklabels(M.index, fontsize=7, color=INK)
    ax.set_title(title, color=INK, fontsize=11, loc="left", pad=10)

    if annotate and nrow * ncol <= 400:
        for i in range(nrow):
            for j in range(ncol):
                v = data[i, j]
                txt = "0" if abs(v) < 0.005 else f"{v:.2f}".lstrip("0").replace("-0.", "-.")
                ax.text(j, i, txt, ha="center", va="center", fontsize=6,
                        color="white" if abs(v) > 0.55 * vmax else INK)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors=MUTED, labelsize=7)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    # -- get the design -------------------------------------------------
    if DESIGN_CSV:
        design = pd.read_csv(DESIGN_CSV, dtype=str)
        src = DESIGN_CSV
    else:
        design = d_optimal(N, include_interaction=FIT_INTERACTION, seed=SEED)
        src = f"freshly built D-optimal N={N}"
    model_label = "main effects + interactions" if FIT_INTERACTION else "main effects"

    report.banner("PRE-TEST RISK CHECK")
    print(f"Design : {src}  ({len(design)} runs)")
    print(f"Model you intend to fit: {model_label}\n")

    # -- correlation map ------------------------------------------------
    R = correlation_matrix(design, FIT_INTERACTION)
    R.to_csv(os.path.join(OUT, "correlation.csv"))
    off = R.to_numpy().copy(); np.fill_diagonal(off, 0.0)
    i, j = np.unravel_index(np.abs(off).argmax(), off.shape)
    print("CORRELATION MAP (among the effects you will fit)")
    print(f"  largest off-diagonal |correlation| = {abs(off[i, j]):.2f}  "
          f"between  {R.index[i]}  &  {R.columns[j]}")
    same = R.index[i].split("[")[0] == R.columns[j].split("[")[0]
    print(f"  -> {'SAME factor (structural, harmless)' if same else 'DIFFERENT factors — a real entanglement to note'}")
    heatmap(R, f"Correlation of fitted effects  ({model_label}, N={len(design)})",
            os.path.join(OUT, "correlation_map.png"))

    # -- alias matrix ---------------------------------------------------
    A = alias_matrix(design, FIT_INTERACTION)
    A.to_csv(os.path.join(OUT, "alias.csv"))
    print("\nALIAS MATRIX (left-out interactions -> bias on fitted effects)")
    if A.shape[1] == 0:
        print("  You are fitting every 2-factor interaction — nothing left out.")
    else:
        worst = np.abs(A.to_numpy())
        # ignore the intercept row for the headline
        rows = [k for k in range(A.shape[0]) if A.index[k] != "intercept"]
        sub = np.abs(A.to_numpy()[rows])
        r, c = np.unravel_index(sub.argmax(), sub.shape)
        print(f"  worst aliasing |A| = {sub[r, c]:.2f}  :  a real "
              f"'{A.columns[c]}'\n    would bias your estimate of "
              f"'{A.index[rows[r]]}' by {sub[r, c]:.2f} per unit.")
        n_bad = int((sub > 0.5).sum())
        print(f"  cells with |bias| > 0.50 (serious confounding): {n_bad}")
        print("  (0 = safe, 1 = inseparable. Small values everywhere = low risk.)")
    heatmap(A, f"Alias matrix: left-out 2FI -> fitted effects  (N={len(design)})",
            os.path.join(OUT, "alias_map.png"), annotate=A.shape[1] <= 30)

    print(f"\nSaved: correlation_map.png, alias_map.png, correlation.csv, alias.csv")
    print(f"  in {os.path.abspath(OUT)}")
