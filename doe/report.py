"""
report.py — shared reporting helpers for the two study scripts
(compare_designs.py and augment_study.py).

Keeping the scorecard, power table, and file-saving in ONE place means
the two studies are always computed and printed the same way, so their
numbers are directly comparable. Neither study script re-implements any
of this — they just call these functions.
"""

import os
import pandas as pd

from metrics import scorecard
from power_sim import power_table

SEP = "=" * 78


def banner(title: str):
    print(f"\n{SEP}\n{title}\n{SEP}")


def slug(name: str) -> str:
    """Filesystem-safe version of a design's display name."""
    return name.replace(" ", "_").replace("+", "").replace("/", "-").replace("=", "")


def save_designs(designs: dict, out_dir: str) -> str:
    """
    Write each design's run list to out_dir/designs/<name>.csv.
    `designs` maps display-name -> (DataFrame, include_interaction).
    """
    dd = os.path.join(out_dir, "designs")
    os.makedirs(dd, exist_ok=True)
    for name, (d, _) in designs.items():
        d.to_csv(os.path.join(dd, f"{slug(name)}.csv"), index=False)
    return dd


def scorecard_table(designs: dict, out_dir: str = None,
                    filename: str = "scorecard.csv") -> pd.DataFrame:
    """One scorecard row per design; optionally saved as CSV."""
    rows = [scorecard(name, d, inter) for name, (d, inter) in designs.items()]
    table = pd.DataFrame(rows)
    if out_dir:
        table.to_csv(os.path.join(out_dir, filename), index=False)
    return table


def power_frame(designs: dict, out_dir: str = None, filename: str = "power.csv",
                n_sims: int = 400, skip=lambda name: False) -> pd.DataFrame:
    """
    Power for every factor, one row per design. `skip(name)` lets a caller
    leave out e.g. the full-factorial reference. Optionally saved as CSV.
    """
    rows = {name: power_table(d, include_interaction=inter, n_sims=n_sims)
            for name, (d, inter) in designs.items() if not skip(name)}
    pf = pd.DataFrame(rows).T
    if out_dir:
        pf.to_csv(os.path.join(out_dir, filename))
    return pf


def print_power(pf: pd.DataFrame):
    """Print a power DataFrame as whole-percent strings."""
    print((pf * 100).round(0).astype(int).astype(str).add("%").to_string())


SCORECARD_LEGEND = """
How to read the scorecard:
  error df    >= 8-10 is comfortable; below that p-values get shaky.
  replicates  duplicate builds -> direct measure of build-to-build noise.
  D-eff %     information per run vs the full factorial (higher = better).
  worst VIF   1.0 = effects perfectly separated; 2 = one effect's
              uncertainty doubled by entanglement; >5 = trouble.
              (+interaction rows show an elevated VIF even for the perfect
              full-factorial design — a coding artifact; compare against
              that reference row, not against 1.)
  max |corr|  0 = fully orthogonal design (the Taguchi ideal)."""
