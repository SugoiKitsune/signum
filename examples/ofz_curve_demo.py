"""Demo: OFZ-style sovereign yield curve in Signum's StatChart.

Reproduces the matplotlib "NSS prior + GP smoothing" chart as a native,
interactive Signum panel — observed bond dots, a smoothed posterior line,
a 95% confidence band, the NSS prior (dashed second curve), a pinned
base-date ghost, AND a date slider that morphs the whole curve through
history.  A linked spread panel below shows (active − base) per maturity.

Run:
    python examples/ofz_curve_demo.py
    # → writes ofz_curve_demo.html, open it in a browser

All data here is SYNTHETIC (NSS + smooth wiggles + noise) — no DB needed.
Swap in your real OFZ curve fit later: pass per-date {mean, lower, upper,
prior, points} frames to StatChart.curve(...).
"""

import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from signum import StatChart


def nss(t, b0, b1, b2, b3, tau1, tau2):
    """Nelson-Siegel-Svensson zero curve."""
    t = np.asarray(t, dtype=float)
    t = np.where(t <= 0, 1e-6, t)
    f1 = (1 - np.exp(-t / tau1)) / (t / tau1)
    f2 = f1 - np.exp(-t / tau1)
    f3 = (1 - np.exp(-t / tau2)) / (t / tau2) - np.exp(-t / tau2)
    return b0 + b1 * f1 + b2 * f2 + b3 * f3


def build_frames():
    rng = np.random.default_rng(7)
    grid = np.linspace(0.25, 15.0, 220)              # ttm grid for the line/band
    bond_ttm = np.sort(rng.uniform(0.3, 14.5, 30))   # observed bond maturities

    # Monthly dates; curve level drifts up over the window (OFZ-like, ~13→14.8%)
    dates = pd.date_range("2023-06-01", "2026-06-01", freq="MS")
    n = len(dates)

    frames = {}
    for k, dt in enumerate(dates):
        w = k / (n - 1)                              # 0 → 1 over time
        # NSS params drift: short end rises, curve flattens/inverts mildly at long end
        b0 = 14.3 + 0.5 * w                          # long-run level
        b1 = -1.4 + 1.0 * w                          # slope (less negative over time)
        b2 = 1.2 - 2.2 * w                           # curvature 1
        b3 = -0.8 + 0.4 * w                          # curvature 2
        prior = nss(grid, b0, b1, b2, b3, 1.6, 5.5)

        # GP posterior mean = prior + smooth, date-shifting wiggles
        wig = (0.18 * np.sin(grid / 1.3 + 1.5 * w * np.pi)
               + 0.10 * np.sin(grid / 3.7 + 0.8 * k))
        mean = prior + wig

        # 95% band: tight where bonds are dense, flaring past the last bond
        edge = np.clip((grid - bond_ttm.max()) / 2.0, 0, None)
        sd = 0.05 + 0.015 * np.sqrt(grid) + 0.55 * edge ** 1.5
        lower, upper = mean - 1.96 * sd, mean + 1.96 * sd

        # Observed dots = GP mean at bond maturities + small idiosyncratic noise
        py = np.interp(bond_ttm, grid, mean) + rng.normal(0, 0.06, size=bond_ttm.shape)

        frames[dt.strftime("%Y-%m-%d")] = {
            "mean": mean, "lower": lower, "upper": upper,
            "prior": prior, "points": (bond_ttm, py),
        }
    return grid, frames, list(frames.keys())


def main():
    grid, frames, dates = build_frames()
    base = dates[0]    # pin the earliest date as the reference ghost; slider
                       # starts on the latest, so the spread shows the full
                       # cumulative curve shift out of the box.

    # Theme-driven colours — mean/prior/spread come straight from the theme
    # palette (try "midnight", "dark", "distfit" too); no hardcoded colours.
    chart = (
        StatChart(theme="ft", height=760, cols=1,
                  title="OFZ sovereign curve — NSS prior + GP smoothing (Signum)")
        .curve(
            grid,
            frames=frames,
            base=base,
            mean_name="GP posterior mean",
            prior_name="NSS prior",
            point_name="OFZ-PD bonds",
            base_name="base date",
            band_label="GP 95% band",
            name="Effective yield (%) vs time to maturity",
            x_label="Time to maturity (years)",
            y_label="Effective yield (%)",
            slider_label="curve date",
        )
        .spread(
            grid,
            frames=frames,
            base=base,
            name="Spread vs base date  (active − base, %)",
            x_label="Time to maturity (years)",
            y_label="Δ yield (%)",
        )
    )

    out = Path(__file__).resolve().parent.parent / "ofz_curve_demo.html"
    chart.save(str(out))
    print(f"Wrote {out}")
    print(f"{len(dates)} dated frames  |  base = {base}  |  grid {grid.min():.2f}–{grid.max():.1f}y")


if __name__ == "__main__":
    main()
