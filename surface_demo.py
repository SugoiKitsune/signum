"""Surface3D PoC demo — synthetic implied-vol surface.

Run:  python surface_demo.py   ->  writes surface_demo.html (open in a browser).
In Jupyter:  Surface3D(...).surface(...).show()
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))
from signum.engine.surface3d import Surface3D  # noqa: E402


def synthetic_iv_surface():
    """A plausible equity IV surface: smile + skew + decaying term structure."""
    ttm = np.linspace(0.05, 2.0, 28)          # x: time to maturity (years)
    moneyness = np.linspace(0.7, 1.3, 28)     # y: K / S
    TT, KK = np.meshgrid(ttm, moneyness)      # shape (ny, nx)
    atm = 0.18 + 0.05 * np.exp(-1.6 * TT)     # ATM term structure
    skew = -0.07 * (KK - 1.0)                 # equity negative skew
    smile = 0.30 * (KK - 1.0) ** 2 / np.sqrt(TT + 0.10)
    iv = atm + skew + smile
    return ttm, moneyness, iv


if __name__ == "__main__":
    ttm, moneyness, iv = synthetic_iv_surface()
    chart = Surface3D(theme="midnight", height=560,
                      title="Synthetic equity IV surface", colorscale="viridis")
    chart.surface(
        ttm, moneyness, iv,
        x_label="TTM (yrs)", y_label="Moneyness K/S", z_label="Implied vol",
        wireframe=True, shading="color",
    )
    out = Path(__file__).parent / "surface_demo.html"
    chart.save(str(out))
    print(f"wrote {out}  ({len(chart.render())} bytes)")
