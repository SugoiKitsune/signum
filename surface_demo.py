"""Surface3D PoC demos — writes self-contained HTML you can open in a browser.

    python surface_demo.py
        -> surface_iv.html   (synthetic equity implied-vol surface)
        -> surface_bs.html   (Black-Scholes call-price surface)

Both files inline the vendored echarts + echarts-gl, so they work offline.
Drag to rotate; scroll to zoom; hover for labelled x/y/z values.
In Jupyter:  Surface3D(...).surface(...).show()
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))
from signum.engine.surface3d import Surface3D  # noqa: E402

_OUT = Path(__file__).parent
_norm_cdf = np.vectorize(lambda x: 0.5 * (1.0 + math.erf(x / math.sqrt(2.0))))


def iv_surface():
    """Plausible equity IV surface: smile + skew + decaying term structure."""
    ttm = np.linspace(0.05, 2.0, 28)          # x: time to maturity (years)
    moneyness = np.linspace(0.7, 1.3, 28)     # y: K / S
    TT, KK = np.meshgrid(ttm, moneyness)      # (ny, nx)
    atm = 0.18 + 0.05 * np.exp(-1.6 * TT)     # ATM term structure
    skew = -0.07 * (KK - 1.0)                 # equity negative skew
    smile = 0.30 * (KK - 1.0) ** 2 / np.sqrt(TT + 0.10)
    iv = atm + skew + smile
    return ttm, moneyness, iv


def bs_call_surface():
    """Black-Scholes call price vs spot (x) and time-to-expiry (y)."""
    spot = np.linspace(60, 140, 32)           # x: underlying spot
    tau = np.linspace(0.02, 1.0, 32)          # y: time to expiry (years)
    K, r, sig = 100.0, 0.02, 0.25
    SS, TT = np.meshgrid(spot, tau)           # (ny, nx)
    d1 = (np.log(SS / K) + (r + 0.5 * sig ** 2) * TT) / (sig * np.sqrt(TT))
    d2 = d1 - sig * np.sqrt(TT)
    price = SS * _norm_cdf(d1) - K * np.exp(-r * TT) * _norm_cdf(d2)
    return spot, tau, price


if __name__ == "__main__":
    x, y, z = iv_surface()
    Surface3D(theme="midnight", height=560, title="Synthetic equity IV surface",
              colorscale="viridis").surface(
        x, y, z, x_label="TTM (yrs)", y_label="Moneyness K/S", z_label="Implied vol",
    ).save(str(_OUT / "surface_iv.html"))

    x, y, z = bs_call_surface()
    Surface3D(theme="distfit", height=560, title="Black-Scholes call price",
              colorscale="turbo").surface(
        x, y, z, x_label="Spot", y_label="TTE (yrs)", z_label="Call price",
    ).save(str(_OUT / "surface_bs.html"))

    print("wrote surface_iv.html and surface_bs.html — open either in a browser")
