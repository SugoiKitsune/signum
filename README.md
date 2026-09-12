<h1>
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/SugoiKitsune/signum/main/assets/logo_signum_white.svg">
    <img src="https://raw.githubusercontent.com/SugoiKitsune/signum/main/assets/logo_signum_black.svg" width="32" height="32" alt="Signum" style="vertical-align: middle;">
  </picture>
  Signum
</h1>

Financial charting for Python — **Jupyter**, **Dash**, **Streamlit**, standalone HTML.  
Inspired by Lightweight Charts.

<p align="center">
  <img src="https://raw.githubusercontent.com/SugoiKitsune/signum/main/assets/hero.png" alt="Signum — interactive backtest dashboard with live threshold slider, signal, and equity" width="900">
</p>

---

## Install

```bash
pip install signum-charts          # core (pandas + numpy)
pip install "signum-charts[all]"   # + yfinance, dash, streamlit
```

> The distribution is **`signum-charts`** on PyPI; you still `import signum` in code.

## Quick Start

```python
from signum import Chart
import yfinance as yf

df = yf.download("AAPL", period="1y", auto_adjust=True).reset_index()
chart = Chart(theme="dark", height=450).candlestick(df).volume(df)
chart  # renders inline in Jupyter
```

## Features

- **Series** — `candlestick`, `bar`, `line`, `area`, `baseline`, `histogram`, `volume`, `allocation`, `seasonality` (years overlaid on one Jan–Dec axis, with a season brush), `projection_cone` (bootstrap cone past an embargo, realized path over it, live embargo slider)
- **Annotations** — `price_line`, `hline`, `marker`, `signals`, `shade`, `set_watermark`
- **StatChart** — `distribution` (histogram + KDE), `scatter`, `curve` (fitted line + confidence band + date slider), `spread`
- **Themes** — `dark` (default), `light`, `ft`, `midnight`, `rome`, `glass`, `notion-dark`, `notion-light`
- **Dashboard** — multi-pane sync (crosshair, zoom, scroll)
- **Output** — `.show()`, `.save()`, `.to_dash()`, `.to_streamlit()`, `.render()`

All series/annotation methods return `self` for **fluent chaining**:

```python
chart = (
    Chart(theme="ft", height=500, watermark="AAPL")
    .candlestick(df)
    .line(sma_20, name="SMA 20", color="#FF6D00", width=1)
    .volume(df)
    .signals(df, signal_col="signal")
    .shade(df, position_col="position")
    .price_line(220, title="Target", color="#FF9800")
)
```

## Portfolio Allocation

Visualize portfolio allocations across multiple assets without stacking:

```python
# DataFrame with allocation percentages per asset
allocation_df = pd.DataFrame({
    'Date': dates,
    'AAPL': [50, 30, 0, 60],
    'GOOGL': [30, 40, 100, 20],
    'MSFT': [20, 30, 0, 20],
})

chart = (
    Chart(theme="dark", height=400)
    .allocation(allocation_df, allocation_cols=['AAPL', 'GOOGL', 'MSFT'])
    .hline(50, label="50%", style=2)  # Add reference line
)
```

## Dashboard

```python
from signum import Chart, Dashboard

dash = Dashboard(
    panes=[
        Chart(height=300).candlestick(df).volume(df),
        Chart(height=150).baseline(signal_df, base_value=0),
        Chart(height=180).area(equity_df, name="Equity"),
    ],
    titles=["Price", "Signal", "Equity"],
)
```

## Statistical Charts

`StatChart` renders distributions, scatter, and fitted **curves** on the same Canvas
pipeline (Jupyter / Dash / Streamlit / HTML).

```python
from signum import StatChart

# Term-structure / yield curve: dots + smoothed line + 95% band + a 2nd curve
StatChart(theme="light", height=420, title="Sovereign curve").curve(
    ttm_grid,                       # x grid (e.g. time to maturity)
    mean=gp_mean, lower=lo, upper=hi,   # fitted line + confidence band
    points=(bond_ttm, bond_yield),  # observed dots
    prior=nss_curve,                # optional second curve (dashed)
).show()
```

Pass per-date `frames={date: {"mean": ..., "lower": ..., "upper": ..., "prior": ...,
"points": (x, y)}}` to get a **date slider** that morphs the curve, pin a `base=` date
as a dashed ghost, and chain `.spread(grid, frames=..., base=...)` for a linked
`active − base` difference panel below — all driven by the one slider. See the
**Yield Curve** section of [`demo.ipynb`](demo.ipynb).

## 3-D Surfaces (experimental)

`Surface3D` is a rotatable WebGL surface for vol surfaces and term-structure /
continuous-time models, themed from the same `THEMES` as everything else. It is a
proof of concept: importable, but **not yet exported from `signum` or folded into
`StatChart`**.

```python
from signum.engine.surface3d import Surface3D

Surface3D(theme="midnight", height=560, title="IV surface",
          colorscale="viridis", auto_rotate=False).surface(
    ttm, moneyness, iv,                       # x (nx), y (ny), z (ny×nx) or DataFrame
    x_label="TTM (yrs)", y_label="Moneyness", z_label="Implied vol",
    wireframe=True, shading="color",
).show()        # Jupyter — also .render() / .save(path)
```

- `z` takes a 2-D array `(ny, nx)` or a DataFrame (index → y, columns → x); NaN punches holes.
- `colorscale`: `viridis` / `magma` / `plasma` / `turbo` / `rdylbu`, or an explicit list.
- Drag to rotate · scroll to zoom · hover for labelled x / y / z.
- echarts is vendored and pinned to **5.4.3** — echarts-gl 2.0.9 renders blank on 5.5.x.
- Each inline chart embeds ~2 MB of echarts, so don't commit executed outputs.

Demos live under `support/`: `surface_demo.ipynb` (inline) and
`python support/surface_demo.py` (writes two self-contained HTML files — an implied-vol
surface and a Black-Scholes price surface).

Next steps: fold into `StatChart` as `StatChart.surface(x, y, z)`, mixed 2-D + 3-D
panels in one grid, and dated frames with a slider like `curve()`.
