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

## Quick Start

```python
from signum import Chart
import yfinance as yf

df = yf.download("AAPL", period="1y", auto_adjust=True).reset_index()
chart = Chart(theme="dark", height=450).candlestick(df).volume(df)
chart  # renders inline in Jupyter
```

## Features

- **Series** — `candlestick`, `bar`, `line`, `area`, `baseline`, `histogram`, `volume`, `allocation`
- **Annotations** — `price_line`, `hline`, `marker`, `signals`, `shade`, `set_watermark`
- **StatChart** — `distribution` (histogram + KDE), `scatter`, `curve` (fitted line + confidence band + date slider), `spread`
- **Themes** — `dark` (default), `light`, `ft`, `midnight`, `rome`, `distfit`
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
`active − base` difference panel below — all driven by the one slider. See
[`examples/ofz_curve_demo.py`](examples/ofz_curve_demo.py).
