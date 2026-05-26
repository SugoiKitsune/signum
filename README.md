<h1>
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/SugoiKitsune/signum/main/assets/logo_signum_white.svg">
    <img src="https://raw.githubusercontent.com/SugoiKitsune/signum/main/assets/logo_signum_black.svg" width="32" height="32" alt="Signum" style="vertical-align: middle;">
  </picture>
  Signum
</h1>

Financial charting for Python — **Jupyter**, **Dash**, **Streamlit**, standalone HTML.  
Inspired by Lightweight Charts.

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
- **Themes** — `dark` (default), `light`, `ft`, `midnight`
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
