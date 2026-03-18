<h1><img src="assets/logo_signum.svg" width="32" height="32" alt="Signum" style="vertical-align: middle;"> Signum</h1>

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

- **Series** — `candlestick`, `line`, `area`, `baseline`, `histogram`, `volume`
- **Annotations** — `price_line`, `marker`, `signals`, `shade`, `set_watermark`
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
