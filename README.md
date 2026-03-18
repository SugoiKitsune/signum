<h1><img src="assets/logo_signum.svg" width="32" height="32" alt="Signum" style="vertical-align: middle;"> Signum</h1>

Professional financial charting, inspired by Lightweight Charts.  
One chart object — works in **Jupyter**, **Dash**, **Streamlit**, and **standalone HTML**.

---

## Quick Start

```python
from signum import Chart

import yfinance as yf
df = yf.download("AAPL", period="1y", auto_adjust=True).reset_index()

chart = Chart(theme="dark", height=450)
chart.candlestick(df).volume(df)
chart  # renders inline in Jupyter
```

## Themes

| Theme | Description |
|-------|-------------|
| `dark` | Dark background (default) |
| `light` | Light / white background |
| `ft` | Financial Times — cream, Georgia serif, burgundy/teal |
| `midnight` | Deep dark variant |

```python
Chart(theme="ft", height=400, watermark="AAPL").candlestick(df).volume(df)
```

## Chart Methods

### Series

All series methods return `self` for fluent chaining.

```python
chart = Chart(theme="dark", height=450, watermark="AAPL")

# Candlestick (OHLCV DataFrame)
chart.candlestick(df)

# Line overlay (auto-detects value column)
chart.line(sma_df, name="SMA 20", color="#FF6D00", width=1)

# Volume histogram (bottom 20% of chart)
chart.volume(df)

# Area chart (equity curves, NAV)
chart.area(df, value_col="Close", name="AAPL")

# Baseline chart (green above / red below a base value)
chart.baseline(returns_df, base_value=0)

# Histogram
chart.histogram(df, value_col="delta", color="#26a69a")
```

### Annotations

```python
# Horizontal price line
chart.price_line(220.50, title="Target", color="#FF9800")

# Marker at a specific date
chart.marker("2025-06-15", text="Buy", position="belowBar", shape="arrowUp", color="#26a69a")

# Signal column → markers (1 = buy, -1 = sell)
chart.signals(df, signal_col="signal")

# Watermark
chart.set_watermark("AAPL")
```

### Fluent Chaining

```python
chart = (
    Chart(theme="dark", height=500, watermark="AAPL")
    .candlestick(df)
    .line(sma_20, name="SMA 20", color="#FF6D00", width=1)
    .line(sma_50, name="SMA 50", color="#E91E63", width=1)
    .volume(df)
    .price_line(avg_price, title="Average", color="#FF9800")
)
```

## Multi-Pane Dashboard

Vertically stacked charts with synchronized crosshair, zoom, and scroll.

```python
from signum import Chart, Dashboard

price_pane = (
    Chart(height=300)
    .candlestick(df)
    .line(sma_20, name="SMA 20", color="#FF6D00", width=1)
    .volume(df)
)
signal_pane = Chart(height=150).baseline(signal_df, base_value=0)
equity_pane = Chart(height=180).area(equity_df, name="Equity")

dash = Dashboard(
    panes=[price_pane, signal_pane, equity_pane],
    titles=["Price", "Signal", "Equity"],
)
dash  # renders inline in Jupyter
```

## Output Methods

| Method | Where |
|--------|-------|
| `chart` / `chart.show()` | Jupyter notebook (inline iframe) |
| `chart.save("out.html")` | Standalone HTML file |
| `chart.to_dash(id="x")` | Dash app (`html.Iframe`) |
| `chart.to_streamlit()` | Streamlit dashboard |
| `chart.render()` | Raw HTML string |

### Dash

```python
from dash import Dash, html
from signum import Chart

chart = Chart(theme="dark").candlestick(df).volume(df)

app = Dash(__name__)
app.layout = html.Div([chart.to_dash(id="price-chart")])
app.run(debug=True)
```

### Streamlit

```python
import streamlit as st
from signum import Chart

chart = Chart(theme="dark").candlestick(df).volume(df)
st.title("Dashboard")
chart.to_streamlit()
```

## Project Structure

```
signum/
├── __init__.py          # Exports: Chart, Dashboard
├── engine/
│   ├── chart.py         # Chart class
│   ├── dashboard.py     # Dashboard class (multi-pane sync)
│   └── themes.py        # Theme definitions
├── vendor/
│   └── signum-charts.js # Bundled chart engine
├── assets/
│   └── logo_signum.svg  # Signum logo
└── demo.ipynb           # Interactive demo notebook
```
