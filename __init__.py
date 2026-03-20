"""Signum - Professional financial charting, inspired by Lightweight Charts.

Works seamlessly in Jupyter notebooks, Dash apps, Streamlit dashboards, and standalone HTML.

Usage:
    from signum import Chart

    chart = Chart(theme="dark")
    chart.candlestick(df).line(sma_df, name="SMA 50").volume(df)
    chart.show()  # Jupyter
    chart.save("chart.html")  # Standalone
    chart.to_dash(id="chart")  # Dash component
"""

from .engine.chart import Chart
from .engine.dashboard import Dashboard
from .engine.statchart import StatChart

__all__ = ["Chart", "Dashboard", "StatChart"]
__version__ = "0.3.0"
