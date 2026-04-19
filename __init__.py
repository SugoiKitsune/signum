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
from .engine.sfera import SferaData, sfera

_default_execution = 1   # global default for threshold_control()


def set_execution(mode):
    """Set the global default execution mode for threshold_control() calls.

    ``0`` — same bar (lookahead) | ``1`` — next close / MOC (default) |
    ``N`` — lag N bars | ``'NO'`` — next open / MOO (requires open_returns)
    """
    global _default_execution
    _default_execution = mode


__all__ = ["Chart", "Dashboard", "StatChart", "SferaData", "sfera", "set_execution"]
__version__ = "0.36.0"
