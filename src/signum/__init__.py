"""Signum — professional financial charting for Python.

Distribution: ``pip install signum-charts``  ·  Import: ``import signum``

Renders interactive, Lightweight-Charts-style visuals that display inline in
**Jupyter**, embed in **Dash** / **Streamlit** apps, or export to **standalone
HTML** — all from the same fluent API.

Public API
----------
``Chart``      Price/series charts: candlestick, bar, line, area, baseline,
               histogram, volume, allocation; plus annotations (price_line,
               hline, marker, signals, shade, watermark). Methods return
               ``self`` for chaining.
``StatChart``  Statistical panels: distribution (histogram + KDE), scatter,
               curve (fitted line + confidence band + date slider), spread.
``Dashboard``  Stack multiple ``Chart`` panes with synced crosshair/zoom/scroll.
``sfera`` / ``SferaData``  Data helpers.
``THEMES`` / ``THEME_NAMES`` / ``resolve_theme``  Theme palettes + validation.
``set_execution``  Global default execution lag for ``threshold_control()``.

Themes (pass as ``theme=``): see ``THEME_NAMES`` —
dark (default), light, ft, midnight, rome, glass.

Output methods on every chart: ``.show()`` (Jupyter), ``.save(path)`` (HTML),
``.to_dash(id=...)``, ``.to_streamlit()``, ``.render()`` (raw HTML string).

Quick start
-----------
    import signum
    import yfinance as yf

    df = yf.download("AAPL", period="1y", auto_adjust=True).reset_index()
    (signum.Chart(theme="dark", height=450)
        .candlestick(df)
        .line(sma_df, name="SMA 50")
        .volume(df)
        .show())

For a full, machine-readable API reference see ``llms.txt`` in the repository.
"""

from .engine.chart import Chart
from .engine.dashboard import Dashboard
from .engine.statchart import StatChart
from .engine.sfera import SferaData, sfera
from .engine.themes import THEMES, THEME_NAMES, resolve_theme

_default_execution = 1   # global default for threshold_control()


def set_execution(mode):
    """Set the global default execution mode for threshold_control() calls.

    ``0`` — same bar (lookahead) | ``1`` — next close / MOC (default) |
    ``N`` — lag N bars | ``'NO'`` — next open / MOO (requires open_returns)
    """
    global _default_execution
    _default_execution = mode


__all__ = [
    "Chart",
    "Dashboard",
    "StatChart",
    "SferaData",
    "sfera",
    "set_execution",
    "THEMES",
    "THEME_NAMES",
    "resolve_theme",
]
__version__ = "0.40.0"
