"""Signum Chart - Professional financial charting, inspired by Lightweight Charts.

Renders publication-quality charts from pandas DataFrames.
Works in Jupyter notebooks (inline), Dash apps, Streamlit, and standalone HTML.
"""

import json
import html as html_module
from pathlib import Path
from typing import Optional, List, Dict, Any

import pandas as pd

from .themes import THEMES

# ── Local JS bundle ───────────────────────────────────────────────────────
_VENDOR_DIR = Path(__file__).resolve().parent.parent / "vendor"
_LC_JS_PATH = _VENDOR_DIR / "signum-charts.js"
_LC_JS_CACHE: Optional[str] = None

# ── Signum logo (64x64 SVG, base64) ──────────────────────────────────────
_LOGO_B64 = (
    "PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA2"
    "NCA2NCIgd2lkdGg9IjY0IiBoZWlnaHQ9IjY0Ij4NCiAgPCEtLSBTaWdudW0gdjI6IEVuaGFu"
    "Y2VkIHNpZ25hbCBwdWxzZSB0aHJvdWdoIGRpYW1vbmQgd2l0aCBpbm5lciBnZW9tZXRyeSAt"
    "LT4NCiAgPCEtLSBPdXRlciBkaWFtb25kIGZyYW1lIC0tPg0KICA8cG9seWdvbiBwb2ludHM9"
    "IjMyLDIgNjIsMzIgMzIsNjIgMiwzMiIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ry"
    "b2tlLXdpZHRoPSIyIi8+DQogIDwhLS0gQ29ybmVyIHRpY2sgbWFya3Mgb24gb3V0ZXIgZGlh"
    "bW9uZCAtLT4NCiAgPGxpbmUgeDE9IjMyIiB5MT0iMiIgeDI9IjMyIiB5Mj0iOCIgc3Ryb2tl"
    "PSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIxLjUiLz4NCiAgPGxpbmUgeDE9IjYyIiB5MT0iMzIi"
    "IHgyPSI1NiIgeTI9IjMyIiBzdHJva2U9IndoaXRlIiBzdHJva2Utd2lkdGg9IjEuNSIvPg0K"
    "ICA8bGluZSB4MT0iMzIiIHkxPSI2MiIgeDI9IjMyIiB5Mj0iNTYiIHN0cm9rZT0id2hpdGUi"
    "IHN0cm9rZS13aWR0aD0iMS41Ii8+DQogIDxsaW5lIHgxPSIyIiB5MT0iMzIiIHgyPSI4IiB5"
    "Mj0iMzIiIHN0cm9rZT0id2hpdGUiIHN0cm9rZS13aWR0aD0iMS41Ii8+DQogIDwhLS0gU2ln"
    "bmFsIHB1bHNlIHdhdmUgKG1vcmUgY29tcGxleCB3YXZlZm9ybSkgLS0+DQogIDxwb2x5bGlu"
    "ZSBwb2ludHM9IjYsMzIgMTYsMzIgMTksMzIgMjIsMjIgMjUsNDAgMjgsMTYgMzIsNDggMzUs"
    "MjAgMzgsMzggNDEsMjggNDQsMzIgNDgsMzIgNTgsMzIiDQogICAgICAgICAgICBmaWxsPSJu"
    "b25lIiBzdHJva2U9IndoaXRlIiBzdHJva2Utd2lkdGg9IjIuMiIgc3Ryb2tlLWxpbmVqb2lu"
    "PSJyb3VuZCIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+DQogIDwhLS0gU21hbGwgZmxhbmtp"
    "bmcgZG90cyAtLT4NCiAgPGNpcmNsZSBjeD0iMTAiIGN5PSIzMiIgcj0iMS4yIiBmaWxsPSJ3"
    "aGl0ZSIvPg0KICA8Y2lyY2xlIGN4PSI1NCIgY3k9IjMyIiByPSIxLjIiIGZpbGw9IndoaXRl"
    "Ii8+DQo8L3N2Zz4NCg=="
)


def _get_lc_js() -> str:
    """Load the bundled LC JS (cached after first read)."""
    global _LC_JS_CACHE
    if _LC_JS_CACHE is None:
        _LC_JS_CACHE = _LC_JS_PATH.read_text(encoding="utf-8")
    return _LC_JS_CACHE


class Chart:
    """Financial chart renderer, inspired by Lightweight Charts.

    Usage:
        chart = Chart(theme="dark")
        chart.candlestick(df).line(sma_df, name="SMA 50").volume(df)
        chart.show()       # Jupyter notebook
        chart.save("out.html")  # Standalone HTML file
        chart.to_dash(id="c")   # Dash component
    """

    LC_VERSION = "5.1.0"

    # ── Init ──────────────────────────────────────────────────────────────

    def __init__(
        self,
        theme: str = "dark",
        width: Optional[int] = None,
        height: int = 400,
        watermark: Optional[str] = None,
        logo: bool = True,
    ):
        self._theme_name = theme.lower()
        self._theme = THEMES.get(self._theme_name, THEMES["dark"])
        self._width = width
        self._height = height
        self._watermark = watermark
        self._logo = logo
        self._series: List[Dict[str, Any]] = []
        self._price_lines: List[Dict[str, Any]] = []
        self._markers: Dict[int, List[Dict[str, Any]]] = {}
        self._line_color_idx = 0

    # ── Data Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _detect_time_col(df: pd.DataFrame) -> str:
        for col in ("time", "Time", "date", "Date", "datetime", "Datetime", "timestamp"):
            if col in df.columns:
                return col
        if isinstance(df.index, pd.DatetimeIndex):
            return "__index__"
        raise ValueError(
            "Cannot detect time column. Use a column named 'time' or 'date', "
            "or set a DatetimeIndex."
        )

    def _prepare_time(self, df: pd.DataFrame) -> pd.DataFrame:
        time_col = self._detect_time_col(df)
        df = df.copy()
        if time_col == "__index__":
            df["time"] = df.index
        elif time_col != "time":
            df["time"] = df[time_col]

        if pd.api.types.is_datetime64_any_dtype(df["time"]):
            df["time"] = df["time"].dt.strftime("%Y-%m-%d")
        else:
            df["time"] = pd.to_datetime(df["time"]).dt.strftime("%Y-%m-%d")
        return df

    @staticmethod
    def _col_ci(df: pd.DataFrame, name: str) -> str:
        """Case-insensitive column lookup."""
        for col in df.columns:
            if col.lower() == name.lower():
                return col
        return name

    @staticmethod
    def _json(obj: Any) -> str:
        """JSON serialize with numpy type handling."""
        def default(o):
            if hasattr(o, "item"):
                return o.item()
            raise TypeError(f"Object of type {type(o)} is not JSON serializable")
        return json.dumps(obj, default=default)

    def _find_value_col(self, df: pd.DataFrame, value_col: Optional[str]) -> str:
        if value_col:
            return value_col
        for name in ("value", "close", "Close", "VALUE", "price", "Price"):
            if name in df.columns:
                return name
        for col in df.columns:
            if col != "time" and pd.api.types.is_numeric_dtype(df[col]):
                return col
        raise ValueError("Cannot detect value column. Pass value_col explicitly.")

    def _next_line_color(self) -> str:
        colors = self._theme.get("line_colors", ["#2962FF"])
        color = colors[self._line_color_idx % len(colors)]
        self._line_color_idx += 1
        return color

    # ── Series Methods ────────────────────────────────────────────────────

    def candlestick(self, df: pd.DataFrame, **options) -> "Chart":
        """Add OHLC candlestick series. DataFrame needs time + open/high/low/close columns."""
        df = self._prepare_time(df)
        o, h, l, c = (self._col_ci(df, x) for x in ("open", "high", "low", "close"))
        data = (
            df[["time", o, h, l, c]]
            .rename(columns={o: "open", h: "high", l: "low", c: "close"})
            .dropna(subset=["open", "high", "low", "close"])
            .to_dict("records")
        )
        series_opts = {**self._theme.get("candlestick", {}), **options}
        self._series.append({"type": "CandlestickSeries", "data": data, "options": series_opts})
        return self

    def line(
        self,
        df: pd.DataFrame,
        name: Optional[str] = None,
        color: Optional[str] = None,
        value_col: Optional[str] = None,
        width: int = 2,
        **options,
    ) -> "Chart":
        """Add line series. Automatically detects value/close column."""
        df = self._prepare_time(df)
        vcol = self._find_value_col(df, value_col)
        data = (
            df[["time", vcol]]
            .rename(columns={vcol: "value"})
            .dropna(subset=["value"])
            .to_dict("records")
        )
        series_opts = {**options}
        series_opts["color"] = color or self._next_line_color()
        series_opts["lineWidth"] = width
        if name:
            series_opts["title"] = name
        self._series.append({"type": "LineSeries", "data": data, "options": series_opts})
        return self

    def area(
        self,
        df: pd.DataFrame,
        name: Optional[str] = None,
        value_col: Optional[str] = None,
        **options,
    ) -> "Chart":
        """Add area series (filled line chart)."""
        df = self._prepare_time(df)
        vcol = self._find_value_col(df, value_col)
        data = (
            df[["time", vcol]]
            .rename(columns={vcol: "value"})
            .dropna(subset=["value"])
            .to_dict("records")
        )
        series_opts = {**self._theme.get("area", {}), **options}
        if name:
            series_opts["title"] = name
        self._series.append({"type": "AreaSeries", "data": data, "options": series_opts})
        return self

    def histogram(
        self,
        df: pd.DataFrame,
        name: Optional[str] = None,
        value_col: Optional[str] = None,
        color: Optional[str] = None,
        **options,
    ) -> "Chart":
        """Add histogram series."""
        df = self._prepare_time(df)
        vcol = self._find_value_col(df, value_col)
        data = (
            df[["time", vcol]]
            .rename(columns={vcol: "value"})
            .dropna(subset=["value"])
            .to_dict("records")
        )
        series_opts = {**self._theme.get("histogram", {}), **options}
        if color:
            series_opts["color"] = color
        if name:
            series_opts["title"] = name
        self._series.append({"type": "HistogramSeries", "data": data, "options": series_opts})
        return self

    def volume(self, df: pd.DataFrame, **options) -> "Chart":
        """Add volume histogram overlaid at the bottom 20% of the chart."""
        df = self._prepare_time(df)
        vcol = self._col_ci(df, "volume")
        ocol = self._col_ci(df, "open")
        ccol = self._col_ci(df, "close")

        vol_theme = self._theme.get("volume", {})
        up_color = vol_theme.get("upColor", "rgba(38, 166, 154, 0.5)")
        down_color = vol_theme.get("downColor", "rgba(239, 83, 80, 0.5)")

        mask = df[[vcol, ocol, ccol]].notna().all(axis=1)
        df_clean = df[mask]
        colors = [up_color if c >= o else down_color for c, o in zip(df_clean[ccol], df_clean[ocol])]
        records = [
            {"time": t, "value": float(v), "color": clr}
            for t, v, clr in zip(df_clean["time"], df_clean[vcol], colors)
        ]

        series_opts = {"priceFormat": {"type": "volume"}, "priceScaleId": "volume", **options}
        self._series.append({
            "type": "HistogramSeries",
            "data": records,
            "options": series_opts,
            "price_scale": {"id": "volume", "scaleMargins": {"top": 0.8, "bottom": 0}},
        })
        return self

    def baseline(
        self,
        df: pd.DataFrame,
        base_value: float = 0,
        value_col: Optional[str] = None,
        **options,
    ) -> "Chart":
        """Add baseline series (green above / red below a base value)."""
        df = self._prepare_time(df)
        vcol = self._find_value_col(df, value_col)
        data = (
            df[["time", vcol]]
            .rename(columns={vcol: "value"})
            .dropna(subset=["value"])
            .to_dict("records")
        )
        series_opts = {
            "baseValue": {"type": "price", "price": base_value},
            **self._theme.get("baseline", {}),
            **options,
        }
        self._series.append({"type": "BaselineSeries", "data": data, "options": series_opts})
        return self

    # ── Annotations ───────────────────────────────────────────────────────

    def signals(
        self,
        df: pd.DataFrame,
        signal_col: str = "signal",
        buy_text: str = "BUY",
        sell_text: str = "SELL",
        buy_color: Optional[str] = None,
        sell_color: Optional[str] = None,
        series_index: int = 0,
    ) -> "Chart":
        """Map a signal column (1 = buy, -1 = sell) to chart markers.

        Expects a DataFrame with a time/date column and a signal column
        containing 1 (buy), -1 (sell), or 0/NaN (no signal).

        Usage:
            chart.candlestick(df).signals(df, signal_col="signal")
        """
        df = self._prepare_time(df)
        up_clr = buy_color or self._theme.get("candlestick", {}).get("upColor", "#26a69a")
        dn_clr = sell_color or self._theme.get("candlestick", {}).get("downColor", "#ef5350")
        for _, row in df.iterrows():
            sig = row.get(signal_col, 0)
            if pd.isna(sig) or sig == 0:
                continue
            sig = int(sig)
            self._markers.setdefault(series_index, []).append({
                "time": row["time"],
                "position": "belowBar" if sig == 1 else "aboveBar",
                "shape": "arrowUp" if sig == 1 else "arrowDown",
                "color": up_clr if sig == 1 else dn_clr,
                "text": buy_text if sig == 1 else sell_text,
            })
        return self

    def price_line(
        self,
        price: float,
        title: str = "",
        color: Optional[str] = None,
        line_width: int = 1,
        line_style: int = 2,
        series_index: int = 0,
    ) -> "Chart":
        """Add a horizontal price line to a series."""
        self._price_lines.append({
            "series_index": series_index,
            "options": {
                "price": price,
                "title": title,
                "color": color or self._theme.get("line", {}).get("color", "#2962FF"),
                "lineWidth": line_width,
                "lineStyle": line_style,
                "axisLabelVisible": True,
            },
        })
        return self

    def marker(
        self,
        time: str,
        text: str = "",
        position: str = "aboveBar",
        shape: str = "circle",
        color: Optional[str] = None,
        series_index: int = 0,
    ) -> "Chart":
        """Add a marker to a series at a specific time."""
        self._markers.setdefault(series_index, []).append({
            "time": time,
            "position": position,
            "shape": shape,
            "color": color or self._theme.get("line", {}).get("color", "#2962FF"),
            "text": text,
        })
        return self

    def set_watermark(self, text: str) -> "Chart":
        """Set a watermark text on the chart background."""
        self._watermark = text
        return self

    def shade(
        self,
        df: pd.DataFrame,
        position_col: str = "position",
        color: Optional[str] = None,
        opacity: float = 0.08,
    ) -> "Chart":
        """Shade the chart background during active-position periods.

        Expects a DataFrame with a time column and a position column where
        non-zero values indicate the strategy is "in the deal".
        Renders as a transparent area series spanning the full price range.

        Usage:
            chart.candlestick(df).shade(df, position_col="position")
        """
        df = self._prepare_time(df)
        pcol = position_col
        if pcol not in df.columns:
            raise ValueError(f"Column '{pcol}' not found in DataFrame")

        # Parse color + apply opacity
        base_color = color or self._theme.get("candlestick", {}).get("upColor", "#26a69a")
        hex_c = base_color.lstrip("#")
        r, g, b = int(hex_c[0:2], 16), int(hex_c[2:4], 16), int(hex_c[4:6], 16)
        fill = f"rgba({r},{g},{b},{opacity})"
        line_clr = f"rgba({r},{g},{b},{opacity * 2})"

        # Build data: value=1 when in position, 0 when not
        records = []
        for _, row in df.iterrows():
            val = row.get(pcol, 0)
            in_pos = 0 if (pd.isna(val) or val == 0) else 1
            records.append({"time": row["time"], "value": in_pos})

        series_opts = {
            "priceScaleId": "_shade",
            "lineWidth": 1,
            "lineColor": "rgba(0,0,0,0)",
            "lineType": 2,
            "topColor": fill,
            "bottomColor": "transparent",
            "crosshairMarkerVisible": False,
            "crosshairMarkerRadius": 0,
            "pointMarkersVisible": False,
            "lastValueVisible": False,
            "priceLineVisible": False,
        }
        self._series.append({
            "type": "AreaSeries",
            "data": records,
            "options": series_opts,
            "price_scale": {
                "id": "_shade",
                "scaleMargins": {"top": 0, "bottom": 0},
            },
        })
        return self

    # ── Build HTML ────────────────────────────────────────────────────────

    def _build_chart_options(self) -> dict:
        opts = {**self._theme.get("chart", {})}
        if self._width:
            opts["width"] = self._width
        opts["height"] = self._height
        if not self._width:
            opts["autoSize"] = True

        # Disable default attribution logo
        layout = opts.get("layout", {})
        layout["attributionLogo"] = False

        # If theme has CSS texture background, make canvas transparent
        if self._theme.get("background_css"):
            layout["background"] = {"type": "solid", "color": "transparent"}

        opts["layout"] = layout

        if self._watermark:
            is_dark = self._theme_name in ("dark", "midnight", "distfit")
            opts["watermark"] = {
                "visible": True,
                "text": self._watermark,
                "fontSize": 48,
                "color": "rgba(255,255,255,0.07)" if is_dark else "rgba(0,0,0,0.07)",
            }
        return opts

    def _build_series_js(self, var_prefix: str = "", chart_var: str = "chart") -> str:
        lines = []
        for i, s in enumerate(self._series):
            var = f"{var_prefix}s{i}"
            lines.append(
                f"const {var} = {chart_var}.addSeries(LightweightCharts.{s['type']}, "
                f"{self._json(s['options'])});"
            )
            lines.append(f"{var}.setData({self._json(s['data'])});")

            if "price_scale" in s:
                ps = s["price_scale"]
                ps_opts = {"scaleMargins": ps["scaleMargins"]}
                if ps["id"].startswith("_"):
                    ps_opts["visible"] = False
                lines.append(
                    f"{chart_var}.priceScale('{ps['id']}').applyOptions("
                    f"{self._json(ps_opts)});"
                )

            for pl in self._price_lines:
                if pl["series_index"] == i:
                    lines.append(f"{var}.createPriceLine({self._json(pl['options'])});")

            if i in self._markers:
                sorted_markers = sorted(self._markers[i], key=lambda m: m["time"])
                lines.append(
                    f"LightweightCharts.createSeriesMarkers({var}, "
                    f"{self._json(sorted_markers)});"
                )

        return "\n        ".join(lines)

    def _build_html(self) -> str:
        chart_opts = self._json(self._build_chart_options())
        series_js = self._build_series_js()
        bg = (
            self._theme.get("chart", {})
            .get("layout", {})
            .get("background", {})
            .get("color", "#1e1e1e")
        )
        width_css = "100%" if not self._width else f"{self._width}px"
        lc_js = _get_lc_js()

        # Theme may override body background with CSS marble/texture
        custom_bg_css = self._theme.get("background_css", "")
        bg_css = custom_bg_css if custom_bg_css else f"background:{bg};"

        # SVG marble texture (rendered behind the canvas)
        bg_svg = self._theme.get("background_svg", "")

        # Detect light background → invert the white SVG logo to black
        _logo_invert = ""
        if bg.startswith("#") and len(bg) >= 7:
            _bg_hex = bg.lstrip("#")
            _r, _g, _b = (int(_bg_hex[i:i+2], 16) for i in (0, 2, 4))
            if _r * 0.299 + _g * 0.587 + _b * 0.114 > 150:
                _logo_invert = "filter:invert(1);"
        elif custom_bg_css or bg_svg:
            # Only invert if the theme is actually light (distfit is dark with a CSS gradient)
            if self._theme_name not in ("dark", "midnight", "distfit"):
                _logo_invert = "filter:invert(1);"

        return f"""<!DOCTYPE html>
<html><head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<script>{lc_js}</script>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{{bg_css}overflow:hidden;position:relative;border-radius:12px}}
#fc{{width:{width_css};height:{self._height}px;position:relative;z-index:1}}
#err{{position:absolute;top:0;left:0;color:red;font-size:11px;z-index:9999;padding:4px;background:rgba(0,0,0,0.9);display:none}}
#signum-logo{{position:absolute;left:12px;bottom:-20px;z-index:5;opacity:0.6;pointer-events:none;{_logo_invert}}}
</style>
</head><body>
{bg_svg}
<div id="fc"></div>
<div id="err"></div>
{'<img id="signum-logo" src="data:image/svg+xml;base64,' + _LOGO_B64 + '" width="24" height="24" alt="Signum">' if self._logo else ''}
<script>
try {{
    const chart = LightweightCharts.createChart(document.getElementById('fc'), {chart_opts});
    {series_js}
    chart.timeScale().fitContent();
    window.addEventListener('resize', () => chart.timeScale().fitContent());
}} catch(e) {{
    var el = document.getElementById('err');
    el.style.display = 'block';
    el.innerText = 'LC ERROR: ' + e.message;
}}
</script>
</body></html>"""

    # ── Display Methods ───────────────────────────────────────────────────

    def _repr_html_(self) -> str:
        """Jupyter notebook inline display via iframe + blob URL."""
        import base64
        chart_html = self._build_html()
        b64 = base64.b64encode(chart_html.encode("utf-8")).decode("ascii")
        h = self._height + 30
        uid = f"fc{id(self)}"
        return (
            f'<div id="{uid}" style="width:100%;height:{h}px;border-radius:12px;overflow:hidden;">'
            f'</div><script>'
            f'(function(){{'
            f'var a=atob("{b64}"),b=new Uint8Array(a.length);'
            f'for(var i=0;i<a.length;i++)b[i]=a.charCodeAt(i);'
            f'var blob=new Blob([b],{{type:"text/html;charset=utf-8"}});'
            f'var url=URL.createObjectURL(blob);'
            f'var f=document.createElement("iframe");'
            f'f.src=url;f.style.width="100%";f.style.height="{h}px";'
            f'f.style.border="none";f.style.borderRadius="12px";'
            f'document.getElementById("{uid}").appendChild(f);'
            f'}})();'
            f'</script>'
        )

    def show(self):
        """Display the chart in a Jupyter notebook."""
        try:
            from IPython.display import display, HTML
            display(HTML(self._repr_html_()))
        except ImportError:
            print("IPython not available. Use .save() or .render() instead.")

    def render(self) -> str:
        """Return the chart as a standalone HTML string."""
        return self._build_html()

    def save(self, path: str):
        """Save the chart as a standalone HTML file."""
        with open(path, "w", encoding="utf-8") as f:
            f.write(self._build_html())

    def to_dash(self, id: Optional[str] = None, style: Optional[dict] = None):
        """Return a Dash html.Iframe component containing the chart.

        Usage in Dash:
            app.layout = html.Div([chart.to_dash(id="my-chart")])
        """
        from dash import html

        default_style = {
            "width": "100%",
            "height": f"{self._height + 30}px",
            "border": "none",
            "borderRadius": "4px",
        }
        if style:
            default_style.update(style)

        return html.Iframe(
            id=id or "forge-chart",
            srcDoc=self._build_html(),
            style=default_style,
        )

    def to_streamlit(self, height: Optional[int] = None):
        """Render the chart in a Streamlit app.

        Usage:
            chart.to_streamlit()
        """
        import streamlit.components.v1 as components
        components.html(self._build_html(), height=height or self._height + 30)
