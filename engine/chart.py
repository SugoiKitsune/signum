"""Signum Chart - Professional financial charting, inspired by Lightweight Charts.

Renders publication-quality charts from pandas DataFrames.
Works in Jupyter notebooks (inline), Dash apps, Streamlit, and standalone HTML.
"""

import json
import math
import html as html_module
from pathlib import Path
from typing import Optional, List, Dict, Any

import pandas as pd

from .themes import THEMES
from .logos import LOGO_APEX as _LOGO_B64  # swap to LOGO_DIAMOND to restore the classic logo

# ── Local JS bundle ───────────────────────────────────────────────────────
_VENDOR_DIR = Path(__file__).resolve().parent.parent / "vendor"
_LC_JS_PATH = _VENDOR_DIR / "signum-charts.js"
_LC_JS_CACHE: Optional[str] = None


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
        self._threshold_config: Optional[Dict[str, Any]] = None
        self._smoothing_configs: List[Dict[str, Any]] = []
        self._stats_legend: Optional[Dict[str, Any]] = None
        self._bg_image_config: Optional[Dict] = None

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

    def _prepare_time(self, df) -> pd.DataFrame:
        if isinstance(df, pd.Series):
            df = df.to_frame(df.name or "value")
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
        # LightweightCharts whitespace pattern: NaN rows become {time: ...} only
        # (no 'value' key) — this keeps the full time axis anchored while the
        # line shows a gap.  {value: null} is NOT valid in LWC LineSeries.
        tmp = df[["time", vcol]].rename(columns={vcol: "value"})
        valid = tmp["value"].notna()
        data = sorted(
            tmp[valid].to_dict("records") + [{"time": t} for t in tmp.loc[~valid, "time"]],
            key=lambda r: r["time"],
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
        # LightweightCharts whitespace pattern: NaN rows become {time: ...} only.
        tmp = df[["time", vcol]].rename(columns={vcol: "value"})
        valid = tmp["value"].notna()
        data = sorted(
            tmp[valid].to_dict("records") + [{"time": t} for t in tmp.loc[~valid, "time"]],
            key=lambda r: r["time"],
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

    def stats_legend(
        self,
        metrics: Dict[str, Any],
        position: str = "top-left",
    ) -> "Chart":
        """Overlay a performance-stats box on the chart.

        Parameters
        ----------
        metrics : dict
            Ordered dict of label → value pairs, e.g.::

                chart.stats_legend({
                    "Return": "+47.2%",
                    "CAGR": "8.4%",
                    "Sharpe": "1.42",
                    "Max DD": "-12.3%",
                })

        position : str
            Corner anchor: ``"top-left"`` (default), ``"top-right"``,
            ``"bottom-left"``, ``"bottom-right"``.
        """
        self._stats_legend = {"metrics": dict(metrics), "position": position}
        return self

    def threshold_control(
        self,
        df: pd.DataFrame,
        threshold: float = 0.0,
        min_val: float = -0.05,
        max_val: float = 0.05,
        step: float = 0.001,
        value_col: Optional[str] = None,
        series_index: int = 0,
        buy_color: Optional[str] = None,
        sell_color: Optional[str] = None,
    ) -> "Chart":
        """Embed an interactive threshold slider that rebuilds buy/sell markers in real time.

        Fully self-contained in the rendered HTML — no Python callbacks needed.
        Drag the slider → JS recomputes entry/exit markers on the live chart instantly.

        Usage::

            chart = (
                Chart(theme="dark", height=400)
                .candlestick(df)
                .threshold_control(pred_cum, threshold=0.0, min_val=-0.05, max_val=0.05, step=0.001)
            )

        Parameters
        ----------
        df : DataFrame
            Time-series of signal/predicted-return values.
        threshold : float
            Initial threshold (default 0.0).  Values >= threshold → long.
        min_val / max_val : float
            Slider range endpoints.
        step : float
            Slider granularity (e.g. 0.001 → 3 decimal places).
        series_index : int
            Index of the series to attach markers to (default 0 = first series).
        """
        df = self._prepare_time(df)
        vcol = self._find_value_col(df, value_col)
        data = (
            df[["time", vcol]]
            .rename(columns={vcol: "value"})
            .dropna(subset=["value"])
            .to_dict("records")
        )
        up_clr = buy_color or self._theme.get("candlestick", {}).get("upColor", "#26a69a")
        dn_clr = sell_color or self._theme.get("candlestick", {}).get("downColor", "#ef5350")
        decimals = max(0, -int(math.floor(math.log10(step)))) if step > 0 else 3
        self._threshold_config = {
            "data": data,
            "threshold": threshold,
            "min_val": min_val,
            "max_val": max_val,
            "step": step,
            "decimals": decimals,
            "series_index": series_index,
            "buy_color": up_clr,
            "sell_color": dn_clr,
        }
        return self

    def smoothing_control(
        self,
        raw_series: Optional[pd.Series] = None,
        series_index: int = -1,
        mode: str = "rolling",
        window_init: int = 20,
        window_min: int = 2,
        window_max: int = 252,
        window_step: int = 1,
        label: Optional[str] = None,
        color: Optional[str] = None,
        variants: Optional[dict] = None,
        variants_init: Optional[Any] = None,
    ) -> "Chart":
        """Add an interactive slider that updates a chart series with a smoothed line.

        Two modes:

        **Built-in (SMA / EMA)** — pass ``raw_series`` and ``mode``:
            Slider recomputes rolling mean or EMA in pure JS, no Python needed.

        **Custom smoother** — pass ``variants`` dict:
            Pre-compute your own smoothed series in Python (Kalman, HP filter,
            LOWESS, anything) as ``{param_value: pd.Series}``.
            The slider swaps between the pre-computed arrays in JS — no callbacks.

        Parameters
        ----------
        raw_series : pd.Series with DatetimeIndex — source for built-in SMA/EMA.
        series_index : Which series to update. ``-1`` = last series added.
        mode : ``"rolling"`` (SMA) or ``"ema"`` (exponential). Ignored if *variants* given.
        window_init : Starting window. Ignored if *variants* given.
        window_min, window_max, window_step : Slider range. Ignored if *variants* given.
        label : Slider label prefix.
        color : Accent colour for the slider thumb.
        variants : Dict ``{param_value: pd.Series}`` — pre-computed smoothed series.
            Keys are shown in the slider label; order is preserved.
        variants_init : Which key to start on (default: middle of the dict).
        """
        acc = color or "#a0c4ff"

        if variants is not None:
            # ── Custom smoother path — pre-computed arrays, JS just swaps ──
            keys = list(variants.keys())
            arrays = []
            for _s in variants.values():
                _s = _s.dropna()
                if hasattr(_s.index, "strftime"):
                    _times = _s.index.strftime("%Y-%m-%d").tolist()
                else:
                    _times = [str(t) for t in _s.index]
                arrays.append([{"time": t, "value": float(v)} for t, v in zip(_times, _s.values)])
            if variants_init is None:
                init_idx = len(keys) // 2
            elif variants_init in keys:
                init_idx = keys.index(variants_init)
            else:
                init_idx = 0
            lbl = label or "param"
            self._smoothing_configs.append({
                "mode":          "variants",
                "variants_data": arrays,
                "variants_keys": [str(k) for k in keys],
                "variants_init": init_idx,
                "series_index":  series_index,
                "label":         lbl,
                "color":         acc,
            })
            return self

        # ── Built-in SMA / EMA path ──────────────────────────────────────
        if raw_series is None:
            raise ValueError("smoothing_control: provide either raw_series or variants=")
        s = raw_series.dropna()
        if hasattr(s.index, "strftime"):
            times = s.index.strftime("%Y-%m-%d").tolist()
        else:
            times = [str(t) for t in s.index]
        raw_data = [{"time": t, "value": float(v)} for t, v in zip(times, s.values)]
        lbl = label or ("hl" if mode == "ema" else "win")
        self._smoothing_configs.append({
            "raw_data":     raw_data,
            "series_index": series_index,
            "mode":         mode,
            "window_init":  window_init,
            "window_min":   window_min,
            "window_max":   window_max,
            "window_step":  window_step,
            "label":        lbl,
            "color":        acc,
        })
        return self

    def background_image(
        self,
        url: str,
        blur: int = 0,
        tint: str = "rgba(6,6,20,0.40)",
        glass_blur: int = 16,
        glass_tint: str = "rgba(10,10,26,0.55)",
    ) -> "Chart":
        """Set a custom background image with a frosted-glass panel over it.

        The chart canvas becomes transparent; the image is rendered as the
        body background and the chart floats as a frosted-glass card on top.

        Parameters
        ----------
        url : str
            Image URL (``https://...``) or a base64 data URI
            (``data:image/jpeg;base64,...``).
        blur : int
            Blur applied directly to the background image layer (px). Default 0.
        tint : str
            Colour overlay between the image and the glass panel.
        glass_blur : int
            ``backdrop-filter: blur(Xpx)`` strength on the glass panel (default 16).
        glass_tint : str
            Semi-transparent background colour of the glass panel.
        """
        self._bg_image_config = {
            "url": url,
            "blur": blur,
            "tint": tint,
            "glass_blur": glass_blur,
            "glass_tint": glass_tint,
        }
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

        # If theme has CSS texture background or bg image, make canvas transparent
        if self._theme.get("background_css") or self._bg_image_config:
            layout["background"] = {"type": "solid", "color": "rgba(0,0,0,0)"}

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

        # ── Threshold slider components ───────────────────────────────────
        slider_html = ""
        slider_js = ""
        slider_extra_height = 0
        if self._threshold_config:
            tc = self._threshold_config
            dec = tc["decimals"]
            is_dark_bg = self._theme_name in ("dark", "midnight", "distfit")
            bar_bg = "rgba(0,0,0,0.55)" if is_dark_bg else "rgba(255,255,255,0.72)"
            lbl_c = "rgba(255,255,255,0.88)" if is_dark_bg else "rgba(0,0,0,0.78)"
            cnt_c = "rgba(255,255,255,0.48)" if is_dark_bg else "rgba(0,0,0,0.42)"
            thr0 = tc["threshold"]
            svar = f"s{tc['series_index']}"
            tdata_json = self._json(tc["data"])
            slider_extra_height = 36
            self._slider_extra_height = slider_extra_height
            slider_html = (
                f'<div id="th-bar" style="display:flex;align-items:center;justify-content:center;'
                f'gap:10px;background:transparent;padding:6px 16px;white-space:nowrap;'
                f'height:{slider_extra_height}px">'
                f'<span id="th-label" style="color:{lbl_c};'
                f"font:11px/1 'SF Mono','Consolas',monospace;min-width:72px\">"
                f'\u03b8\u00a0=\u00a0{thr0:.{dec}f}</span>'
                f'<input id="th-slider" type="range" '
                f'min="{tc["min_val"]}" max="{tc["max_val"]}" step="{tc["step"]}" '
                f'value="{thr0}" '
                f'style="width:220px;cursor:pointer;accent-color:{tc["buy_color"]}">'
                f'<span id="th-count" style="color:{cnt_c};font:11px/1 sans-serif;'
                f'min-width:60px;text-align:right"></span>'
                f'</div>'
            )
            # Parse shading color from buy_color hex
            bc = tc["buy_color"].lstrip("#")
            sr, sg, sb = int(bc[0:2], 16), int(bc[2:4], 16), int(bc[4:6], 16)
            shade_fill = f"rgba({sr},{sg},{sb},0.10)"
            shade_line = f"rgba({sr},{sg},{sb},0.25)"

            slider_js = "\n".join([
                "    // ── Threshold slider + position shading ──────────────────────",
                f"    const _td = {tdata_json};",
                f'    const _tBuy = "{tc["buy_color"]}";',
                f'    const _tSell = "{tc["sell_color"]}";',
                f"    const _tDec = {dec};",
                "",
                "    // Shading area series (position overlay)",
                "    const _shadeSeries = chart.addSeries(LightweightCharts.AreaSeries, {",
                '        priceScaleId: "_thShade",',
                "        lineWidth: 1,",
                f'        lineColor: "{shade_line}",',
                "        lineType: 2,",
                f'        topColor: "{shade_fill}",',
                '        bottomColor: "transparent",',
                "        crosshairMarkerVisible: false,",
                "        pointMarkersVisible: false,",
                "        lastValueVisible: false,",
                "        priceLineVisible: false,",
                "    });",
                '    chart.priceScale("_thShade").applyOptions({visible:false,scaleMargins:{top:0,bottom:0}});',
                "",
                "    function _buildShade(thr) {",
                "        const d = []; let on = false;",
                "        for (let i = 0; i < _td.length; i++) {",
                "            const above = _td[i].value >= thr;",
                "            if (above && !on) on = true;",
                "            else if (!above && on) on = false;",
                "            d.push({time: _td[i].time, value: on ? 1 : 0});",
                "        }",
                "        return d;",
                "    }",
                "",
                "    function _mkrs(thr) {",
                "        const m = []; let on = false;",
                "        for (let i = 0; i < _td.length; i++) {",
                "            const a = _td[i].value >= thr;",
                '            if (a && !on) { m.push({time:_td[i].time,position:"belowBar",shape:"arrowUp",color:_tBuy,text:""}); on=true; }',
                '            else if (!a && on) { m.push({time:_td[i].time,position:"aboveBar",shape:"arrowDown",color:_tSell,text:""}); on=false; }',
                "        }",
                "        return m;",
                "    }",
                "",
                f"    const _thP = LightweightCharts.createSeriesMarkers({svar}, _mkrs({thr0}));",
                f"    _shadeSeries.setData(_buildShade({thr0}));",
                "    (function() {",
                f"        const m0 = _mkrs({thr0});",
                '        document.getElementById("th-count").textContent = m0.filter(x=>x.shape==="arrowUp").length + " signals";',
                "    })();",
                '    document.getElementById("th-slider").addEventListener("input", function() {',
                "        const thr = parseFloat(this.value);",
                '        document.getElementById("th-label").textContent = "\\u03b8\\u00a0=\\u00a0" + thr.toFixed(_tDec);',
                "        const m = _mkrs(thr);",
                "        _thP.setMarkers(m);",
                "        _shadeSeries.setData(_buildShade(thr));",
                '        document.getElementById("th-count").textContent = m.filter(x=>x.shape==="arrowUp").length + " signals";',
                "    });",
            ])

        # ── Stats legend overlay ──────────────────────────────────────────
        stats_html = ""
        if self._stats_legend:
            sl = self._stats_legend
            pos = sl["position"]
            is_dark_bg = self._theme_name in ("dark", "midnight", "distfit")
            box_bg = "rgba(10,10,26,0.62)" if is_dark_bg else "rgba(255,255,255,0.68)"
            lbl_c  = "rgba(255,255,255,0.55)" if is_dark_bg else "rgba(0,0,0,0.45)"
            val_c  = "rgba(255,255,255,0.92)" if is_dark_bg else "rgba(0,0,0,0.88)"
            corner_css = {
                "top-left":     "top:8px;left:8px",
                "top-right":    "top:8px;right:8px",
                "bottom-left":  "bottom:8px;left:8px",
                "bottom-right": "bottom:8px;right:8px",
            }.get(pos, "top:8px;left:8px")
            rows = "".join(
                f'<tr>'
                f'<td style="color:{lbl_c};padding:1px 8px 1px 0;white-space:nowrap">'
                f'{html_module.escape(str(k))}</td>'
                f'<td style="color:{val_c};text-align:right;font-weight:600">'
                f'{html_module.escape(str(v))}</td>'
                f'</tr>'
                for k, v in sl["metrics"].items()
            )
            stats_html = (
                f'<div style="position:absolute;{corner_css};z-index:6;'
                f'background:{box_bg};'
                f'backdrop-filter:blur(18px) saturate(180%);-webkit-backdrop-filter:blur(18px) saturate(180%);'
                f'border:1px solid rgba(255,255,255,0.10);'
                f'border-radius:10px;padding:8px 12px;pointer-events:none;">' 
                f'<table style="border-collapse:collapse;'
                f'font:11px/1.6 \'SF Mono\',\'Consolas\',monospace">'
                f'{rows}</table></div>'
            )

        # ── Background image glass overlay ────────────────────────────────
        _bgi = self._bg_image_config
        if _bgi:
            # Put image on body so backdrop-filter on #glass can blur it
            _url_safe = _bgi["url"].replace('"', "%22")
            bg_css = f'background:url("{_url_safe}") center/cover no-repeat;'
            # Optional image-level blur via a transparent backdrop-filter helper div
            _blur_div = (
                f'<div style="position:absolute;inset:0;z-index:0;'
                f'backdrop-filter:blur({_bgi["blur"]}px);'
                f'-webkit-backdrop-filter:blur({_bgi["blur"]}px);"></div>'
                if _bgi["blur"] > 0 else ""
            )
            _glass_open = (
                f'{_blur_div}'
                f'<div id="bg-tint" style="position:absolute;inset:0;z-index:1;'
                f'background:{_bgi["tint"]}"></div>'
                f'<div id="glass" style="position:absolute;inset:0;z-index:2;'
                f'backdrop-filter:blur({_bgi["glass_blur"]}px) saturate(180%);'
                f'-webkit-backdrop-filter:blur({_bgi["glass_blur"]}px) saturate(180%);'
                f'background:{_bgi["glass_tint"]};border-radius:12px;overflow:hidden;">'
            )
            _glass_close = "</div>"
        else:
            _glass_open = ""
            _glass_close = ""

        # ── Smoothing control sliders ──────────────────────────────────────
        smoothing_html = ""
        smoothing_js   = ""
        smoothing_extra_height = 0
        is_dark_bg = self._theme_name in ("dark", "midnight", "distfit")
        _lbl_c = "rgba(255,255,255,0.88)" if is_dark_bg else "rgba(0,0,0,0.78)"
        for sc_idx, sc in enumerate(self._smoothing_configs):
            smoothing_extra_height += 36
            sid = f"sm-slider-{sc_idx}"
            lid = f"sm-label-{sc_idx}"
            n_series = len(self._series)
            target_idx = sc["series_index"] if sc["series_index"] >= 0 else n_series + sc["series_index"]
            svar_sm = f"s{target_idx}"
            mode = sc["mode"]
            lbl  = sc["label"]

            if mode == "variants":
                # ── variants: discrete switcher, JS swaps pre-computed arrays ──
                n_var    = len(sc["variants_keys"])
                init_idx = sc["variants_init"]
                init_key = sc["variants_keys"][init_idx]
                smoothing_html += (
                    f'<div style="display:flex;align-items:center;justify-content:center;'
                    f'gap:10px;background:transparent;padding:4px 16px;white-space:nowrap;height:36px">'
                    f'<span id="{lid}" style="color:{_lbl_c};'
                    f"font:11px/1 'SF Mono','Consolas',monospace;min-width:80px\">"
                    f'{lbl} {init_key}</span>'
                    f'<input id="{sid}" type="range" '
                    f'min="0" max="{n_var - 1}" step="1" '
                    f'value="{init_idx}" '
                    f'style="width:220px;cursor:pointer;accent-color:{sc["color"]}">'
                    f'</div>\n'
                )
                var_data_id = f"_smVarData{sc_idx}"
                var_keys_id = f"_smVarKeys{sc_idx}"
                smoothing_js += "\n".join([
                    f"    // \u2500\u2500 Smoothing slider #{sc_idx} (variants) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500",
                    f"    const {var_data_id} = {self._json(sc['variants_data'])};",
                    f"    const {var_keys_id} = {self._json(sc['variants_keys'])};",
                    f"    {svar_sm}.setData({var_data_id}[{init_idx}]);",
                    f"    document.getElementById('{sid}').addEventListener('input', function() {{",
                    f"        const idx = parseInt(this.value);",
                    f"        document.getElementById('{lid}').textContent = '{lbl} ' + {var_keys_id}[idx];",
                    f"        {svar_sm}.setData({var_data_id}[idx]);",
                    f"    }});",
                    "",
                ])

            else:
                # ── built-in SMA / EMA: compute fn runs in JS ──────────────────
                raw_id   = f"_smRaw{sc_idx}"
                raw_json = self._json(sc["raw_data"])
                smoothing_html += (
                    f'<div style="display:flex;align-items:center;justify-content:center;'
                    f'gap:10px;background:transparent;padding:4px 16px;white-space:nowrap;height:36px">'
                    f'<span id="{lid}" style="color:{_lbl_c};'
                    f"font:11px/1 'SF Mono','Consolas',monospace;min-width:80px\">"
                    f'{lbl} {sc["window_init"]}</span>'
                    f'<input id="{sid}" type="range" '
                    f'min="{sc["window_min"]}" max="{sc["window_max"]}" step="{sc["window_step"]}" '
                    f'value="{sc["window_init"]}" '
                    f'style="width:220px;cursor:pointer;accent-color:{sc["color"]}">'
                    f'</div>\n'
                )
                if mode == "ema":
                    compute_fn = (
                        "function _smCompute(rd, win) {\n"
                        "    const k = 2 / (win + 1); let ema = null; const out = [];\n"
                        "    for (const d of rd) {\n"
                        "        ema = ema === null ? d.value : d.value * k + ema * (1 - k);\n"
                        "        out.push({time: d.time, value: ema});\n"
                        "    }\n"
                        "    return out;\n"
                        "}"
                    )
                else:
                    compute_fn = (
                        "function _smCompute(rd, win) {\n"
                        "    const out = []; let sum = 0, buf = [];\n"
                        "    for (const d of rd) {\n"
                        "        buf.push(d.value); sum += d.value;\n"
                        "        if (buf.length > win) { sum -= buf.shift(); }\n"
                        "        out.push({time: d.time, value: sum / buf.length});\n"
                        "    }\n"
                        "    return out;\n"
                        "}"
                    )
                smoothing_js += "\n".join([
                    f"    // \u2500\u2500 Smoothing slider #{sc_idx} ({mode}) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500",
                    f"    const {raw_id} = {raw_json};",
                    f"    {compute_fn}",
                    f"    {svar_sm}.setData(_smCompute({raw_id}, {sc['window_init']}));",
                    f"    document.getElementById('{sid}').addEventListener('input', function() {{",
                    f"        const win = parseInt(this.value);",
                    f"        document.getElementById('{lid}').textContent = '{lbl} ' + win;",
                    f"        {svar_sm}.setData(_smCompute({raw_id}, win));",
                    f"    }});",
                    "",
                ])

        total_extra = slider_extra_height + smoothing_extra_height
        self._slider_extra_height = total_extra

        return f"""<!DOCTYPE html>
<html><head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<script>{lc_js}</script>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{{bg_css}overflow:hidden;position:relative;border-radius:12px;height:{self._height + 28 + total_extra}px}}
#fc{{width:{width_css};height:{self._height}px;position:relative;z-index:1}}
#err{{position:absolute;top:0;left:0;color:red;font-size:11px;z-index:9999;padding:4px;background:rgba(0,0,0,0.9);display:none}}
#signum-logo{{position:absolute;right:12px;bottom:4px;z-index:5;opacity:0.7;pointer-events:none;{_logo_invert}}}
</style>
</head><body>
{bg_svg}{_glass_open}<div id="fc"></div>
{stats_html}
<div id="err"></div>
{slider_html}
{smoothing_html}
{_glass_close}{'<img id="signum-logo" src="data:image/svg+xml;base64,' + _LOGO_B64 + '" width="30" height="30" alt="Signum">' if self._logo else ''}
<script>
try {{
    const chart = LightweightCharts.createChart(document.getElementById('fc'), {chart_opts});
    {series_js}
    chart.timeScale().fitContent();
    {slider_js}
    {smoothing_js}
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
        extra = getattr(self, "_slider_extra_height", 0)
        h = self._height + 30 + extra
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
