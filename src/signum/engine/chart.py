"""Signum Chart - Professional financial charting, inspired by Lightweight Charts.

Renders publication-quality charts from pandas DataFrames.
Works in Jupyter notebooks (inline), Dash apps, Streamlit, and standalone HTML.
"""

import json
import math
import re
import html as html_module
from functools import lru_cache
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd

from .themes import THEMES, resolve_theme
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


# ── Allocation palette ────────────────────────────────────────────────────────
# Muted, sophisticated tones for portfolio holdings.
_ALLOC_PALETTE = [
    "#6BA3D0",  # Soft steel blue
    "#82C785",  # Sage green
    "#F4A261",  # Warm apricot
    "#E76F51",  # Terracotta
    "#9B87C7",  # Soft lavender
    "#5BC0BE",  # Teal
    "#F18F9C",  # Dusty rose
    "#8AB17D",  # Olive green
]

# Past the eighth holding the palette has to produce colours this list does not
# contain.  They are picked, not computed from a formula — see ``_extend_palette``
# — and each is chosen to sit as far as possible from every colour already in use,
# in the same muted register as the eight above (both the lightness band and the
# chroma ceiling are derived from them, so the extras read as more of the same
# palette rather than as a second, louder one).
#
# The measuring stick throughout is OKLab ΔE, never hex equality.  Two hexes one
# bit apart are the same colour to a reader, so "no repeats" has to mean
# perceptually distinct or it means nothing: earlier attempts here produced
# #00ca1b vs #1bca00 (ΔE 0.6) and #189541 vs #189530 (ΔE 1.9) — each a different
# hex and an identical band on screen.  HLS is unusable for this, since 10° of its
# hue is invisible through the greens and obvious through the blues; hence the
# OKLCH conversions below.


# ── OKLCH ⇄ sRGB ──────────────────────────────────────────────────────────────
# Björn Ottosson's OKLab, plus the polar form.  Perceptually uniform, so equal
# steps in it are equal steps to the eye — the whole point of using it here.

def _hex_to_oklch(hex_color: str) -> tuple:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    r, g, b = (c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
               for c in (r, g, b))
    l = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3)
    m = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3)
    s = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3)
    lig = 0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s
    a_ = 1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s
    b_ = 0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s
    return lig, math.hypot(a_, b_), math.atan2(b_, a_) % (2 * math.pi)


def _oklch_to_rgb(lig: float, chroma: float, hue: float) -> tuple:
    a_, b_ = chroma * math.cos(hue), chroma * math.sin(hue)
    l = (lig + 0.3963377774 * a_ + 0.2158037573 * b_) ** 3
    m = (lig - 0.1055613458 * a_ - 0.0638541728 * b_) ** 3
    s = (lig - 0.0894841775 * a_ - 1.2914855480 * b_) ** 3
    r = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    b = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
    return tuple(1.055 * (c ** (1 / 2.4)) - 0.055 if c > 0.0031308
                 else 12.92 * c for c in (max(c, 0.0) for c in (r, g, b)))


def _oklch_to_hex(lig: float, chroma: float, hue: float) -> str:
    """OKLCH → hex, walking chroma down until the colour fits in sRGB.

    Clamping the channels instead would silently shift hue — the failure mode
    where two different requests clip to the same displayable colour.
    """
    lo, hi = 0.0, chroma
    for _ in range(18):
        if all(-1e-4 <= c <= 1 + 1e-4 for c in _oklch_to_rgb(lig, hi, hue)):
            break
        lo, hi = lo, (lo + hi) / 2
    rgb = _oklch_to_rgb(lig, hi, hue)
    return "#%02x%02x%02x" % tuple(round(min(1.0, max(0.0, c)) * 255) for c in rgb)


def _oklab_of(hex_color: str) -> tuple:
    lig, chroma, hue = _hex_to_oklch(hex_color)
    return lig, chroma * math.cos(hue), chroma * math.sin(hue)


def _candidate_pool(l_lo: float, l_hi: float, c_cap: float) -> List[tuple]:
    """Displayable colours on a coarse OKLCH lattice, as (hex, OKLab) pairs.

    Sampled in OKLCH so the lattice is perceptually even, then deduped by hex —
    high-lightness rows run out of gamut and collapse onto each other, and a
    duplicate in the pool would let the sampler "spend" a slot on a colour that is
    already taken.
    """
    seen, pool = set(), []
    steps = max(2, int(round((l_hi - l_lo) / 0.075)) + 1)
    for li in range(steps):
        lig = l_lo + (l_hi - l_lo) * li / (steps - 1)
        for deg in range(0, 360, 6):
            for frac in (1.0, 0.62, 0.34):
                hexc = _oklch_to_hex(lig, c_cap * frac, math.radians(deg))
                if hexc not in seen:
                    seen.add(hexc)
                    pool.append((hexc, _oklab_of(hexc)))
    return pool


@lru_cache(maxsize=32)
def _extend_palette(n: int, base: tuple) -> tuple:
    """Grow ``base`` to ``n`` colours by farthest-point sampling in OKLab.

    Each new colour is the displayable one whose *nearest already-used* colour is
    as far away as possible.  That maximises the minimum separation directly,
    which is the property actually wanted ("no two holdings look alike") —
    unlike stepping a formula, which can only hope for it and, on this palette's
    uneven hue gaps, misses: red and orange sit ~20° apart, so anything that
    subdivides per-slot gaps crams red's extras into that gap and reproduces the
    near-duplicates it was meant to remove.

    Sampling is also self-ordering: pick k+1 is far from every earlier pick, the
    immediately preceding one included, so consecutive bands separate too.
    """
    lch = [_hex_to_oklch(c) for c in base]
    mid = sum(l for l, _, _ in lch) / len(lch)
    # Chroma ceiling comes from the palette itself, so the extras sit in its
    # register — a fixed cap would make them louder than a muted base or duller
    # than a vivid one.
    c_cap = max(c for _, c, _ in lch)
    # Widen the lightness band only as far as the count actually forces.  Spread in
    # lightness is what buys separation once hue runs out, but it is also what makes
    # a palette look like it contains near-blacks and near-whites, so it is spent
    # reluctantly: at 25 holdings a ±0.09 band already separates as well as a wide
    # one (both bottom out at the base palette's own closest pair), and only past
    # ~35 does holding it tight start costing real distance.
    half = min(0.15, 0.05 + 0.0024 * (n - len(base)))
    pool = _candidate_pool(max(0.34, mid - half), min(0.88, mid + half), c_cap)
    chosen = list(base)
    chosen_lab = [_oklab_of(c) for c in base]
    taken = set(base)
    # Distance from each candidate to the nearest colour already chosen, updated
    # incrementally — recomputing it per pick would be O(n²·pool) for no gain.
    near = [min(math.dist(lab, c) for c in chosen_lab) for _, lab in pool]
    while len(chosen) < n:
        best = max(range(len(pool)), key=lambda i: near[i])
        hexc, lab = pool[best]
        if hexc in taken:                       # pool exhausted — nothing left to add
            break
        chosen.append(hexc)
        taken.add(hexc)
        near[best] = -1.0
        for i, (_, lab_i) in enumerate(pool):
            if near[i] > 0:
                near[i] = min(near[i], math.dist(lab_i, lab))
    return tuple(chosen)


def _alloc_colors(n: int, base: List[str]) -> List[str]:
    """Return ``n`` colours from ``base``, extended so none of them repeat.

    The first ``len(base)`` come through untouched — the validated order above is
    the common case and is not paraphrased.  Past that the palette is grown by
    farthest-point sampling (see ``_extend_palette``), so 25 or 50 holdings each
    get their own colour rather than a second copy of someone else's.

    "Distinct" here means *visibly* distinct, not merely a different hex — the
    floor is measured in OKLab ΔE, because two hexes one bit apart are the same
    colour to a reader.  Separation still narrows as N grows: sRGB does not hold
    50 colours as far apart as it holds 8, so past ~16 bands the honest fix is
    fewer of them — roll the tail into an "Other" row — not more colours.
    """
    if not base:
        return []
    if n <= len(base):
        return list(base[:n])
    out = list(_extend_palette(n, tuple(base)))
    while len(out) < n:          # only reachable if the lattice ran dry (~500 colours)
        out.append(out[len(out) % len(base)])
    return out


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

    # ── Execution utility ─────────────────────────────────────────────────

    @staticmethod
    def apply_execution(signal, returns, execution=1, open_returns=None, carry_in=True):
        """Apply an execution lag to a signal and return daily P&L.

        Lag = number of bars between signal and FILL (position entry).
        Signal fires at close[T].  Fill happens at close[T + execution].
        First return earned starts the bar AFTER the fill.

        Parameters
        ----------
        signal : pd.Series  Binary (0/1) or continuous signal.
        returns : pd.Series  Close-to-close returns aligned with signal.
        execution : int or ``"NO"``
            ``0``  — fill at close[T] (precise, same bar).  Earn cc_ret[T+1].
            ``1``  — fill at close[T+1] (MOC next bar, default).  Earn cc_ret[T+2].
            ``N``  — fill at close[T+N].  Earn cc_ret[T+N+1].
            ``"NO"`` — fill at open[T+1] (MOO).  Entry bar earns oc_ret;
                  subsequent holding bars earn cc_ret (close-to-close).
        open_returns : pd.Series, optional
            Open-to-close (intraday) returns: ``close / open - 1``.
            Required when ``execution="NO"`` (MOO).

        Returns
        -------
        pd.Series — daily strategy returns.
        """
        import warnings as _w
        _is_moo = False
        if isinstance(execution, str):
            _exec = execution.upper()
            if _exec in ("NO", "NM"):
                if _exec == "NM":
                    _w.warn(
                        "execution='NM' is deprecated; use execution='NO' (next open).",
                        UserWarning, stacklevel=2,
                    )
                if open_returns is None:
                    raise ValueError(
                        "execution='NO' (MOO) requires open_returns. "
                        "Pass open_returns=<open-to-close intraday returns (close/open-1)> or use execution=1 (MOC)."
                    )
                else:
                    cov = open_returns.replace(0, float("nan")).notna().mean()
                    if cov < 0.95:
                        _w.warn(
                            f"execution='NO': open_returns coverage is only {cov:.1%}; "
                            "missing bars earn 0.",
                            UserWarning, stacklevel=2,
                        )
                    ret, lag = open_returns, 1
                    _is_moo = True
            else:
                raise ValueError(
                    f"Unknown execution={execution!r}. "
                    "Use an integer (0=same_bar, 1=MOC, 2=lag2, N=lagN) or 'NO' (next open / MOO)."
                )
        elif isinstance(execution, int):
            lag, ret = execution + 1, returns
        else:
            raise ValueError(
                f"execution must be an integer or 'NO', got {execution!r}."
            )

        shifted = signal.shift(lag)
        if carry_in:
            # Assume position was already established before the window —
            # fill the first `lag` NaN bars with the first signal value.
            shifted = shifted.fillna(signal.iloc[0])
        else:
            shifted = shifted.fillna(0)

        if _is_moo:
            # MOO: entry bar earns open→close; holding bars earn close→close
            _prev = shifted.shift(1)
            _prev = _prev.fillna(shifted.iloc[0] if carry_in else 0)
            entry = (shifted > 0) & (_prev <= 0)
            blended = returns.copy()
            blended[entry] = open_returns[entry]
            return shifted * blended

        return shifted * ret

    # ── Init ──────────────────────────────────────────────────────────────

    def __init__(
        self,
        theme: str = "dark",
        width: Optional[int] = None,
        height: int = 400,
        watermark: Optional[str] = None,
        logo: bool = True,
        y_format: Optional[str] = None,
    ):
        self._theme_name = theme.lower()
        self._theme = resolve_theme(theme)
        self._width = width
        self._height = height
        self._watermark = watermark
        self._logo = logo
        self._y_format = y_format  # 'kmb' for K/M/B suffixes, 'percent' for %
        self._series: List[Dict[str, Any]] = []
        self._price_lines: List[Dict[str, Any]] = []
        self._markers: Dict[int, List[Dict[str, Any]]] = {}
        self._line_color_idx = 0
        self._threshold_config: Optional[Dict[str, Any]] = None
        self._smoothing_configs: List[Dict[str, Any]] = []
        self._stats_legend: Optional[Dict[str, Any]] = None
        self._bg_image_config: Optional[Dict] = None
        self._alloc_tooltip: Optional[Dict[str, Any]] = None
        self._seasonality_config: Optional[Dict[str, Any]] = None
        self._cone_config: Optional[Dict[str, Any]] = None

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
        line: bool = False,
        tag: bool = False,
        **options,
    ) -> "Chart":
        """Add line series. Automatically detects value/close column.

        line : bool, default False
            Show a dashed horizontal line that follows the crosshair at the
            series value on the hovered day (Dashboard only).
        tag : bool, default False
            Show that day's value as a label on the right axis (Dashboard only).
        """
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
        series_opts = {"priceLineVisible": False, "lastValueVisible": False, **options}
        series_opts["color"] = color or self._next_line_color()
        series_opts["lineWidth"] = width
        if name:
            series_opts["title"] = name
        self._series.append({"type": "LineSeries", "data": data, "options": series_opts})
        if line or tag:
            self._series[-1]["_value_line"] = {"line": line, "tag": tag}
        return self

    def area(
        self,
        df: pd.DataFrame,
        name: Optional[str] = None,
        color: Optional[str] = None,
        value_col: Optional[str] = None,
        line: bool = False,
        tag: bool = False,
        **options,
    ) -> "Chart":
        """Add area series (filled line chart).

        line : bool, default False
            Dashed horizontal line following the crosshair at the hovered
            day's value (Dashboard only).
        tag : bool, default False
            That day's value as a right-axis label (Dashboard only).
        """
        df = self._prepare_time(df)
        vcol = self._find_value_col(df, value_col)
        # LightweightCharts whitespace pattern: NaN rows become {time: ...} only.
        tmp = df[["time", vcol]].rename(columns={vcol: "value"})
        valid = tmp["value"].notna()
        data = sorted(
            tmp[valid].to_dict("records") + [{"time": t} for t in tmp.loc[~valid, "time"]],
            key=lambda r: r["time"],
        )
        series_opts = {
            **self._theme.get("area", {}),
            "priceLineVisible": False,
            "lastValueVisible": False,
            **options,
        }
        if color:
            series_opts["lineColor"] = color
            if color.startswith("#") and len(color) >= 7:
                r, g, b = (int(color[i:i+2], 16) for i in (1, 3, 5))
                series_opts["topColor"] = f"rgba({r},{g},{b},0.4)"
                series_opts["bottomColor"] = f"rgba({r},{g},{b},0.04)"
        if name:
            series_opts["title"] = name
        self._series.append({"type": "AreaSeries", "data": data, "options": series_opts})
        if line or tag:
            self._series[-1]["_value_line"] = {"line": line, "tag": tag}
        return self

    def histogram(
        self,
        df: pd.DataFrame,
        name: Optional[str] = None,
        value_col: Optional[str] = None,
        color: Optional[str] = None,
        line: bool = False,
        tag: bool = False,
        **options,
    ) -> "Chart":
        """Add histogram series.

        line : bool, default False
            Dashed horizontal line following the crosshair at the hovered
            day's value (Dashboard only).
        tag : bool, default False
            That day's value as a right-axis label (Dashboard only).
        """
        df = self._prepare_time(df)
        vcol = self._find_value_col(df, value_col)
        # LightweightCharts whitespace pattern: NaN rows become {time: ...} only.
        tmp = df[["time", vcol]].rename(columns={vcol: "value"})
        valid = tmp["value"].notna()
        data = sorted(
            tmp[valid].to_dict("records") + [{"time": t} for t in tmp.loc[~valid, "time"]],
            key=lambda r: r["time"],
        )
        series_opts = {
            **self._theme.get("histogram", {}),
            "priceLineVisible": False,
            "lastValueVisible": False,
            **options,
        }
        if color:
            series_opts["color"] = color
        if name:
            series_opts["title"] = name
        self._series.append({"type": "HistogramSeries", "data": data, "options": series_opts})
        if line or tag:
            self._series[-1]["_value_line"] = {"line": line, "tag": tag}
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
        line: bool = False,
        tag: bool = False,
        **options,
    ) -> "Chart":
        """Add baseline series (green above / red below a base value).

        line : bool, default False
            Dashed horizontal line following the crosshair at the hovered
            day's value (Dashboard only).
        tag : bool, default False
            That day's value as a right-axis label (Dashboard only).
        """
        df = self._prepare_time(df)
        vcol = self._find_value_col(df, value_col)
        # LightweightCharts whitespace pattern: NaN rows become {time: ...} only.
        tmp = df[["time", vcol]].rename(columns={vcol: "value"})
        valid = tmp["value"].notna()
        data = sorted(
            tmp[valid].to_dict("records") + [{"time": t} for t in tmp.loc[~valid, "time"]],
            key=lambda r: r["time"],
        )
        series_opts = {
            "baseValue": {"type": "price", "price": base_value},
            **self._theme.get("baseline", {}),
            "priceLineVisible": False,
            "lastValueVisible": False,
            **options,
        }
        self._series.append({"type": "BaselineSeries", "data": data, "options": series_opts})
        if line or tag:
            self._series[-1]["_value_line"] = {"line": line, "tag": tag}
        return self

    def bar(self, df: pd.DataFrame, **options) -> "Chart":
        """Add OHLC bar series (vertical bars with ticks for open/close).
        
        Traditional bar chart representation of OHLC data, where each bar
        shows the high-low range with left tick for open and right tick for close.
        
        DataFrame needs time + open/high/low/close columns.
        """
        df = self._prepare_time(df)
        o, h, l, c = (self._col_ci(df, x) for x in ("open", "high", "low", "close"))
        data = (
            df[["time", o, h, l, c]]
            .rename(columns={o: "open", h: "high", l: "low", c: "close"})
            .dropna(subset=["open", "high", "low", "close"])
            .to_dict("records")
        )
        series_opts = {**self._theme.get("candlestick", {}), **options}
        self._series.append({"type": "BarSeries", "data": data, "options": series_opts})
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

    def events(
        self,
        events,
        color: str = "#9aa0b4",
        opacity: float = 0.07,
        label_color: Optional[str] = None,
        series_index: int = 0,
        label_pos: str = "top",
    ) -> "Chart":
        """Highlight named historical periods with a faint full-height band + a small label.

        label_pos : "top" (default) or "bottom" — where the label sits in the chart.

        events : list of dicts, each ``{"start": date, "end": date (optional), "label": str}``.
                 If ``end`` is omitted the band is a thin marker at ``start``.

        Usage::

            chart.line(px).events([
                {"start": "2008-09-01", "end": "2009-03-09", "label": "GFC"},
                {"start": "2020-02-19", "end": "2020-04-01", "label": "COVID"},
            ])
        """
        hexc = color.lstrip("#")
        r, g, b = int(hexc[0:2], 16), int(hexc[2:4], 16), int(hexc[4:6], 16)
        fill = f"rgba({r},{g},{b},{opacity})"
        lbl_clr = label_color or f"rgba({r},{g},{b},{min(1.0, opacity*7):.2f})"
        for ev in events:
            s_ts = pd.Timestamp(ev["start"]); e_ts = pd.Timestamp(ev.get("end", ev["start"]))
            start, end = s_ts.strftime("%Y-%m-%d"), e_ts.strftime("%Y-%m-%d")
            pre = (s_ts - pd.Timedelta(days=2)).strftime("%Y-%m-%d")
            post = (e_ts + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
            label = ev.get("label", "")
            self._series.append({
                "type": "AreaSeries",
                # 0-anchors before/after pin the hidden scale to span 0..1 so the band fills the FULL height
                "data": [{"time": pre, "value": 0}, {"time": start, "value": 1},
                         {"time": end, "value": 1}, {"time": post, "value": 0}],
                "options": {
                    "priceScaleId": "_events", "lineColor": "rgba(0,0,0,0)",
                    "topColor": fill, "bottomColor": fill, "lineWidth": 0,
                    "lastValueVisible": False, "priceLineVisible": False,
                    "crosshairMarkerVisible": False, "pointMarkersVisible": False,
                },
                "price_scale": {"id": "_events", "scaleMargins": {"top": 0, "bottom": 0}},
            })
            band_idx = len(self._series) - 1     # the band series itself carries the label (top/bottom of chart)
            if label:
                # "top": below value=1 so the text sits *inside* the top edge (not clipped).
                # "bottom": above value=0 so it sits just over the x-axis.
                if label_pos == "bottom":
                    mk = {"time": pre, "position": "aboveBar"}
                else:
                    mk = {"time": start, "position": "belowBar"}
                mk.update({"shape": "circle", "color": lbl_clr, "text": label})
                self._markers.setdefault(band_idx, []).append(mk)
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

    def hline(
        self,
        y: float,
        label: str = "",
        color: Optional[str] = None,
        width: int = 1,
        style: int = 2,
        series_index: int = 0,
    ) -> "Chart":
        """Add a horizontal reference line at y-value.
        
        Simplified alias for price_line with more intuitive naming for non-price charts.
        
        Parameters
        ----------
        y : float
            Y-axis value where the line should be drawn.
        label : str, optional
            Text label for the line.
        color : str, optional
            Line color (hex or rgba). Defaults to theme line color.
        width : int, default 1
            Line width in pixels.
        style : int, default 2
            Line style: 0=solid, 1=dotted, 2=dashed, 3=large dashed, 4=sparse dotted.
        series_index : int, default 0
            Index of the series to attach the line to.
        
        Returns
        -------
        Chart
            Self for method chaining.
        """
        return self.price_line(
            price=y,
            title=label,
            color=color,
            line_width=width,
            line_style=style,
            series_index=series_index,
        )

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
                "pin": (0, 1),
            },
        })
        return self

    def allocation(
        self,
        df: pd.DataFrame,
        allocation_cols: Optional[List[str]] = None,
        colors: Optional[List[str]] = None,
        color_map: Optional[Dict[str, str]] = None,
        opacity: float = 0.6,
        style: str = "area",
        tooltip: bool = True,
        stacked: bool = True,
        hide_zero: bool = True,
        mode: str = "weights",
        value: Optional[Any] = None,
        base_value: float = 100.0,
        weights_pct: Optional[bool] = None,
        **options,
    ) -> "Chart":
        """Add multiple allocation series for portfolio visualization.
        
        Shows allocation percentages across multiple assets. In stacked mode,
        long positions stack upward from 0 and short positions stack downward
        from 0, creating a visual "pie" showing total exposure at each level.
        Lines are subtle to emphasize area changes over boundaries.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with time column and one column per asset allocation.
        allocation_cols : list of str, optional
            Column names containing allocation percentages (0-100 or 0-1).
            If None, uses all numeric columns except 'time'.
        colors : list of str, optional
            Base colours for the holdings, defaulting to the eight muted tones in
            ``_ALLOC_PALETTE``. Whatever the base list, it is *extended* rather
            than cycled past its length: further colours are picked to sit as far
            as possible from every colour already in use, in the same lightness
            and chroma register as the base. So 25 or 50 holdings each get their
            own colour — no two holdings on a chart ever share one.
        color_map : dict of str -> str, optional
            Pin specific holdings to specific colours, e.g. ``{"AAPL": "#3987e5"}``.
            Names not listed fall back to the palette. Use this when holdings rotate
            in and out and you want a name to keep its colour across charts —
            otherwise a colour tracks a holding's *position*, so a dropped name
            repaints everything below it.
        opacity : float, default 0.6
            Fill/bar opacity (0-1).
        style : str, default "area"
            Visualization style: "area" (stacked filled), "histogram" (bars), or "line".
        tooltip : bool, default True
            Show floating info panel on hover with all allocations at that date.
        stacked : bool, default True
            Stack areas cumulatively (longs up from 0, shorts down from 0).
            Set False for overlapping areas with separate scales.
        mode : str, default "weights"
            "weights" — bands are the weights themselves, so the stack sums to the target each day
            (100%, ±100% long/short, ±300% leveraged); the envelope is flat over time.
            "value" — bands are weightᵢ × portfolio NAV(t), with NAV rebased to `base_value` on day 1,
            so the stacked envelope IS the equity curve and each band enlarges as the book compounds.
            The hover tooltip still reports each holding as a % of that day's portfolio.
        value : pd.Series or pd.DataFrame, optional
            Required for mode="value": the portfolio NAV/equity over time (Series indexed by date, or a
            DataFrame with a time column + a value column). Aligned to `df` by date and rebased to `base_value`.
        base_value : float, default 100.0
            Starting NAV for value mode (day-1 total of the stacked area).
        weights_pct : bool, optional
            Force-interpret the weights as percent (True) or fraction (False). Auto-detected when None
            (percent if any |weight| > 1.5).
        **options
            Additional LightweightCharts series options.
        
        Returns
        -------
        Chart
            Self for method chaining.
        
        Example
        -------
        >>> # Stacked areas (default) - longs up from 0, shorts down from 0
        >>> chart = Chart().allocation(df, allocation_cols=['AAPL', 'GOOGL', 'MSFT'])
        >>> 
        >>> # Overlapping (non-stacked)
        >>> chart = Chart().allocation(df, stacked=False)
        """
        df = self._prepare_time(df)
        
        # Auto-detect allocation columns if not specified
        if allocation_cols is None:
            allocation_cols = [
                col for col in df.columns 
                if col != "time" and pd.api.types.is_numeric_dtype(df[col])
            ]
        
        if not allocation_cols:
            raise ValueError("No allocation columns found. Specify allocation_cols explicitly.")

        mode = (mode or "weights").lower()
        if mode not in ("weights", "value"):
            raise ValueError("mode must be 'weights' (sum-to-target each day) or 'value' (compounds with NAV)")

        # Are the weights in percent (0-100) or fraction (0-1)? Governs value-mode scaling + tooltip %.
        _wmax = float(np.nanmax(np.abs(df[allocation_cols].to_numpy(dtype=float)))) if len(df) else 0.0
        _is_pct = (_wmax > 1.5) if weights_pct is None else bool(weights_pct)

        # Value mode: band height = weightᵢ × portfolio NAV(t), NAV rebased to base_value on day 1 — so the
        # stacked envelope IS the equity curve and bands enlarge as the book compounds. Tooltip stays % of book.
        nav_arr = None
        if mode == "value":
            if value is None:
                raise ValueError("mode='value' requires `value` (a portfolio NAV/equity series over time)")
            if isinstance(value, pd.DataFrame):
                _tc = "time" if "time" in value.columns else value.columns[0]
                _vc = [c for c in value.columns if c != _tc][-1]
                _nav = pd.Series(pd.to_numeric(value[_vc], errors="coerce").values,
                                 index=pd.to_datetime(value[_tc]))
            else:
                _nav = pd.Series(pd.to_numeric(value, errors="coerce").values,
                                 index=pd.to_datetime(getattr(value, "index", value)))
            _nav = _nav.sort_index()
            _nav = _nav / _nav.dropna().iloc[0] * float(base_value)   # rebase day-1 to base_value
            nav_arr = _nav.reindex(pd.to_datetime(df["time"])).ffill().bfill().to_numpy()

        # Colours: the palette above, extended (never repeated) to however many
        # holdings there are, then overridden per name by color_map.
        base_palette = _ALLOC_PALETTE if colors is None else list(colors)
        colors = _alloc_colors(len(allocation_cols), base_palette)
        if color_map:
            colors = [color_map.get(col, colors[i]) for i, col in enumerate(allocation_cols)]

        # Determine series type
        if style.lower() in ("area", "filled"):
            series_type = "AreaSeries"
        elif style.lower() in ("histogram", "bar", "bars"):
            series_type = "HistogramSeries"
        else:
            series_type = "LineSeries"
        
        # Per-asset magnitudes to plot: raw weights, or value-scaled bands (weightᵢ × NAV) in value mode.
        if mode == "value":
            _frac = 100.0 if _is_pct else 1.0
            plot_df = df[["time"]].copy()
            for col in allocation_cols:
                plot_df[col] = (pd.to_numeric(df[col], errors="coerce").to_numpy() / _frac) * nav_arr
        else:
            plot_df = df

        # Compute cumulative sums for stacking
        if stacked and series_type == "AreaSeries":
            # For long/short portfolios: stack longs upward from 0, shorts downward from 0
            cumulative_df = plot_df[["time"]].copy()

            for i in plot_df.index:
                long_sum = 0.0
                short_sum = 0.0

                for col in allocation_cols:
                    val = plot_df.loc[i, col]
                    if pd.isna(val):
                        val = 0.0

                    if val > 0:
                        long_sum += val
                        cumulative_df.loc[i, col] = long_sum
                    elif val < 0:
                        short_sum += val
                        cumulative_df.loc[i, col] = short_sum
                    else:
                        # Zero-weight name: carry the running long total (a zero-thickness band riding the
                        # current top) instead of dropping the line to 0. Keeps every cumulative line
                        # monotonically nested through linear interpolation, so a name dropping out / loading
                        # in at a rotation no longer crosses its neighbours and leaves a black V-notch gap.
                        cumulative_df.loc[i, col] = long_sum
        else:
            cumulative_df = plot_df.copy()
        
        # Add a series for each allocation (reverse order for proper stacking visual)
        for idx, col in enumerate(reversed(allocation_cols)):
            rev_idx = len(allocation_cols) - 1 - idx
            
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame")
            
            tmp = cumulative_df[["time", col]].rename(columns={col: "value"})
            valid = tmp["value"].notna()
            data = sorted(
                tmp[valid].to_dict("records") + [{"time": t} for t in tmp.loc[~valid, "time"]],
                key=lambda r: r["time"],
            )
            
            # Parse color and add opacity.  Non-hex colours (a CSS name or rgba()
            # via color_map) can't be decomposed, so they pass through as-is —
            # r/g/b still need values for the stacked branch below.
            base_color = colors[rev_idx]
            r = g = b = None
            if base_color.startswith('#') and len(base_color.lstrip("#")) == 6:
                hex_c = base_color.lstrip("#")
                r, g, b = int(hex_c[0:2], 16), int(hex_c[2:4], 16), int(hex_c[4:6], 16)
                color_rgba = f"rgba({r},{g},{b},{opacity})"
                line_rgba = f"rgba({r},{g},{b},{min(opacity * 1.5, 1)})"
            else:
                color_rgba = base_color
                line_rgba = base_color

            # Build series options based on type
            if series_type == "AreaSeries":
                # For stacked mode: use BaselineSeries to ensure shorts fill toward 0, not chart bottom
                if stacked:
                    # Fully opaque fills so adjacent bands don't bleed through each other,
                    # especially during sign-flip transitions where a series' line wedges
                    # through 0 and crosses over neighboring bands.
                    soft_line = f"rgba({r},{g},{b},0.85)" if r is not None else base_color
                    solid_fill = f"rgba({r},{g},{b},1.0)" if r is not None else base_color
                    
                    series_opts = {
                        "baseValue": {"type": "price", "price": 0},
                        "topLineColor": soft_line,
                        "topFillColor1": solid_fill,
                        "topFillColor2": solid_fill,
                        "bottomLineColor": soft_line,
                        "bottomFillColor1": solid_fill,
                        "bottomFillColor2": solid_fill,
                        "lineWidth": 1.5,
                        "lastValueVisible": False,
                        "priceLineVisible": False,
                        "crosshairMarkerVisible": True,
                        "crosshairMarkerRadius": 2,
                        "crosshairMarkerBorderWidth": 0,
                        "crosshairMarkerBorderColor": base_color,
                        "crosshairMarkerBackgroundColor": base_color,
                        "priceScaleId": "right",
                        **options
                    }
                    series_type_override = "BaselineSeries"
                else:
                    series_opts = {
                        "lineColor": line_rgba,
                        "topColor": color_rgba,
                        "bottomColor": f"rgba({r},{g},{b},0.05)" if r is not None else "transparent",
                        "lineWidth": 2,
                        "title": col,
                        "priceScaleId": f"_alloc_{rev_idx}",
                        **options
                    }
                    series_type_override = "AreaSeries"
                # Only use separate scale if not stacking
                price_scale = None if stacked else {
                    "id": f"_alloc_{rev_idx}",
                    "scaleMargins": {"top": 0, "bottom": 0},
                }
            elif series_type == "HistogramSeries":
                series_opts = {
                    "color": color_rgba,
                    "title": col,
                    "priceScaleId": "right" if stacked else f"_alloc_{rev_idx}",
                    **options
                }
                series_type_override = "HistogramSeries"
                price_scale = None if stacked else {
                    "id": f"_alloc_{rev_idx}",
                    "scaleMargins": {"top": 0, "bottom": 0},
                }
            else:  # LineSeries
                series_opts = {
                    "color": line_rgba,
                    "lineWidth": 2,
                    "title": col,
                    **options
                }
                series_type_override = "LineSeries"
                price_scale = None
            
            # Use the potentially overridden series type (e.g., BaselineSeries for stacked areas)
            series_dict = {"type": series_type_override, "data": data, "options": series_opts}
            if price_scale:
                series_dict["price_scale"] = price_scale
            self._series.append(series_dict)
        
        # Store allocation tooltip metadata (use original non-cumulative data)
        if tooltip:
            # Build allocation data lookup using the same date string format as series data
            # After _prepare_time, dates are in "YYYY-MM-DD" string format
            alloc_data = {}
            _pmul = 100.0 if (mode == "value" and not _is_pct) else 1.0  # value-mode fractions → report as %
            for _, row in df.iterrows():
                date_key = str(row["time"])  # Already a string from _prepare_time
                alloc_data[date_key] = {col: float(row[col]) * _pmul if pd.notna(row[col]) else 0.0
                                        for col in allocation_cols}
            
            self._alloc_tooltip = {
                "assets": allocation_cols,
                "colors": list(colors),
                "data": alloc_data,
                "hide_zero": hide_zero,
            }
        
        # Configure right price scale with percentage formatting
        if stacked and series_type == "AreaSeries":
            self._series.append({
                "type": "__price_scale_config__",
                "scale_id": "right",
                "options": {
                    "mode": 0,
                    "visible": True,
                    "borderVisible": False,
                    "scaleMargins": {"top": 0.05, "bottom": 0.05},
                },
                "formatter": "value" if mode == "value" else "percent"
            })
        
        return self

    def seasonality(
        self,
        df: pd.DataFrame,
        period: Any = "Y",
        value_col: Optional[str] = None,
        mode: str = "percent",
        average: bool = True,
        brush: bool = True,
        seasons: Optional[List[str]] = None,
        colors: Optional[List[str]] = None,
        highlight_last: bool = True,
        average_color: Optional[str] = None,
    ) -> "Chart":
        """Overlay every season of a series on one shared in-season axis.

        Each season (calendar year by default) becomes its own line, all of them
        drawn against the same Jan-Dec axis, so a shape that repeats every year
        shows up as lines that stack on top of each other.

        Usage::

            Chart(theme="dark", height=460).seasonality(spx["close"]).show()

        Parameters
        ----------
        df : DataFrame or Series
            Time series to split. Needs a time column / DatetimeIndex.
        period : str or int, default ``"Y"``
            Season length. ``"Y"`` aligns by calendar date (the axis reads
            Jan...Dec, the way a seasonality chart should). ``"Q"``, ``"M"``,
            ``"W"`` and an integer bar count align by *position within the
            season* instead, and the axis is relabelled ``D1, D2, ...`` - use
            these when the cycle you are after has no calendar name.
        mode : ``"percent"`` or ``"value"``, default ``"percent"``
            ``"percent"`` rebases every season to its own first value = 0%, so
            seasons sitting at different price levels stay comparable. Pass
            ``"value"`` for a series that crosses zero (cumulative returns,
            spreads) - rebasing those is meaningless. Switchable in the chart.
        average : bool, default True
            Draw the mean across the *selected* seasons. It is recomputed live
            as the brush moves, which is what makes the brush worth having.
        brush : bool, default True
            Two-handle range control above the chart, selecting which seasons
            are drawn and which feed the average. Drag it to compare one span
            of seasons against another - a regime change shows up as the
            average line moving while you slide the window. Past twelve
            seasons it opens on the most recent twelve; the rest are one drag
            away rather than an unreadable thicket on first paint.
        seasons : list of str, optional
            Restrict to these season labels (``"2019"``, ``"2019Q1"``, ...).
        highlight_last : bool, default True
            Draw the most recent (usually incomplete) season thicker.

        Notes
        -----
        Feb 29 is dropped under ``period="Y"``: it has no counterpart in the
        other seasons, so there is nothing to align it against.
        """
        if mode not in ("percent", "value"):
            raise ValueError("mode must be 'percent' or 'value'")

        if isinstance(df, pd.Series):
            df = df.to_frame(df.name or "value")
        d = df.copy()
        tcol = self._detect_time_col(d)
        times = pd.to_datetime(d.index if tcol == "__index__" else d[tcol])
        vcol = self._find_value_col(d, value_col)
        work = pd.DataFrame({
            "t": pd.DatetimeIndex(times),
            "v": pd.to_numeric(d[vcol], errors="coerce").values,
        }).dropna(subset=["t"]).sort_values("t").reset_index(drop=True)
        if work.empty:
            raise ValueError("seasonality(): no dated rows to plot")

        # Split into seasons, and place every row on the shared axis.  Calendar
        # mode keeps the real month/day so the axis reads Jan-Dec; the ordinal
        # modes have no calendar meaning, so they map position-in-season onto
        # consecutive days from the same base and relabel the axis.
        base = pd.Timestamp("2001-01-01")
        unit = "D"
        if isinstance(period, (int, np.integer)) and not isinstance(period, bool):
            n = int(period)
            if n < 2:
                raise ValueError("seasonality(): integer period must be >= 2 bars")
            ordinal, unit = True, "Bar "
            idx = np.arange(len(work))
            work["key"] = idx // n
            pos = idx % n
            labels = {k: g["t"].iloc[0].strftime("%Y-%m-%d")
                      for k, g in work.groupby("key", sort=True)}
        else:
            p = str(period).upper()[:1]
            if p in ("Y", "A"):
                ordinal = False
                work = work[~((work["t"].dt.month == 2) & (work["t"].dt.day == 29))]
                work = work.reset_index(drop=True)
                work["key"] = work["t"].dt.year
                pos = None
                labels = {k: str(k) for k in work["key"].unique()}
            elif p == "Q":
                ordinal = True
                per = work["t"].dt.to_period("Q")
                work["key"] = per.astype(str)
                pos = (work["t"] - per.dt.start_time).dt.days.values
                labels = {k: k for k in work["key"].unique()}
            elif p == "M":
                ordinal = True
                work["key"] = work["t"].dt.strftime("%Y-%m")
                pos = (work["t"].dt.day - 1).values
                labels = {k: k for k in work["key"].unique()}
            elif p == "W":
                ordinal = True
                per = work["t"].dt.to_period("W")
                iso = per.dt.start_time.dt.isocalendar()
                work["key"] = (iso["year"].astype(str) + "-W"
                               + iso["week"].astype(int).map("{:02d}".format)).values
                pos = work["t"].dt.dayofweek.values
                labels = {k: k for k in work["key"].unique()}
            else:
                raise ValueError(
                    "seasonality(): period must be 'Y', 'Q', 'M', 'W' or an int bar count"
                )

        if ordinal:
            work["x"] = (base + pd.to_timedelta(pos, unit="D")).strftime("%Y-%m-%d")
        else:
            work["x"] = "2001-" + work["t"].dt.strftime("%m-%d")
        work = work.drop_duplicates(subset=["key", "x"], keep="last")

        order = work.groupby("key")["t"].min().sort_values().index.tolist()
        keys = [k for k in order if str(labels[k]) in set(seasons)] if seasons else order
        if not keys:
            raise ValueError("seasonality(): no seasons left after filtering")

        grid = sorted(work.loc[work["key"].isin(keys), "x"].unique())
        gpos = {x: i for i, x in enumerate(grid)}

        pct: List[List[Optional[float]]] = []
        val: List[List[Optional[float]]] = []
        for k in keys:
            g = work[work["key"] == k]
            row_v: List[Optional[float]] = [None] * len(grid)
            row_p: List[Optional[float]] = [None] * len(grid)
            anchor_s = g.loc[g["v"].notna(), "v"]
            anchor = float(anchor_s.iloc[0]) if len(anchor_s) else float("nan")
            # A season anchored at zero cannot be rebased - leave it out of
            # percent mode rather than divide by it.
            rebasable = anchor == anchor and anchor != 0
            for x, v in zip(g["x"], g["v"]):
                if pd.isna(v):
                    continue
                i = gpos[x]
                row_v[i] = float(v)
                if rebasable:
                    row_p[i] = (float(v) / anchor - 1.0) * 100.0
            val.append(row_v)
            pct.append(row_p)

        season_labels = [str(labels[k]) for k in keys]
        n_seasons = len(keys)
        pal = _alloc_colors(n_seasons, list(colors) if colors else _ALLOC_PALETTE)

        # The right-axis badges are off: eleven "year value" tags stacked down
        # the price scale hide the lines they label, and a colour key does the
        # same job without touching the plot.
        src = pct if mode == "percent" else val
        base_index = len(self._series)
        for i, lab in enumerate(season_labels):
            data = [{"time": grid[j], "value": v}
                    for j, v in enumerate(src[i]) if v is not None]
            self._series.append({
                "type": "LineSeries",
                "data": data,
                "options": {
                    "color": pal[i],
                    "lineWidth": 2 if (highlight_last and i == n_seasons - 1) else 1,
                    "title": lab,
                    "priceLineVisible": False,
                    "lastValueVisible": False,
                },
            })

        # First/last grid index each season covers.  The mean carries a season's
        # last value forward inside that span: a calendar day is a weekday in
        # some years and a weekend in others, so without it the set of seasons
        # contributing flips from day to day and the average saws up and down.
        spans = []
        for i in range(n_seasons):
            got = [j for j, v in enumerate(val[i]) if v is not None]
            spans.append([got[0], got[-1]] if got else [-1, -1])

        avg_index = None
        if average:
            avg_data = []
            last: List[Optional[float]] = [None] * n_seasons
            for j in range(len(grid)):
                total, count = 0.0, 0
                for i in range(n_seasons):
                    if src[i][j] is not None:
                        last[i] = src[i][j]
                    if last[i] is not None and spans[i][0] <= j <= spans[i][1]:
                        total += last[i]
                        count += 1
                if count:
                    avg_data.append({"time": grid[j], "value": total / count})
            avg_index = len(self._series)
            self._series.append({
                "type": "LineSeries",
                "data": avg_data,
                "options": {
                    "color": average_color or self._theme.get("line", {}).get("color", "#2962FF"),
                    "lineWidth": 3,
                    "title": "Avg",
                    "priceLineVisible": False,
                    "lastValueVisible": False,
                },
            })

        self._seasonality_config = {
            "grid": grid,
            "labels": season_labels,
            "colors": pal,
            "spans": spans,
            "pct": pct,
            "val": val,
            "mode": mode,
            "average_on": bool(average),
            "ordinal": bool(ordinal),
            "unit": unit,
            # Past a dozen seasons the overlay stops being readable, so the
            # brush opens on the most recent ones.  The rest are one drag away.
            "sel": [max(0, n_seasons - 12), n_seasons - 1],
            "base_index": base_index,
            "avg_index": avg_index,
            "brush": bool(brush) and n_seasons > 1,
        }
        return self

    def forecast(
        self,
        pred_df: pd.DataFrame,
        close: Optional[pd.Series] = None,
        threshold: Optional[float] = None,
        pred_cols: Optional[List[str]] = None,
        path: str = "latest",
        path_color: Optional[str] = None,
        shade: bool = True,
        series_index: int = 0,
        buy_color: Optional[str] = None,
        sell_color: Optional[str] = None,
    ) -> "Chart":
        """Overlay model forecast paths and entry/exit signals on the chart.

        Designed for TKAN / NN model output where each row is a multi-step
        price prediction from that date's anchor close.

        Parameters
        ----------
        pred_df : DataFrame
            Index = trade dates.  Columns = prediction steps, named ``d1``, ``d2``,
            … ``dN`` (auto-detected) or specified via *pred_cols*.
            Values are **price ratios** relative to the anchor close (e.g. 1.02 = +2%).
            If *close* is ``None``, values are treated as absolute prices instead.
        close : Series, optional
            Anchor close prices indexed by trade date (same index as pred_df).
            Pass the raw price series so ratios can be converted to price levels.
            If omitted, pred_df is assumed to already contain absolute prices.
        threshold : float, optional
            Entry signal threshold applied to ``max(d1…dN)`` per row.
            Rows where the max predicted ratio meets or exceeds *threshold* trigger
            an entry signal (arrowUp / shading).  E.g. ``threshold=1.015`` → +1.5%.
        pred_cols : list[str], optional
            Explicit column names to use as prediction steps.
            Defaults to all columns matching ``d<integer>`` sorted numerically.
        path : str
            ``"latest"`` (default) — draw the most recent date's forecast as a line
            extending past the last available price bar.
            ``"none"`` — skip the forecast path line.
        path_color : str, optional
            CSS colour for the forecast path line.  Defaults to the theme's up colour.
        shade : bool
            If ``True`` (default), shade chart background during active-signal periods.
        series_index : int
            Which series to attach entry/exit markers to (default 0 = first series).
        buy_color / sell_color : str, optional
            Marker colours.  Defaults to the theme's candlestick up/down colours.

        Examples
        --------
        ::

            from signum import Chart, sfera

            ohlc  = sfera.ohlc("CAC", start="2015-01-01")
            close = sfera.total_return("CACT", start="2015-01-01")

            chart = (
                Chart(theme="dark", height=500)
                .candlestick(ohlc)
                .forecast(pred_df, close=close, threshold=1.015)
            )
        """
        # ── 1. Detect prediction columns ──────────────────────────────────
        if pred_cols is None:
            pred_cols = sorted(
                [c for c in pred_df.columns if re.match(r"^d\d+$", c)],
                key=lambda c: int(c[1:]),
            )
        if not pred_cols:
            raise ValueError(
                "forecast(): no prediction columns found.  "
                "Name them 'd1', 'd2', … or pass pred_cols explicitly."
            )
        n_steps = len(pred_cols)

        # ── 2. Convert ratios → absolute prices ───────────────────────────
        if close is not None:
            common = pred_df.index.intersection(close.index)
            abs_df = pred_df.loc[common, pred_cols].multiply(
                close.loc[common], axis=0
            )
        else:
            common = pred_df.index
            abs_df = pred_df[pred_cols].copy()

        # ── 3. Build signal series (per-date: 1 = signal ON) ──────────────
        up_clr = buy_color or self._theme.get("candlestick", {}).get("upColor", "#26a69a")
        dn_clr = sell_color or self._theme.get("candlestick", {}).get("downColor", "#ef5350")

        if threshold is not None and close is not None:
            # Signal fires when max predicted ratio >= threshold
            signal = (pred_df.loc[common, pred_cols].max(axis=1) >= threshold).astype(int)
        else:
            signal = pd.Series(0, index=common)

        # ── 4. Entry / exit markers at state transitions ───────────────────
        prev = 0
        for date, val in signal.items():
            t_str = date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date)
            if val == 1 and prev == 0:
                self._markers.setdefault(series_index, []).append({
                    "time": t_str,
                    "position": "belowBar",
                    "shape": "arrowUp",
                    "color": up_clr,
                    "text": "",
                })
            elif val == 0 and prev == 1:
                self._markers.setdefault(series_index, []).append({
                    "time": t_str,
                    "position": "aboveBar",
                    "shape": "arrowDown",
                    "color": dn_clr,
                    "text": "",
                })
            prev = val

        # ── 5. Background shading during active-signal periods ─────────────
        if shade and signal.any():
            hex_c = up_clr.lstrip("#")
            if len(hex_c) == 6:
                r, g, b = int(hex_c[0:2], 16), int(hex_c[2:4], 16), int(hex_c[4:6], 16)
            else:
                r, g, b = 38, 166, 154  # fallback teal
            fill = f"rgba({r},{g},{b},0.08)"
            line_clr = f"rgba({r},{g},{b},0.16)"
            shade_records = [
                {"time": (d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)),
                 "value": int(v)}
                for d, v in signal.items()
            ]
            shade_opts = {
                "priceScaleId": "_fc_shade",
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
                "data": shade_records,
                "options": shade_opts,
                "price_scale": {
                    "id": "_fc_shade",
                    "scaleMargins": {"top": 0, "bottom": 0},
                    "pin": (0, 1),
                },
            })

        # ── 6. Forecast path for the latest date ──────────────────────────
        if path == "latest" and not abs_df.empty:
            last_date = abs_df.index[-1]
            last_prices = abs_df.iloc[-1]  # Series: d1..dN → absolute prices

            # Forward dates: N business days starting the day after last_date
            fwd_dates = pd.bdate_range(
                start=last_date + pd.Timedelta("1D"),
                periods=n_steps,
            )

            # Anchor point = close on last_date (start of path)
            if close is not None and last_date in close.index:
                anchor_price = float(close.loc[last_date])
            else:
                anchor_price = float(last_prices.iloc[0])  # fallback

            path_data = [{"time": last_date.strftime("%Y-%m-%d"), "value": anchor_price}]
            for fwd_d, price in zip(fwd_dates, last_prices.values):
                if np.isfinite(price):
                    path_data.append({"time": fwd_d.strftime("%Y-%m-%d"), "value": float(price)})

            pc = path_color or up_clr
            path_opts = {
                "color": pc,
                "lineWidth": 2,
                "lineStyle": 1,  # dashed
                "crosshairMarkerVisible": True,
                "crosshairMarkerRadius": 5,
                "lastValueVisible": True,
                "priceLineVisible": False,
                "title": "forecast",
            }
            self._series.append({
                "type": "LineSeries",
                "data": path_data,
                "options": path_opts,
            })

        return self

    def projection_cone(
        self,
        equity,
        embargo,
        value_col: Optional[str] = None,
        paths: int = 2000,
        block: int = 20,
        horizon: Optional[int] = None,
        ruin: float = 0.2,
        theoretical=None,
        realized: bool = True,
        slider: bool = True,
        seed: int = 42,
        color: Optional[str] = None,
        realized_color: Optional[str] = None,
        theoretical_color: Optional[str] = None,
        label: Optional[str] = None,
    ) -> "Chart":
        """Project an equity curve past an embargo date and overlay what really happened.

        Everything before the embargo is the record; a block bootstrap of those
        returns says what the future *should* look like if the record holds,
        drawn as a cone (5–95% and 25–75% bands around the median).  The curve
        after the embargo is then drawn over the cone, rebased to 100 at the
        embargo, so you see at a glance whether the live period sits where the
        record said it would.

        Usage::

            Chart(theme="light", height=420).projection_cone(nav, embargo="2026-05-19")

        Drag the embargo slider under the chart and the cone re-bootstraps
        live: pull it back a year and you can see whether the strategy stayed
        inside its own cone through the past — a walk-forward test in one drag.

        Parameters
        ----------
        equity : Series or DataFrame
            NAV / equity curve with a time column or DatetimeIndex.
        embargo : str, Timestamp, int or float
            Where the record ends.  A date, a bar index (negative counts from
            the end) or a fraction of the series in (0, 1).
        paths, block : int
            Bootstrap size and block length in bars.  Blocks keep short-run
            autocorrelation that iid resampling would destroy.
        horizon : int, optional
            Bars to project.  Default: every bar available after the embargo,
            so the cone ends where the data ends.  Pass a number to project
            beyond the last bar (dates are extended by business days).
        ruin : float
            Drawdown from the embargo level that counts as ruin, e.g. ``0.2``.
            ``P(ruin)`` in the header is the share of paths that ever touch it.
        theoretical : Series, optional
            A second post-embargo curve (a model's own expectation, a no-cost
            version) drawn dashed for comparison, rebased the same way.
        realized : bool
            Initial state of the in-chart ``Realized`` toggle.  ``False`` opens
            blind: nothing after the embargo is drawn (the theoretical line
            included), only the cone, until you click to reveal.
        slider : bool
            Embargo slider under the chart.
        seed : int
            Fixed, so the cone is the same on every render and does not
            shimmer while the slider moves.
        color, realized_color : str, optional
            Record + cone colour, and the realized path colour.  Default: the
            theme's first and second line colours.
        label : str, optional
            Strategy name, shown as the first entry of the key.

        Notes
        -----
        The chart is self-contained: the bootstrap runs in the page, not in
        Python, which is what lets the embargo move without a callback.
        """
        if not (0 < ruin < 1):
            raise ValueError("ruin must be a drawdown fraction in (0, 1)")
        if paths < 100 or block < 1:
            raise ValueError("paths must be >= 100 and block >= 1")

        df = self._prepare_time(equity)
        vcol = self._find_value_col(df, value_col)
        eq = pd.to_numeric(df[vcol], errors="coerce")
        keep = eq.notna() & (eq > 0)
        w = (pd.DataFrame({"time": df.loc[keep, "time"], "v": eq[keep].astype(float)})
             .sort_values("time").drop_duplicates("time", keep="last"))
        times = w["time"].tolist()
        vals = w["v"].tolist()
        n = len(vals)
        e_min = max(3 * block, 30)
        if n < e_min + 2:
            raise ValueError(f"projection_cone(): need at least {e_min + 2} bars, got {n}")
        e_max = n - 2

        # ── Resolve the embargo to a bar index ────────────────────────────
        if isinstance(embargo, bool):
            raise TypeError("embargo must be a date, a bar index or a fraction")
        if isinstance(embargo, (int, np.integer)):
            e0 = int(embargo) if embargo >= 0 else n + int(embargo)
        elif isinstance(embargo, float):
            if not (0 < embargo < 1):
                raise ValueError("a float embargo is a fraction of the series in (0, 1)")
            e0 = int(round(n * embargo))
        else:
            key = pd.Timestamp(embargo).strftime("%Y-%m-%d")
            e0 = int(np.searchsorted(np.array(times), key))
        if not (e_min <= e0 <= e_max):
            raise ValueError(
                f"embargo resolves to bar {e0}; it must be within [{e_min}, {e_max}] "
                f"({times[e_min]} … {times[e_max]}) so there is a record to bootstrap "
                f"from and at least one bar after it"
            )

        # ── Optional second curve, aligned on the same dates ──────────────
        theo = None
        if theoretical is not None:
            td = self._prepare_time(theoretical)
            tcol = self._find_value_col(td, None)
            look = dict(zip(td["time"], pd.to_numeric(td[tcol], errors="coerce")))
            theo = [None if (t not in look or pd.isna(look[t]) or look[t] <= 0)
                    else float(look[t]) for t in times]

        # A fixed horizon may run past the data; extend the axis so it can.
        times_ext = list(times)
        if horizon:
            ext = pd.bdate_range(pd.Timestamp(times[-1]) + pd.Timedelta("1D"), periods=int(horizon))
            times_ext += [d.strftime("%Y-%m-%d") for d in ext]

        # Colours come off the theme's line palette in order, the way line()
        # picks them: the record and its cone share the first (the cone is the
        # record continued), the realized path takes the second so it reads
        # against the cone in whatever palette the theme has.
        is_dark = self._theme_name in ("dark", "midnight", "glass")
        primary = color or self._next_line_color()
        second = realized_color or self._next_line_color()
        colors = {
            "pre": primary,
            "median": primary,
            "band": primary,
            "realized": second,
            "theo": theoretical_color or ("rgba(255,255,255,0.45)" if is_dark else "rgba(0,0,0,0.38)"),
            "shade": "rgba(255,255,255,0.05)" if is_dark else "rgba(0,0,0,0.04)",
        }

        # Series are created empty; the page fills every one of them from the
        # bootstrap, because the embargo can move after the fact.
        quiet = {"priceLineVisible": False, "lastValueVisible": False,
                 "crosshairMarkerVisible": False}
        idx: Dict[str, Optional[int]] = {}

        idx["shade"] = len(self._series)
        self._series.append({
            "type": "AreaSeries", "data": [],
            "options": {**quiet, "priceScaleId": "_cone_shade", "lineWidth": 1,
                        "lineColor": "rgba(0,0,0,0)", "lineType": 2,
                        "topColor": colors["shade"], "bottomColor": colors["shade"],
                        "pointMarkersVisible": False},
            "price_scale": {"id": "_cone_shade", "scaleMargins": {"top": 0, "bottom": 0},
                            "pin": (0, 1)},
        })
        idx["pre"] = len(self._series)
        self._series.append({
            "type": "LineSeries", "data": [],
            "options": {**quiet, "color": colors["pre"], "lineWidth": 2},
        })
        idx["theo"] = None
        if theo is not None:
            idx["theo"] = len(self._series)
            self._series.append({
                "type": "LineSeries", "data": [],
                "options": {**quiet, "color": colors["theo"], "lineWidth": 1,
                            "lineStyle": 1},
            })
        idx["median"] = len(self._series)
        self._series.append({
            "type": "LineSeries", "data": [],
            "options": {**quiet, "color": colors["median"], "lineWidth": 2},
        })
        idx["realized"] = len(self._series)
        self._series.append({
            "type": "LineSeries", "data": [],
            "options": {**quiet, "color": colors["realized"], "lineWidth": 2,
                        "crosshairMarkerVisible": True},
        })

        self._cone_config = {
            "times": times_ext,
            "vals": vals,
            "theo": theo,
            "n": n,
            "e0": e0,
            "e_min": e_min,
            "e_max": e_max,
            "paths": int(paths),
            "block": int(block),
            "horizon": int(horizon or 0),
            "ruin": float(ruin),
            "seed": int(seed),
            "realized": bool(realized),
            "slider": bool(slider),
            "label": label or "",
            "colors": colors,
            "idx": idx,
        }
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

    def _get_formatter_js(self) -> str:
        """Return raw JS function literal for price formatting, or empty string."""
        if not self._y_format:
            # DEFAULT: trim trailing zeros (200.00 -> 200, 100.10 -> 100.1) and thousand-separate large
            # values, instead of the charting lib's fixed 2-decimal default that printed "0.00" on every
            # axis. Real (non-zero) decimals are kept; only the all-zero tail is dropped.
            return ("function(p){var a=Math.abs(p);"
                    "if(a>=1000)return Math.round(p).toLocaleString('en-US');"
                    "var v=Math.round(p*100)/100;return ''+v;}")
        if self._y_format == "kmb":
            return (
                "function(p){var a=Math.abs(p);"
                "if(a>=1e9)return(p/1e9).toFixed(2)+'B';"
                "if(a>=1e6)return(p/1e6).toFixed(2)+'M';"
                "if(a>=1e3)return(p/1e3).toFixed(1)+'K';"
                "return p.toFixed(0);}"
            )
        if self._y_format == "percent":
            return "function(p){return p.toFixed(1)+'%';}"
        if self._y_format == "num":
            # thousand-separated integers for |v|>=1000; trimmed (no trailing-zero) decimals otherwise
            return ("function(p){var a=Math.abs(p);"
                    "if(a>=1000)return Math.round(p).toLocaleString('en-US');"
                    "var v=Math.round(p*100)/100;return ''+v;}")
        return ""

    @staticmethod
    def _build_scale_switch_html(is_dark: bool, selected: str = "AUTO") -> str:
        """Return floating overlay unit-switch. Position:absolute over the chart — takes no layout space."""
        fg = "rgba(255,255,255,0.75)" if is_dark else "rgba(0,0,0,0.72)"
        bg = "rgba(20,20,30,0.78)" if is_dark else "rgba(248,248,248,0.90)"
        br = "rgba(255,255,255,0.14)" if is_dark else "rgba(0,0,0,0.12)"
        btn_c = "rgba(255,255,255,0.65)" if is_dark else "rgba(0,0,0,0.55)"
        return (
            '<div id="yscale-overlay" style="position:absolute;top:8px;right:8px;z-index:10;'
            'display:flex;flex-direction:column;align-items:flex-end;gap:4px">'
            f'<button id="yscale-btn" title="Y-axis units" '
            f'onclick="var p=document.getElementById(\'yscale-panel\');p.style.display=p.style.display===\'none\'?\'flex\':\'none\'" '
            f'style="background:none;border:none;cursor:pointer;padding:3px 5px;color:{btn_c};'
            f'font-size:14px;line-height:1;opacity:0.28;transition:opacity 0.15s;border-radius:4px" '
            f'onmouseenter="this.style.opacity=\'0.90\'" onmouseleave="this.style.opacity=\'0.28\'">&#9881;</button>'
            f'<div id="yscale-panel" style="display:none;align-items:center;gap:6px;'
            f'background:{bg};border:1px solid {br};border-radius:7px;padding:4px 8px;'
            f'box-shadow:0 2px 8px rgba(0,0,0,0.20)">'
            f'<span style="color:{fg};font:10px/1 sans-serif;white-space:nowrap">Units</span>'
            f'<select id="yscale-select" style="height:22px;min-width:76px;border-radius:5px;'
            f'border:1px solid {br};background:{bg};color:{fg};font:10px/1 sans-serif;'
            f'padding:2px 6px;outline:none">'
            f'<option value="AUTO" {"selected" if selected == "AUTO" else ""}>Auto</option>'
            f'<option value="K" {"selected" if selected == "K" else ""}>Thousands</option>'
            f'<option value="M" {"selected" if selected == "M" else ""}>Millions</option>'
            f'<option value="B" {"selected" if selected == "B" else ""}>Billions</option>'
            f'<option value="RAW" {"selected" if selected == "RAW" else ""}>Raw</option>'
            '</select>'
            '</div>'
            '</div>'
        )

    def _build_y_format_js(self, chart_var: str = "chart") -> str:
        """Return JS snippet to apply a custom y-axis price formatter."""
        fmt = self._get_formatter_js()
        if not fmt:
            return ""
        return f'{chart_var}.applyOptions({{localization:{{priceFormatter:{fmt}}}}});'

    @staticmethod
    def _inject_formatter_into_opts(opts_json: str, formatter_js: str) -> str:
        """Inject a localization.priceFormatter into a JSON options string."""
        if not formatter_js:
            return opts_json
        # Insert before the final closing brace
        return opts_json[:-1] + ',"localization":{"priceFormatter":' + formatter_js + '}}'

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
            is_dark = self._theme_name in ("dark", "midnight", "glass")
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
            # Special handling for price scale configuration (not a real series)
            if s.get("type") == "__price_scale_config__":
                if s.get("formatter") == "percent":
                    # Add percentage formatter
                    lines.append(
                        f"{chart_var}.applyOptions({{"
                        f"localization: {{"
                        f"priceFormatter: (price) => price.toFixed(0) + '%'"
                        f"}}}});"
                    )
                elif s.get("formatter") == "value":
                    # Value (NAV) axis — plain number, k-suffix for thousands
                    lines.append(
                        f"{chart_var}.applyOptions({{"
                        f"localization: {{"
                        f"priceFormatter: (price) => (Math.abs(price) >= 1000 ? (price/1000).toFixed(1)+'k' : price.toFixed(0))"
                        f"}}}});"
                    )
                lines.append(
                    f"{chart_var}.priceScale('{s['scale_id']}').applyOptions("
                    f"{self._json(s['options'])});"
                )
                continue

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
                # A 0/1 shading series autoscales to whatever is on screen: scroll
                # to a stretch that is all zeros and 0 lands mid-pane, so the fill
                # covers the lower half.  Pin the range and 0 stays on the floor.
                if "pin" in ps:
                    lo, hi = ps["pin"]
                    lines.append(
                        f"{var}.applyOptions({{autoscaleInfoProvider: () => "
                        f"({{priceRange: {{minValue: {lo}, maxValue: {hi}}}}})}});"
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

    # ── Seasonality overlay ───────────────────────────────────────────────

    _SEASONALITY_JS = r"""
    // ── Seasonality: season selection, average, percent/value swap ─────
    (function(){
      const CFG = __CFG__;
      const SER = [__SERIES__];
      const AVG = __AVG__;
      const N   = SER.length;
      let lo = CFG.sel[0], hi = CFG.sel[1];
      let mode = CFG.mode;
      let avgOn = CFG.average_on;

      const PCT_FMT = function(p){ return p.toFixed(1) + '%'; };
      const VAL_FMT = __VALFMT__;
      const MON = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
      const BASE = Date.UTC(2001, 0, 1);

      // LC hands time back as a string, a BusinessDay or a UTC stamp depending
      // on how the series was fed - normalise all three.
      function _tp(t){
        if (typeof t === 'string'){ const a = t.split('-'); return {y:+a[0], m:+a[1], d:+a[2]}; }
        if (t && t.year !== undefined) return {y:t.year, m:t.month, d:t.day};
        const dd = new Date(t * 1000);
        return {y:dd.getUTCFullYear(), m:dd.getUTCMonth()+1, d:dd.getUTCDate()};
      }
      function _off(t){ const p = _tp(t); return Math.round((Date.UTC(p.y, p.m-1, p.d) - BASE) / 86400000); }

      // The x axis is synthetic: year 2001 is a carrier, never a real date.
      const TICK_FMT = CFG.ordinal
        ? function(t){ return CFG.unit + (_off(t) + 1); }
        : function(t, tt){ const p = _tp(t); return (tt <= 1) ? MON[p.m-1] : (p.d + ' ' + MON[p.m-1]); };
      const TIME_FMT = CFG.ordinal
        ? function(t){ return CFG.unit + (_off(t) + 1); }
        : function(t){ const p = _tp(t); return p.d + ' ' + MON[p.m-1]; };

      function _src(i){ return (mode === 'percent') ? CFG.pct[i] : CFG.val[i]; }
      function _rows(i){
        const s = _src(i), out = [];
        for (let k = 0; k < CFG.grid.length; k++){
          const v = s[k];
          if (v !== null && v !== undefined) out.push({time: CFG.grid[k], value: v});
        }
        return out;
      }
      // Carry each season's last value forward inside its own span, so the
      // set of seasons in the mean does not change with which years happened
      // to trade on a given calendar day (that is what made it a sawtooth).
      function _avgRows(){
        const out = [], last = new Array(N).fill(null);
        for (let k = 0; k < CFG.grid.length; k++){
          let sum = 0, c = 0;
          for (let i = lo; i <= hi; i++){
            const v = _src(i)[k];
            if (v !== null && v !== undefined) last[i] = v;
            if (last[i] !== null && k >= CFG.spans[i][0] && k <= CFG.spans[i][1]){ sum += last[i]; c++; }
          }
          if (c > 0) out.push({time: CFG.grid[k], value: sum / c});
        }
        return out;
      }
      function _applyFmt(){
        chart.applyOptions({
          localization: { priceFormatter: (mode === 'percent') ? PCT_FMT : VAL_FMT, timeFormatter: TIME_FMT },
          timeScale: { tickMarkFormatter: TICK_FMT }
        });
      }
      // Brush moves only toggle visibility - the data itself is unchanged, so
      // there is no reason to push it again on every pointermove.
      function _applySel(){
        for (let i = 0; i < N; i++){
          const on = (i >= lo && i <= hi);
          SER[i].applyOptions({ visible: on });
          const k = document.getElementById('sb-key-' + i);
          if (k) k.style.opacity = on ? '1' : '0.28';
        }
        if (AVG) AVG.setData(_avgRows());
      }
      function _applyMode(){
        for (let i = 0; i < N; i++) SER[i].setData(_rows(i));
        if (AVG) AVG.setData(_avgRows());
        _applyFmt();
        // Percent and absolute live on different scales entirely. Any earlier
        // drag on the price axis turned autoscale off, and the new data would
        // then sit outside the old range - out of sight. Force it back on.
        chart.priceScale('right').applyOptions({ autoScale: true });
        chart.timeScale().fitContent();
      }

      const track = document.getElementById('sb-track');
      function _pc(i){ return N > 1 ? (i / (N - 1)) * 100 : 0; }
      function _paint(){
        if (!track) return;
        const h0 = document.getElementById('sb-h0'), h1 = document.getElementById('sb-h1');
        const fill = document.getElementById('sb-fill');
        h0.style.left = _pc(lo) + '%';
        h1.style.left = _pc(hi) + '%';
        fill.style.left = _pc(lo) + '%';
        fill.style.width = (_pc(hi) - _pc(lo)) + '%';
      }
      function _at(e){
        const r = track.getBoundingClientRect();
        let f = (e.clientX - r.left) / Math.max(1, r.width);
        f = Math.min(1, Math.max(0, f));
        return Math.round(f * (N - 1));
      }
      function _grab(el, which){
        el.addEventListener('pointerdown', function(e){
          e.preventDefault();
          e.stopPropagation();
          try { el.setPointerCapture(e.pointerId); } catch (err) {}
          const move = function(ev){
            const i = _at(ev);
            if (which === 0) lo = Math.min(i, hi); else hi = Math.max(i, lo);
            _paint(); _applySel();
          };
          const up = function(){
            el.removeEventListener('pointermove', move);
            el.removeEventListener('pointerup', up);
            el.removeEventListener('pointercancel', up);
          };
          el.addEventListener('pointermove', move);
          el.addEventListener('pointerup', up);
          el.addEventListener('pointercancel', up);
        });
      }
      if (track){
        _grab(document.getElementById('sb-h0'), 0);
        _grab(document.getElementById('sb-h1'), 1);
        // Clicking the bare track jumps whichever handle is nearer.
        track.addEventListener('pointerdown', function(e){
          const i = _at(e);
          if (Math.abs(i - lo) <= Math.abs(i - hi)) lo = Math.min(i, hi); else hi = Math.max(i, lo);
          _paint(); _applySel();
        });
      }

      const modeSel = document.getElementById('sb-mode');
      if (modeSel){
        modeSel.value = mode;
        modeSel.addEventListener('change', function(){ mode = this.value; _applyMode(); });
      }
      const avgBtn = document.getElementById('sb-avg');
      if (avgBtn && AVG){
        const _paintAvg = function(){
          avgBtn.style.opacity = avgOn ? '1' : '0.42';
          AVG.applyOptions({ visible: avgOn });
        };
        avgBtn.addEventListener('click', function(){ avgOn = !avgOn; _paintAvg(); });
        _paintAvg();
      }

      _paint();
      _applySel();   // the initial window may be narrower than the data
      _applyFmt();
    })();
"""

    def _build_seasonality(self, is_dark: bool):
        """Return (html, js, extra_height) for the season brush + controls."""
        cfg = self._seasonality_config
        if not cfg:
            return "", "", 0

        fg = "rgba(255,255,255,0.88)" if is_dark else "rgba(0,0,0,0.78)"
        dim = "rgba(255,255,255,0.45)" if is_dark else "rgba(0,0,0,0.40)"
        br = "rgba(255,255,255,0.16)" if is_dark else "rgba(0,0,0,0.14)"
        panel = "rgba(255,255,255,0.06)" if is_dark else "rgba(0,0,0,0.05)"
        rail = "rgba(255,255,255,0.14)" if is_dark else "rgba(0,0,0,0.12)"
        knob = "#e8e8ec" if is_dark else "#1e1e22"
        mono = "font:11px/1 'SF Mono','Consolas',monospace"

        labels = cfg["labels"]
        n = len(labels)
        has_avg = cfg["avg_index"] is not None
        # A native <select> paints its dropdown in its own background colour, so
        # a translucent one leaves the options unreadable - use the canvas colour.
        solid = (self._theme.get("chart", {}).get("layout", {})
                 .get("background", {}).get("color", ""))
        if not solid.startswith("#"):
            solid = "#1e1e1e" if is_dark else "#ffffff"

        # ── Season brush: the only thing above the chart ──────────────────
        brush = ""
        extra = 0
        if cfg["brush"]:
            extra = 40
            step = 1 if n <= 12 else max(1, -(-n // 10))
            shown = sorted(set(list(range(0, n, step)) + [n - 1]))
            ticks = "".join(
                f'<span style="position:absolute;left:{(i / (n - 1)) * 100:.4f}%;'
                f'transform:translateX(-50%);color:{dim};{mono};'
                f'white-space:nowrap;pointer-events:none">{html_module.escape(labels[i])}</span>'
                for i in shown
            )
            knob_css = (
                f"position:absolute;top:50%;width:13px;height:13px;margin:-7px 0 0 -7px;"
                f"border-radius:50%;background:{knob};border:1px solid {br};"
                f"box-shadow:0 1px 3px rgba(0,0,0,0.35);cursor:ew-resize;touch-action:none;z-index:2"
            )
            brush = (
                # 21px of side padding: 12 to line up with the overlays below,
                # plus 9 so a handle parked at either end is not clipped.
                f'<div id="sb-wrap" style="position:relative;z-index:6;height:{extra}px;'
                f'padding:3px 21px 0">'
                f'<div style="position:relative;height:14px">{ticks}</div>'
                f'<div id="sb-track" style="position:relative;height:20px;cursor:pointer;'
                f'touch-action:none">'
                f'<div style="position:absolute;top:50%;left:0;right:0;height:4px;'
                f'margin-top:-2px;border-radius:2px;background:{rail}"></div>'
                f'<div id="sb-fill" style="position:absolute;top:50%;height:4px;'
                f'margin-top:-2px;border-radius:2px;background:{fg}"></div>'
                f'<div id="sb-h0" style="{knob_css}"></div>'
                f'<div id="sb-h1" style="{knob_css}"></div>'
                f'</div></div>'
            )

        # ── Overlays inside the plot, just under the brush ────────────────
        # Same idiom as the y-scale gear: absolutely positioned over the canvas,
        # so they take no height and stop competing with the brush for the top.
        top = extra + 8
        small = "font:10px/1 'SF Mono','Consolas',monospace"

        # Colour key, top-left: replaces the right-axis "year value" badges.
        # Entries dim as the brush hides their season.
        swatches = "".join(
            f'<span id="sb-key-{i}" style="display:inline-flex;align-items:center;gap:4px;'
            f'transition:opacity 0.15s">'
            f'<i style="width:10px;height:3px;border-radius:2px;background:{cfg["colors"][i]}"></i>'
            f'{html_module.escape(labels[i])}</span>'
            for i in range(n)
        )
        # Controls follow the key on the same row.  Top-right put them over the
        # price axis; here they are clear of it whatever width the axis takes.
        pill = (f"border:1px solid {br};border-radius:4px;color:{fg};{small};"
                f"padding:3px 6px;outline:none;cursor:pointer")
        ctl = ('<div id="sb-ctl" style="display:inline-flex;align-items:center;gap:5px;'
               'margin-left:6px;pointer-events:auto">')
        if has_avg:
            ctl += (
                f'<button id="sb-avg" title="Mean across the selected seasons" '
                f'style="{pill};background:{panel}">Avg</button>'
            )
        ctl += (
            f'<select id="sb-mode" title="Y-axis scale" style="{pill};background:{solid}">'
            f'<option value="percent">%</option>'
            f'<option value="value">Abs</option>'
            f'</select></div>'
        )
        key = (
            f'<div id="sb-key" style="position:absolute;top:{top}px;left:12px;z-index:10;'
            f'right:80px;display:flex;flex-wrap:wrap;align-items:center;gap:4px 10px;'
            f'color:{dim};{small};pointer-events:none">{swatches}{ctl}</div>'
        )

        html = brush + key

        b, avg_i = cfg["base_index"], cfg["avg_index"]
        series_vars = ",".join(f"s{b + i}" for i in range(n))
        val_fmt = self._get_formatter_js() or (
            "function(p){var a=Math.abs(p);"
            "if(a>=1000)return Math.round(p).toLocaleString('en-US');"
            "return ''+(Math.round(p*100)/100);}"
        )
        payload = {k: cfg[k] for k in
                   ("grid", "labels", "spans", "pct", "val", "mode", "average_on",
                    "ordinal", "unit", "sel")}
        js = (
            self._SEASONALITY_JS
            .replace("__CFG__", self._json(payload))
            .replace("__SERIES__", series_vars)
            .replace("__AVG__", f"s{avg_i}" if avg_i is not None else "null")
            .replace("__VALFMT__", val_fmt)
        )
        return html, js, extra

    # ── Projection cone ───────────────────────────────────────────────────

    _CONE_JS = r"""
    // ── Projection cone: block bootstrap + band primitive + embargo slider ──
    (function(){
      const C = __CFG__;
      const PRE = __PRE__, REAL = __REAL__, MED = __MED__, THEO = __THEO__, SHADE = __SHADE__;
      const N = C.n, V = C.vals, D = C.times;
      const R = new Float64Array(N);
      for (let i = 1; i < N; i++) R[i] = V[i] / V[i-1] - 1;
      let e = C.e0;
      let reveal = C.realized;   // post-embargo data shown, or embargoed

      // Small, fast, seeded: the same embargo always gives the same cone.
      function mulberry32(a){
        return function(){
          a |= 0; a = a + 0x6D2B79F5 | 0;
          let t = Math.imul(a ^ a >>> 15, 1 | a);
          t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
          return ((t ^ t >>> 14) >>> 0) / 4294967296;
        };
      }

      // Circular block bootstrap of the returns up to and including bar e,
      // streamed one step at a time so memory is O(paths), not O(paths x H).
      function bootstrap(e, P){
        const pool = R.subarray(1, e + 1), n = pool.length;
        const B = Math.max(1, Math.min(C.block, n));
        const H = C.horizon > 0 ? C.horizon : (N - 1 - e);
        const rnd = mulberry32(C.seed);
        const lvl = new Float64Array(P).fill(100), low = new Float64Array(P).fill(100);
        const pos = new Int32Array(P), left = new Int32Array(P), tmp = new Float64Array(P);
        const q = { t: [D[e]], p5: [100], p25: [100], p50: [100], p75: [100], p95: [100] };
        const at = f => tmp[Math.round(f * (P - 1))];
        for (let k = 1; k <= H; k++){
          for (let p = 0; p < P; p++){
            if (left[p] === 0){ pos[p] = (rnd() * n) | 0; left[p] = B; }
            lvl[p] *= 1 + pool[pos[p]];
            pos[p] = (pos[p] + 1) % n; left[p]--;
            if (lvl[p] < low[p]) low[p] = lvl[p];
            tmp[p] = lvl[p];
          }
          tmp.sort();
          q.t.push(D[e + k]);
          q.p5.push(at(0.05)); q.p25.push(at(0.25)); q.p50.push(at(0.50));
          q.p75.push(at(0.75)); q.p95.push(at(0.95));
        }
        let ruined = 0;
        const floor = 100 * (1 - C.ruin);
        for (let p = 0; p < P; p++) if (low[p] <= floor) ruined++;
        return { q: q, H: H, pruin: ruined / P };
      }

      // Bands and the embargo line are drawn by a series primitive on the
      // median, in that series' own price/time coordinate space.
      const band = {
        _q: null, _chart: null, _series: null, _req: null,
        set(q){ this._q = q; if (this._req) this._req(); },
        attached(p){ this._chart = p.chart; this._series = p.series; this._req = p.requestUpdate; },
        detached(){ this._chart = null; this._series = null; this._req = null; },
        updateAllViews(){},
        paneViews(){ return [this._view]; },
      };
      function _poly(ctx, q, lo, hi, alpha){
        const ts = band._chart.timeScale(), s = band._series;
        const xs = [], yl = [], yh = [];
        for (let i = 0; i < q.t.length; i++){
          const x = ts.timeToCoordinate(q.t[i]);
          const a = s.priceToCoordinate(q[lo][i]), b = s.priceToCoordinate(q[hi][i]);
          if (x === null || a === null || b === null) continue;
          xs.push(x); yl.push(a); yh.push(b);
        }
        if (xs.length < 2) return;
        ctx.beginPath();
        ctx.moveTo(xs[0], yh[0]);
        for (let i = 1; i < xs.length; i++) ctx.lineTo(xs[i], yh[i]);
        for (let i = xs.length - 1; i >= 0; i--) ctx.lineTo(xs[i], yl[i]);
        ctx.closePath();
        ctx.globalAlpha = alpha; ctx.fillStyle = C.colors.band; ctx.fill(); ctx.globalAlpha = 1;
      }
      band._view = {
        zOrder(){ return 'bottom'; },
        renderer(){
          return { draw(target){
            target.useMediaCoordinateSpace(function(scope){
              const ctx = scope.context, q = band._q;
              if (!q || !band._chart) return;
              _poly(ctx, q, 'p5', 'p95', 0.10);
              _poly(ctx, q, 'p25', 'p75', 0.22);
              const x = band._chart.timeScale().timeToCoordinate(D[e]);
              if (x !== null){
                ctx.save();
                ctx.setLineDash([4, 4]); ctx.lineWidth = 1;
                ctx.strokeStyle = C.colors.pre; ctx.globalAlpha = 0.7;
                ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, scope.mediaSize.height); ctx.stroke();
                ctx.restore();
              }
            });
          } };
        },
      };
      MED.attachPrimitive(band);

      function rows(from, to, base, arr){
        const out = [];
        for (let i = from; i <= to; i++){
          if (arr[i] === null || arr[i] === undefined) continue;
          out.push({ time: D[i], value: arr[i] / base * 100 });
        }
        return out;
      }
      const pct = v => (v >= 0 ? '+' : '−') + Math.abs(Math.round(v)) + '%';
      const stats = document.getElementById('pc-stats');
      const lbl = document.getElementById('pc-lbl');

      // While the slider is moving a 300-path preview keeps the frame rate up
      // over long horizons; the full set is drawn on release.
      const PREVIEW = Math.min(300, C.paths);
      function render(P){
        const base = V[e];
        const bs = bootstrap(e, P || C.paths);
        const end = e + bs.H;
        PRE.setData(rows(0, e, base, V));
        MED.setData(bs.q.t.map((t, i) => ({ time: t, value: bs.q.p50[i] })));
        band.set(bs.q);
        _postEmbargo();
        const sh = [];
        for (let i = 0; i <= Math.max(N - 1, end); i++) sh.push({ time: D[i], value: i >= e ? 1 : 0 });
        SHADE.setData(sh);
        const L = bs.q.p50.length - 1;
        if (stats){
          stats.textContent =
            'median ' + pct(bs.q.p50[L] - 100) + ' · P5 ' + pct(bs.q.p5[L] - 100)
            + ' · P95 ' + pct(bs.q.p95[L] - 100) + ' · P(ruin) ' + Math.round(bs.pruin * 100) + '%';
          stats.title = 'At the end of the horizon (' + bs.H + ' bars), over ' + (P || C.paths)
            + ' block-bootstrap paths (block ' + C.block + ') of the ' + e + ' returns before the embargo.'
            + ' P(ruin): share of paths that ever fell ' + Math.round(C.ruin * 100) + '% below the embargo level.';
        }
        if (lbl) lbl.textContent = 'Embargo ' + D[e];
      }

      // Everything after the embargo is future information: one switch
      // hides or shows it all, leaving the cone as the only view forward.
      const rv = document.getElementById('pc-reveal');
      function _postEmbargo(){
        const base = V[e];
        REAL.setData(reveal ? rows(e, N - 1, base, V) : []);
        if (THEO && C.theo){
          const tb = C.theo[e];
          THEO.setData((reveal && tb) ? rows(e, N - 1, tb, C.theo) : []);
        }
        if (rv) rv.style.opacity = reveal ? '1' : '0.42';
      }
      if (rv) rv.addEventListener('click', function(){ reveal = !reveal; _postEmbargo(); });

      const sl = document.getElementById('pc-slider');
      if (sl){
        let pending = false;
        sl.addEventListener('input', function(){
          e = parseInt(this.value, 10);
          if (pending) return;
          pending = true;
          requestAnimationFrame(function(){ pending = false; render(PREVIEW); });
        });
        sl.addEventListener('change', function(){ e = parseInt(this.value, 10); render(); });
      }
      render();
    })();
"""

    def _build_cone(self, is_dark: bool):
        """Return (html, js, extra_height) for the projection-cone chrome."""
        cfg = self._cone_config
        if not cfg:
            return "", "", 0

        fg = "rgba(255,255,255,0.88)" if is_dark else "rgba(0,0,0,0.78)"
        dim = "rgba(255,255,255,0.50)" if is_dark else "rgba(0,0,0,0.45)"
        br = "rgba(255,255,255,0.16)" if is_dark else "rgba(0,0,0,0.14)"
        panel = "rgba(255,255,255,0.06)" if is_dark else "rgba(0,0,0,0.05)"
        mono = "font:11px/1 'SF Mono','Consolas',monospace"
        small = "font:10px/1 'SF Mono','Consolas',monospace"
        col = cfg["colors"]

        # ── Header: stats line + colour key, top-left inside the plot ─────
        def line(c, dashed=False):
            style = f"width:14px;height:0;border-top:2px {'dashed' if dashed else 'solid'} {c}"
            return f'<i style="display:inline-block;{style}"></i>'

        def box(c, alpha):
            return (f'<i style="display:inline-block;width:14px;height:8px;border-radius:2px;'
                    f'background:{c};opacity:{alpha}"></i>')

        items = [(line(col["realized"]), "realized"), (line(col["median"]), "median")]
        items.append((box(col["band"], 0.32), "25–75%"))
        items.append((box(col["band"], 0.16), "5–95%"))
        if cfg["idx"]["theo"] is not None:
            items.append((line(col["theo"], dashed=True), "theoretical"))
        key = "".join(
            f'<span style="display:inline-flex;align-items:center;gap:5px">{sw}{txt}</span>'
            for sw, txt in items
        )
        # The plot carries only the key (and the strategy name, quietly, as its
        # first entry).  The numbers live next to the slider that changes them;
        # without a slider they drop to a dim second line here instead.
        if cfg["label"]:
            key = (f'<span style="color:{fg}">{html_module.escape(cfg["label"])}</span>' + key)
        pill = (f"border:1px solid {br};border-radius:4px;color:{fg};{small};"
                f"padding:3px 6px;outline:none;cursor:pointer;background:{panel}")
        key += (f'<button id="pc-reveal" title="Show or hide everything after the embargo" '
                f'style="{pill};margin-left:6px;pointer-events:auto">Realized</button>')
        stats_span = f'<span id="pc-stats" style="color:{dim};{small};white-space:nowrap"></span>'
        header = (
            f'<div id="pc-hdr" style="position:absolute;top:8px;left:12px;right:70px;z-index:10;'
            f'display:flex;flex-direction:column;gap:5px;pointer-events:none">'
            f'<div style="display:flex;flex-wrap:wrap;align-items:center;gap:4px 12px;'
            f'color:{dim};{small}">{key}</div>'
            f'{"" if cfg["slider"] else stats_span}'
            f'</div>'
        )

        # ── Embargo slider + the numbers it drives, under the chart ───────
        bar, extra = "", 0
        if cfg["slider"]:
            extra = 36
            bar = (
                f'<div id="pc-bar" style="display:flex;align-items:center;justify-content:center;'
                f'gap:12px;padding:4px 16px;white-space:nowrap;height:36px">'
                f'<span id="pc-lbl" style="color:{fg};{mono}"></span>'
                f'<input id="pc-slider" type="range" min="{cfg["e_min"]}" max="{cfg["e_max"]}" '
                f'step="1" value="{cfg["e0"]}" title="Drag to move the embargo" '
                f'style="flex:1;max-width:420px;accent-color:{col["realized"]};cursor:ew-resize">'
                f'{stats_span}'
                f'</div>'
            )

        i = cfg["idx"]
        payload = {k: cfg[k] for k in ("times", "vals", "theo", "n", "e0", "paths", "block",
                                       "horizon", "ruin", "seed", "realized", "colors")}
        js = (
            self._CONE_JS
            .replace("__CFG__", self._json(payload))
            .replace("__PRE__", f"s{i['pre']}")
            .replace("__REAL__", f"s{i['realized']}")
            .replace("__MED__", f"s{i['median']}")
            .replace("__THEO__", f"s{i['theo']}" if i["theo"] is not None else "null")
            .replace("__SHADE__", f"s{i['shade']}")
        )
        return header + bar, js, extra

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
            # Only invert if the theme is actually light (glass is dark with a CSS gradient)
            if self._theme_name not in ("dark", "midnight", "glass"):
                _logo_invert = "filter:invert(1);"

        # ── Threshold slider components ───────────────────────────────────
        is_dark_bg = self._theme_name in ("dark", "midnight", "glass")
        slider_html = ""
        slider_js = ""
        slider_extra_height = 0
        _kmb_overlay = ""
        if self._y_format == "kmb":
            _kmb_overlay = self._build_scale_switch_html(is_dark_bg)
        if self._threshold_config:
            tc = self._threshold_config
            dec = tc["decimals"]
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
            is_dark_bg = self._theme_name in ("dark", "midnight", "glass")
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

        # ── Allocation tooltip overlay ────────────────────────────────────
        alloc_tooltip_html = ""
        alloc_tooltip_js = ""
        if self._alloc_tooltip:
            at = self._alloc_tooltip
            is_dark_bg = self._theme_name in ("dark", "midnight", "glass")
            box_bg = "rgba(10,10,26,0.85)" if is_dark_bg else "rgba(255,255,255,0.90)"
            lbl_c  = "rgba(255,255,255,0.72)" if is_dark_bg else "rgba(0,0,0,0.65)"
            val_c  = "rgba(255,255,255,0.95)" if is_dark_bg else "rgba(0,0,0,0.92)"
            date_c = "rgba(255,255,255,0.85)" if is_dark_bg else "rgba(0,0,0,0.78)"
            
            alloc_tooltip_html = (
                f'<div id="alloc-tooltip" style="position:absolute;top:12px;left:12px;z-index:7;'
                f'background:{box_bg};'
                f'backdrop-filter:blur(20px) saturate(180%);-webkit-backdrop-filter:blur(20px) saturate(180%);'
                f'border:1px solid rgba(255,255,255,0.12);'
                f'border-radius:8px;padding:8px 12px;display:none;'
                f'box-shadow:0 4px 16px rgba(0,0,0,0.3);min-width:148px;'
                f'max-height:calc(100% - 24px);box-sizing:border-box;overflow:hidden;">'
                f'<div id="alloc-date" style="color:{date_c};font:10px/1.3 \'SF Mono\',\'Consolas\',monospace;'
                f'margin-bottom:6px;padding-bottom:5px;border-bottom:1px solid rgba(255,255,255,0.08);'
                f'font-weight:600"></div>'
                f'<div id="alloc-items" style="font:10px/1.3 \'SF Mono\',\'Consolas\',monospace;'
                f'max-height:calc(100% - 64px);overflow-y:auto"></div>'
                f'<div id="alloc-totals" style="color:{lbl_c};font:10px/1.7 \'SF Mono\',\'Consolas\',monospace;'
                f'margin-top:8px;padding-top:6px;border-top:1px solid rgba(255,255,255,0.08)"></div>'
                f'</div>'
            )
            
            # Prepare allocation data as JSON for JavaScript
            import json
            alloc_data_json = json.dumps(at["data"])
            assets_json = json.dumps(at["assets"])
            colors_json = json.dumps(at["colors"])
            
            # Pre-compute color values to avoid quote escaping issues
            short_color = "#ef5350" if is_dark_bg else "#c62828"
            
            alloc_tooltip_js = "\n".join([
                "    // Allocation tooltip",
                f"    const allocData = {alloc_data_json};",
                f"    const allocAssets = {assets_json};",
                f"    const allocColors = {colors_json};",
                f"    const allocHideZero = {json.dumps(at['hide_zero'])};",
                "    const allocTooltip = document.getElementById('alloc-tooltip');",
                "    const allocDate = document.getElementById('alloc-date');",
                "    const allocItems = document.getElementById('alloc-items');",
                "    const allocTotals = document.getElementById('alloc-totals');",
                "    ",
                "    chart.subscribeCrosshairMove(function(param) {",
                "        if (!param.time || !param.point) {",
                "            allocTooltip.style.display = 'none';",
                "            return;",
                "        }",
                "        ",
                "        const dateKey = String(param.time);",
                "        const weights = allocData[dateKey];",
                "        ",
                "        if (!weights) {",
                "            allocTooltip.style.display = 'none';",
                "            return;",
                "        }",
                "        ",
                "        allocDate.textContent = '📅 ' + dateKey;",
                "        ",
                "        let html = '';",
                "        let netExp = 0, grossExp = 0;",
                "        allocAssets.forEach(function(asset, i) {",
                "            const w = weights[asset] || 0;",
                "            netExp += w;",
                "            grossExp += Math.abs(w);",
                "            if (allocHideZero && Math.abs(w) < 1e-9) return;   // hide zero positions on this date",
                "            const color = allocColors[i];",
                "            const colorDot = '<span style=\"display:inline-block;width:10px;height:10px;border-radius:3px;background:' + color + ';margin-right:8px\"></span>';",
                "            const sign = w >= 0 ? '+' : '';",
                "            const wFormatted = (sign + w.toFixed(1)) + '%';",
                f"            const textColor = w < 0 ? '{short_color}' : '{val_c}';",
                f"            html += '<div style=\"display:flex;justify-content:space-between;align-items:center;margin:1px 0\">';",
                f"            html += '<span style=\"color:{lbl_c};display:flex;align-items:center\">' + colorDot + asset + '</span>';",
                f"            html += '<span style=\"color:' + textColor + ';font-weight:700;margin-left:8px;font-size:11px\">' + wFormatted + '</span>';",
                f"            html += '</div>';",
                "        });",
                "        allocItems.innerHTML = html;",
                "        ",
                "        const netSign = netExp >= 0 ? '+' : '';",
                f"        allocTotals.innerHTML = '<div style=\"display:flex;justify-content:space-between;font-weight:600\"><span>Net</span><span style=\"margin-left:8px\">' + netSign + netExp.toFixed(1) + '%</span></div>' +",
                f"                                '<div style=\"display:flex;justify-content:space-between\"><span>Gross</span><span style=\"margin-left:8px\">' + grossExp.toFixed(1) + '% (' + (grossExp/100).toFixed(2) + 'x)</span></div>';",
                "        ",
                "        allocTooltip.style.display = 'block';",
                "    });",
            ])

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
        is_dark_bg = self._theme_name in ("dark", "midnight", "glass")
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

        season_html, season_js, season_extra = self._build_seasonality(is_dark_bg)
        cone_html, cone_js, cone_extra = self._build_cone(is_dark_bg)

        total_extra = slider_extra_height + smoothing_extra_height + season_extra + cone_extra
        self._slider_extra_height = total_extra

        _resize_js = """
    const _fcEl = document.getElementById('fc');
    function _resizeChartToContainer() {
        if (!_fcEl || !_fcEl.getBoundingClientRect) return;
        const r = _fcEl.getBoundingClientRect();
        const w = Math.max(1, Math.floor(r.width));
        const h = Math.max(1, Math.floor(r.height));
        try {
            if (typeof chart.resize === 'function') chart.resize(w, h);
            else chart.applyOptions({ width: w, height: h });
        } catch (e) {}
    }
    function _reflowChart() {
        _resizeChartToContainer();
        try { chart.timeScale().fitContent(); } catch (e) {}
    }
    _reflowChart();
    requestAnimationFrame(_reflowChart);
    setTimeout(_reflowChart, 80);
    if (typeof ResizeObserver !== 'undefined' && _fcEl) {
        const _ro = new ResizeObserver(() => { _reflowChart(); });
        _ro.observe(_fcEl);
    }
    window.addEventListener('resize', _reflowChart);
"""

        _yscale_js = ""
        if self._y_format == "kmb":
            _yscale_js = """
    function _mkScaleFmt(mode) {
        return function(p) {
            var a = Math.abs(p), d = 1, s = '', z = 2;
            if (mode === 'K') { d = 1e3; s = 'K'; z = a/1e3 < 10 ? 1 : 0; }
            else if (mode === 'M') { d = 1e6; s = 'M'; z = a/1e6 < 10 ? 1 : 0; }
            else if (mode === 'B') { d = 1e9; s = 'B'; z = a/1e9 < 10 ? 1 : 0; }
            else if (mode === 'AUTO') {
                if (a >= 1e9) { d = 1e9; s = 'B'; z = a/1e9 < 10 ? 1 : 0; }
                else if (a >= 1e6) { d = 1e6; s = 'M'; z = a/1e6 < 10 ? 1 : 0; }
                else if (a >= 1e3) { d = 1e3; s = 'K'; z = a/1e3 < 10 ? 1 : 0; }
                else { z = 2; }
            }
            return (p / d).toFixed(z) + s;
        };
    }
    const _ysSel = document.getElementById('yscale-select');
    if (_ysSel) {
        const _applyYScale = () => {
            const mode = _ysSel.value || 'AUTO';
            chart.applyOptions({ localization: { priceFormatter: _mkScaleFmt(mode) } });
        };
        _applyYScale();
        _ysSel.addEventListener('change', _applyYScale);
    }
"""

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
{bg_svg}{_glass_open}{season_html}<div id="fc"></div>
{_kmb_overlay}
{stats_html}
{alloc_tooltip_html}
<div id="err"></div>
{slider_html}
{smoothing_html}
{cone_html}
{_glass_close}{'<img id="signum-logo" src="data:image/svg+xml;base64,' + _LOGO_B64 + '" width="30" height="30" alt="Signum">' if self._logo else ''}
<script>
try {{
    const chart = LightweightCharts.createChart(document.getElementById('fc'), {self._inject_formatter_into_opts(chart_opts, self._get_formatter_js())});
    {series_js}
    {_resize_js}
    {_yscale_js}
    {season_js}
    {cone_js}
    {slider_js}
    {smoothing_js}
    {alloc_tooltip_js}
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
