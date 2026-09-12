"""StatChart – Statistical visualisations (distributions, scatter, heatmap, 3-D).

Renders publication-quality statistical charts from arrays / Series / DataFrames
using HTML5 Canvas — same display pipeline as Chart (Jupyter, Dash, Streamlit, HTML).

    from signum import StatChart

    # Single distribution
    StatChart(theme="dark").distribution(returns, name="Daily Returns").show()

    # Side-by-side grid (like Plotly subplots)
    (StatChart(theme="dark", cols=3, title="Volatility Features")
        .distribution(z_scores, name="IVol Z-Score", color="orange")
        .distribution(ivol,     name="IVol Level",   color="#e91e63")
        .distribution(spread,   name="Vol Spread",   color="#9c27b0")
    ).show()
"""

import json
import re
import html as _html
from typing import Optional, List

import numpy as np
import pandas as pd

from .themes import THEMES, resolve_theme, DARK_THEMES
from .chart import _LOGO_B64


class StatChart:
    """Statistical chart renderer — distributions, scatter plots, and more."""

    def __init__(
        self,
        theme: str = "dark",
        width: Optional[int] = None,
        height: int = 350,
        cols: Optional[int] = None,
        title: Optional[str] = None,
        logo: bool = True,
    ):
        self._theme_name = theme.lower()
        self._theme = resolve_theme(theme)
        self._width = width
        self._height = height
        self._cols = cols
        self._title = title
        self._logo = logo
        self._panels: List[dict] = []
        self._color_idx = 0
        self._slider: Optional[dict] = None  # shared date scrubber for curve/spread frames

    # ── Color cycling ─────────────────────────────────────────────────────

    def _next_color(self) -> str:
        colors = self._theme.get("line_colors", ["#2196F3"])
        c = colors[self._color_idx % len(colors)]
        self._color_idx += 1
        return c

    # ── Panel methods ─────────────────────────────────────────────────────

    def distribution(
        self,
        data,
        bins: int = 50,
        name: Optional[str] = None,
        color: Optional[str] = None,
        show_mean: bool = True,
        show_median: bool = True,
        fit: bool = False,
        fit_color: Optional[str] = None,
        percentiles: Optional[List[float]] = None,
    ) -> "StatChart":
        """Add a distribution histogram panel.

        Parameters
        ----------
        data : array-like, Series, or DataFrame
            Values to histogram.  DataFrames use the first numeric column.
        bins : int
            Number of bins (default 50).
        name : str, optional
            Panel title shown above the histogram.
        color : str, optional
            Bar fill colour (CSS).  Auto-cycles from theme palette if omitted.
        show_mean : bool
            Draw a dashed vertical line at the mean.
        show_median : bool
            Draw a dotted vertical line at the median.
        fit : bool
            Overlay a Gaussian KDE fit curve (smooth PDF estimate).
        fit_color : str, optional
            Colour for the fit curve.  Defaults to white / dark depending on theme.
        percentiles : list of float, optional
            Draw vertical percentile lines, e.g. ``[75, 90, 95, 99]``.
        """
        if isinstance(data, pd.Series):
            arr = data.dropna().values.astype(float)
        elif isinstance(data, pd.DataFrame):
            num_cols = data.select_dtypes(include="number").columns
            arr = data[num_cols[0]].dropna().values.astype(float)
        else:
            arr = np.asarray(data, dtype=float)
            arr = arr[~np.isnan(arr)]

        counts, edges = np.histogram(arr, bins=bins)

        # KDE fit curve (Gaussian kernel, Silverman bandwidth)
        kde_points: Optional[List[List[float]]] = None
        if fit:
            kde_points = self._compute_kde(arr, edges)

        # Percentile values
        pct_vals: Optional[List[dict]] = None
        if percentiles:
            pct_vals = [
                {"p": p, "v": round(float(np.percentile(arr, p)), 6)}
                for p in sorted(percentiles)
            ]

        self._panels.append({
            "type": "distribution",
            "name": name or f"Distribution {len(self._panels) + 1}",
            "color": color or self._next_color(),
            "color_explicit": color is not None,
            "counts": counts.tolist(),
            "edges": [round(float(e), 6) for e in edges],
            "mean": round(float(np.mean(arr)), 6),
            "median": round(float(np.median(arr)), 6),
            "show_mean": show_mean,
            "show_median": show_median,
            "kde": kde_points,
            "fit_color": fit_color,
            "percentiles": pct_vals,
        })
        return self

    @staticmethod
    def _compute_kde(arr: np.ndarray, edges: np.ndarray) -> List[List[float]]:
        """Gaussian KDE scaled to histogram count space (no scipy needed)."""
        n = len(arr)
        std = float(np.std(arr, ddof=1)) or 1.0
        bw = 1.06 * std * n ** (-0.2)  # Silverman's rule
        # Evaluate on a dense grid spanning the histogram range
        x_grid = np.linspace(float(edges[0]), float(edges[-1]), 200)
        bin_width = float(edges[1] - edges[0])
        # Gaussian kernel: K(u) = exp(-u^2 / 2) / sqrt(2*pi)
        density = np.zeros_like(x_grid)
        for xi in arr:
            density += np.exp(-0.5 * ((x_grid - xi) / bw) ** 2)
        density *= (1.0 / (n * bw * np.sqrt(2 * np.pi)))
        # Scale density to histogram counts
        kde_y = density * n * bin_width
        return [
            [round(float(x_grid[i]), 6), round(float(kde_y[i]), 4)]
            for i in range(len(x_grid))
        ]

    # RdYlGn etc. — ColorBrewer stops, low → high (red → green for RdYlGn).
    _COLORSCALES = {
        "RdYlGn":   ["#a50026", "#d73027", "#f46d43", "#fdae61", "#fee08b", "#ffffbf",
                     "#d9ef8b", "#a6d96a", "#66bd63", "#1a9850", "#006837"],
        "RdYlBu":   ["#a50026", "#d73027", "#f46d43", "#fdae61", "#fee090", "#ffffbf",
                     "#e0f3f8", "#abd9e9", "#74add1", "#4575b4", "#313695"],
        "Spectral": ["#9e0142", "#d53e4f", "#f46d43", "#fdae61", "#fee08b", "#ffffbf",
                     "#e6f598", "#abdda4", "#66c2a5", "#3288bd", "#5e4fa2"],
        "viridis":  ["#440154", "#482878", "#3e4a89", "#31688e", "#26828e", "#1f9e89",
                     "#35b779", "#6ece58", "#b5de2b", "#fde725"],
    }

    @staticmethod
    def _lerp_hex(stops: List[str], t: float) -> str:
        """Linearly interpolate a hex colour at ``t`` ∈ [0, 1] across ``stops``."""
        t = 0.0 if t < 0 else 1.0 if t > 1 else t
        pos = t * (len(stops) - 1)
        i = int(pos)
        if i >= len(stops) - 1:
            return stops[-1]
        frac = pos - i
        a, b = stops[i], stops[i + 1]
        ar, ag, ab = int(a[1:3], 16), int(a[3:5], 16), int(a[5:7], 16)
        br, bg, bb = int(b[1:3], 16), int(b[3:5], 16), int(b[5:7], 16)
        return "#%02x%02x%02x" % (round(ar + (br - ar) * frac),
                                  round(ag + (bg - ag) * frac),
                                  round(ab + (bb - ab) * frac))

    @staticmethod
    def _fmt_hover(v) -> str:
        """Compact a hover value — numbers abbreviated, strings verbatim, NaN → em-dash."""
        if v is None:
            return "—"
        try:
            f = float(v)
        except (TypeError, ValueError):
            return str(v)
        if f != f:
            return "—"
        a = abs(f)
        if a >= 1e6: return f"{f/1e6:.1f}M"
        if a >= 1e3: return f"{f/1e3:.1f}k"
        if a >= 100: return f"{f:.0f}"
        if a >= 10:  return f"{f:.1f}"
        if a >= 1:   return f"{f:.2f}"
        return f"{f:.3g}"

    def scatter(
        self,
        x,
        y,
        name: Optional[str] = None,
        color=None,
        size=6,
        symbol="circle",
        outline=None,
        labels=None,
        hover: Optional[dict] = None,
        colorscale="RdYlGn",
        clim: Optional[tuple] = None,
        colorbar=None,
        legend: Optional[List[dict]] = None,
        curves: Optional[List[dict]] = None,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
    ) -> "StatChart":
        """Add an x-y **scatter / bubble** panel — value axes on both sides.

        This is Signum's native way to draw a true x-y relationship (a
        duration×yield map, risk×return, a factor scatter): ``Chart`` is
        time-axis only, so anything with a numeric x uses ``StatChart`` here.

        Beyond a plain dot cloud it supports a **continuous colour scale** (a
        numeric ``color`` → colour bar), **per-point size** (a bubble chart),
        **per-point marker shape**, an **outline** (ring a subset, e.g. your
        holdings), and a **rich multi-field tooltip** — everything a Plotly
        scatter gave you, rendered on Signum's own canvas.

        Parameters
        ----------
        x, y : array-like
            Coordinate arrays (same length).
        name : str, optional
            Panel title.
        color : str | numeric array | list of CSS str, optional
            One colour for every dot, OR a **numeric array** mapped through
            ``colorscale`` (draws a colour bar), OR an explicit per-point CSS
            list.  Omitted ⇒ next theme colour.
        size : number | array-like
            Dot radius in px — a scalar, or per-point for a bubble chart.
        symbol : str | array-like
            Marker shape per point — ``'circle'`` (default), ``'diamond'``,
            ``'square'``, ``'triangle'``, ``'triangle-down'``, ``'cross'``,
            ``'x'``.  Scalar or per-point.
        outline : str | array-like, optional
            Stroke colour around each filled marker (scalar or per-point; a
            ``None`` entry = no ring).  Handy to highlight a subset.
        labels : array-like of str, optional
            Per-point name shown **bold** atop the hover tooltip.
        hover : dict {field: array-like}, optional
            Extra tooltip rows, e.g. ``{"G-spread bp": gsp, "rating": rtg}`` —
            numbers are compacted, strings shown verbatim.
        colorscale : str | list of hex
            Palette for a numeric ``color`` — ``'RdYlGn'`` (default),
            ``'RdYlBu'``, ``'Spectral'``, ``'viridis'``, or an explicit hex list.
        clim : (lo, hi), optional
            Colour-scale limits (default = data min/max).
        colorbar : str | False, optional
            Colour-bar title.  ``False`` suppresses the bar even for numeric
            colour.
        legend : list of dict, optional
            Marker-shape key — ``[{"symbol": "diamond", "label": "holdings"},
            …]`` (optional ``"color"`` per entry).
        curves : list of dict, optional
            Fitted lines drawn **through** the cloud (term-structure / yield
            curves) — each ``{"x": [...], "y": [...], "name": ..., "color": ...,
            "lower": [...], "upper": [...], "dash": bool}``.  ``lower``/``upper``
            shade a confidence band; ``name`` adds a legend entry.  Curve points
            are independent of the scatter points and extend the axis range.
        x_label, y_label : str, optional
            Axis titles.  Also the default x/y hover field names.
        """
        xarr = np.asarray(x, dtype=float)
        yarr = np.asarray(y, dtype=float)
        mask = ~(np.isnan(xarr) | np.isnan(yarr))
        xarr, yarr = xarr[mask], yarr[mask]
        idx = np.where(mask)[0]

        def _sub(a):
            """Align an array-like to the valid (finite x,y) mask; None-safe."""
            if a is None:
                return None
            seq = list(a)
            return [seq[i] for i in idx]

        def _is_scalar(a):
            return isinstance(a, (str, int, float, np.number)) or a is None

        # ── colour: single / numeric-colourmap / explicit per-point ───────
        pcolors = None
        single_color = None
        cb = None
        if color is None:
            single_color = self._next_color()
        elif isinstance(color, str):
            single_color = color
        else:
            vals = _sub(color)
            try:
                nums = [float(v) for v in vals]
                numeric = True
            except (TypeError, ValueError):
                numeric = False
            if numeric:
                stops = (list(colorscale) if isinstance(colorscale, (list, tuple))
                         else self._COLORSCALES.get(colorscale, self._COLORSCALES["RdYlGn"]))
                finite = [v for v in nums if v == v]
                lo, hi = (clim if clim else ((min(finite), max(finite)) if finite else (0.0, 1.0)))
                span = (hi - lo) or 1.0
                pcolors = [self._lerp_hex(stops, (v - lo) / span) if v == v else "#888888" for v in nums]
                if colorbar is not False:
                    cb = {"title": colorbar if isinstance(colorbar, str) else None,
                          "stops": [[round(i / (len(stops) - 1), 4), stops[i]] for i in range(len(stops))],
                          "lo": round(float(lo), 6), "hi": round(float(hi), 6)}
            else:
                pcolors = [str(v) for v in vals]

        # ── size / symbol / outline: scalar or per-point ──────────────────
        sizes = None if _is_scalar(size) else [self._safe_float(v) for v in _sub(size)]
        size_s = float(size) if _is_scalar(size) and size is not None else 6.0
        symbols = None if _is_scalar(symbol) else [str(v) for v in _sub(symbol)]
        symbol_s = symbol if isinstance(symbol, str) else "circle"
        outlines = None if _is_scalar(outline) else [
            (None if (v is None or (isinstance(v, float) and v != v)) else str(v)) for v in _sub(outline)]
        outline_s = outline if isinstance(outline, str) else None

        hov = None
        if hover:
            hov = [{"t": str(k), "v": [self._fmt_hover(w) for w in _sub(v)]} for k, v in hover.items()]

        def _numlist(a):
            return None if a is None else [None if (w is None or (isinstance(w, float) and w != w)) else round(float(w), 6) for w in a]

        def _norm_curve(cv):
            col = str(cv.get("color") or self._next_color())
            return {"x": [round(float(v), 6) for v in cv["x"]],
                    "y": _numlist(cv["y"]),
                    "color": col,
                    "name": (str(cv["name"]) if cv.get("name") else None),
                    "dash": bool(cv.get("dash")),
                    "lower": _numlist(cv.get("lower")),
                    "upper": _numlist(cv.get("upper")),
                    "band": cv.get("band") or self._rgba(col, 0.14)}
        curves_n = [_norm_curve(c) for c in curves] if curves else None

        self._panels.append({
            "type": "scatter",
            "name": (f"Scatter {len(self._panels) + 1}" if name is None else name),
            "color": single_color or "#2196F3",
            "pcolors": pcolors,
            "x": [round(float(v), 6) for v in xarr],
            "y": [round(float(v), 6) for v in yarr],
            "size": size_s,
            "sizes": [s if s is not None else size_s for s in sizes] if sizes is not None else None,
            "symbol": symbol_s,
            "symbols": symbols,
            "outline": outline_s,
            "outlines": outlines,
            "plabels": [str(v) for v in _sub(labels)] if labels is not None else None,
            "hover": hov,
            "colorbar": cb,
            "legend": [{"symbol": str(it.get("symbol", "circle")),
                        "label": str(it.get("label", "")),
                        "color": (str(it["color"]) if it.get("color") else None)}
                       for it in legend] if legend else None,
            "curves": curves_n,
            "x_label": x_label,
            "y_label": y_label,
        })
        return self

    # ── Categorical / binned bars ─────────────────────────────────────────

    def bars(
        self,
        x=None,
        by=None,
        values=None,
        bins=10,
        agg: str = "count",
        mode: str = "group",
        bin_labels: Optional[List[str]] = None,
        name: Optional[str] = None,
        color=None,
        show_values: bool = True,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
    ) -> "StatChart":
        """Add a categorical / binned bar panel — an **ordinal** x-axis.

        Unlike :meth:`distribution` (a continuous numeric axis), this lays out
        equal-width category slots and draws one bar per slot, optionally split
        into a clustered (``mode="group"``) or ``"stack"``-ed series by a second
        descriptive dimension.

        Three roles are chosen *explicitly* so a mixed numeric+descriptive frame
        is never ambiguous about what groups and what becomes the bar height:

        * **bin dimension** — ``x`` (numeric) is bucketed into labelled ranges
          via ``pd.cut``; those ranges become the x-axis slots.
        * **split dimension** — ``by`` (categorical, aligned to ``x``) becomes
          the series, drawn clustered or stacked within each slot.
        * **height** — a ``count`` of rows per bucket, or ``values`` aggregated
          per bucket by ``agg`` ("sum" / "mean" / "median").

        Precomputed input is also accepted (no re-binning):

        * a ``dict`` ``{label: value}`` (single series) or
          ``{label: {series: value}}`` (multi-series),
        * a ``DataFrame`` (index → slot labels, each column → a series),
        * a string/category-indexed ``Series`` ``{label: value}``.

        Parameters
        ----------
        x : array-like / Series / dict / DataFrame
            Raw numeric values to bucket, OR a precomputed mapping (see above).
        by : array-like, optional
            Categorical split aligned to ``x`` (ignored for precomputed input).
        values : array-like, optional
            Numeric quantity to aggregate per bucket.  ``None`` ⇒ row counts.
        bins : int or sequence, optional
            Number of equal-width buckets, or explicit edges, passed to ``pd.cut``.
        agg : {"count", "sum", "mean", "median"}
            Aggregation of ``values`` within a bucket (forced to "count" when
            ``values`` is None).
        mode : {"group", "stack"}
            Clustered side-by-side bars, or stacked into one bar per slot.
        bin_labels : list of str, optional
            Override the auto-generated bucket labels.
        name : str, optional
            Panel title.
        color : str or list, optional
            Bar colour (single series) or a list of colours (one per series).
            Auto-cycles the theme palette if omitted.
        show_values : bool
            Draw the value above each bar / stack (default True).
        """
        labels, series = self._bin_to_series(x, by, values, bins, agg, bin_labels)

        cols = list(color) if isinstance(color, (list, tuple)) else ([color] if color else [])
        single = len(series) == 1 and series[0]["name"] is None
        out_series = []
        for i, s in enumerate(series):
            c = cols[i] if i < len(cols) else None
            out_series.append({
                "name": s["name"],
                "color": c or self._next_color(),
                "values": [self._safe_round(v) for v in s["values"]],
            })

        self._panels.append({
            "type": "bars",
            "name": name or f"Bars {len(self._panels) + 1}",
            "labels": [str(lbl) for lbl in labels],
            "series": out_series,
            "mode": "stack" if str(mode).lower() == "stack" else "group",
            "single": single,
            "show_values": bool(show_values),
            "x_label": x_label,
            "y_label": y_label,
        })
        return self

    # ── Bars: input resolution (raw-binned or precomputed) ────────────────

    def _bin_to_series(self, x, by, values, bins, agg, bin_labels):
        """Resolve raw-or-precomputed input into ``(labels, [{name, values}, ...])``."""
        # ── precomputed: DataFrame (index → slots, columns → series) ──────
        if isinstance(x, pd.DataFrame):
            labels = [str(i) for i in x.index]
            series = [
                {"name": str(c), "values": [self._safe_float(v) for v in x[c].values]}
                for c in x.columns
            ]
            return labels, series

        # ── precomputed: dict ────────────────────────────────────────────
        if isinstance(x, dict):
            keys = list(x.keys())
            if keys and isinstance(x[keys[0]], dict):                 # {label: {series: value}}
                snames: List[str] = []
                for d in x.values():
                    for k in (d or {}):
                        if k not in snames:
                            snames.append(k)
                series = [
                    {"name": str(sn),
                     "values": [self._safe_float((x[k] or {}).get(sn)) for k in keys]}
                    for sn in snames
                ]
                return [str(k) for k in keys], series
            return [str(k) for k in keys], [                          # {label: value}
                {"name": None, "values": [self._safe_float(v) for v in x.values()]}
            ]

        # ── precomputed: textual-index Series {label: value} ─────────────
        if isinstance(x, pd.Series) and self._is_label_index(x.index):
            return [str(i) for i in x.index], [
                {"name": None, "values": [self._safe_float(v) for v in x.values]}
            ]

        # ── raw numeric → bin via pd.cut ─────────────────────────────────
        xs = pd.Series(np.asarray(x, dtype=float))
        mask = xs.notna()
        cats = pd.cut(xs[mask], bins=bins, include_lowest=True)
        cat_index = cats.cat.categories
        codes = cats.cat.codes.values                                 # bucket index per row (-1 = NaN)
        nb = len(cat_index)
        if bin_labels is not None:
            labels = [str(lbl) for lbl in list(bin_labels)[:nb]]
        else:
            labels = [self._interval_label(iv) for iv in cat_index]

        use_count = values is None or str(agg).lower() == "count"
        vals = None if values is None else np.asarray(values, dtype=float)[mask.values]
        bys = None if by is None else np.asarray(by, dtype=object)[mask.values]

        def _height(sel):
            if use_count:
                return float(np.count_nonzero(sel))
            picked = vals[sel]
            picked = picked[~np.isnan(picked)]
            if picked.size == 0:
                return None
            a = str(agg).lower()
            if a == "sum":
                return float(np.sum(picked))
            if a == "median":
                return float(np.median(picked))
            return float(np.mean(picked))                             # default "mean"

        if bys is None:
            return labels, [{"name": None,
                             "values": [_height(codes == b) for b in range(nb)]}]

        groups = [g for g in pd.unique(bys) if g is not None and g == g]
        try:
            groups = sorted(groups)
        except TypeError:
            pass
        series = [
            {"name": str(g),
             "values": [_height((bys == g) & (codes == b)) for b in range(nb)]}
            for g in groups
        ]
        return labels, series

    @staticmethod
    def _safe_float(v):
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        return None if f != f else f

    @staticmethod
    def _safe_round(v):
        return None if v is None or v != v else round(float(v), 6)

    @staticmethod
    def _is_label_index(idx) -> bool:
        """True when a Series index is textual — treat the Series as precomputed labels."""
        return (idx.dtype == object
                or isinstance(idx, pd.CategoricalIndex)
                or str(idx.dtype).startswith("string"))

    @staticmethod
    def _interval_label(iv) -> str:
        """Format a pandas Interval as a compact ``lo–hi`` slot label."""
        def f(v):
            v = float(v)
            if abs(v) >= 1000:
                return f"{v:,.0f}"
            return str(int(v)) if v == int(v) else f"{v:g}"
        return f"{f(iv.left)}–{f(iv.right)}"

    # ── Curve helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _rgba(color: Optional[str], alpha: float) -> Optional[str]:
        """Convert a hex / rgb(a) colour to an rgba() string at the given alpha."""
        if color is None:
            return None
        if color.startswith("#"):
            h = color.lstrip("#")
            if len(h) == 3:
                h = "".join(c * 2 for c in h)
            r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
            return f"rgba({r},{g},{b},{alpha})"
        if color.startswith("rgb"):
            nums = re.findall(r"[\d.]+", color)
            if len(nums) >= 3:
                return f"rgba({nums[0]},{nums[1]},{nums[2]},{alpha})"
        return color

    @staticmethod
    def _num_list(a) -> List[Optional[float]]:
        """Coerce an array-like to a JSON-safe list (NaN → None for whitespace gaps)."""
        out: List[Optional[float]] = []
        for v in np.asarray(a, dtype=float).ravel():
            out.append(None if v != v else round(float(v), 6))
        return out

    def _frames_from(self, frames: dict, keys: tuple) -> tuple:
        """Build per-frame array dicts from a {date: {...}} mapping.

        Returns (frame_dates, frame_array).  Each frame holds the requested
        keys (mean/lower/upper/prior) plus px/py from an optional 'points'
        (x, y) tuple.  All frames share the panel's x grid.
        """
        frame_dates = list(frames.keys())
        frame_array = []
        for d in frame_dates:
            f = frames[d] or {}
            rec = {}
            for k in keys:
                rec[k] = self._num_list(f[k]) if f.get(k) is not None else None
            pts = f.get("points")
            rec["px"] = self._num_list(pts[0]) if pts is not None else None
            rec["py"] = self._num_list(pts[1]) if pts is not None else None
            rec["plabels"] = [str(x) for x in f["labels"]] if f.get("labels") else None
            rec["pcolors"] = [str(x) for x in f["colors"]] if f.get("colors") else None
            rec["psizes"] = self._num_list(f["sizes"]) if f.get("sizes") is not None else None
            frame_array.append(rec)
        return frame_dates, frame_array

    @staticmethod
    def _trade_marker(tr) -> Optional[dict]:
        """Normalize a per-date trade dict to JSON-safe ``{side: [[x, y, label], ...]}``.

        Each side ('long' / 'short') accepts EITHER a single marker ``(x, y, label)``
        OR a list of them ``[(x, y, label), ...]`` — both normalize to a list, so any
        number of long/short positions render on one frame (not just a single pair).
        """
        def _one(m):
            if m is None or m[0] is None or m[1] is None:
                return None
            return [round(float(m[0]), 6), round(float(m[1]), 6),
                    str(m[2]) if len(m) > 2 and m[2] is not None else ""]

        out = {}
        for side in ("long", "short"):
            v = tr.get(side)
            if not v:
                continue
            markers = v if isinstance(v[0], (list, tuple)) else [v]   # single (x,y,l) -> [(x,y,l)]
            ms = [m for m in (_one(x) for x in markers) if m is not None]
            if ms:
                out[side] = ms
        return out or None

    def _register_slider(self, frame_dates, base_idx, label, color):
        """Register (or reconcile) the shared date scrubber.

        All framed panels must share the same ordered date list — the slider
        index maps 1:1 onto it.  The first non-None base date wins.
        """
        if self._slider is None:
            self._slider = {
                "dates": [str(d) for d in frame_dates],
                "base": base_idx,
                "label": label,
                "color": color,
            }
        else:
            if [str(d) for d in frame_dates] != self._slider["dates"]:
                raise ValueError(
                    "curve()/spread() frames must share the same ordered date list "
                    "so a single slider can drive every panel."
                )
            if self._slider["base"] is None and base_idx is not None:
                self._slider["base"] = base_idx

    # ── Panel methods (curve / spread) ────────────────────────────────────

    def curve(
        self,
        x,
        mean=None,
        lower=None,
        upper=None,
        points=None,
        point_labels: Optional[List[str]] = None,
        point_colors: Optional[List[str]] = None,
        point_sizes: Optional[List[float]] = None,
        point_legend: Optional[dict] = None,
        extras: Optional[List[dict]] = None,
        prior=None,
        name: Optional[str] = None,
        color: Optional[str] = None,
        band_color: Optional[str] = None,
        point_color: Optional[str] = None,
        prior_color: Optional[str] = None,
        prior_name: str = "prior",
        band_label: str = "95% band",
        mean_name: str = "fit",
        point_name: str = "bonds",
        base_name: str = "base",
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        resid_label: str = "resid",
        frames: Optional[dict] = None,
        base: Optional[str] = None,
        slider_label: str = "date",
        trades: Optional[dict] = None,
    ) -> "StatChart":
        """Add a fitted-curve panel: scatter points + smooth line + confidence band.

        Built for term-structure / yield-curve style plots — observed dots,
        a smoothed posterior line, a shaded band, and (optionally) a second
        reference line (e.g. an NSS prior).

        Two modes:

        **Static** — pass ``x`` (the grid) with ``mean`` / ``lower`` / ``upper``
        / ``prior`` arrays aligned to it, and ``points=(px, py)`` for the dots::

            StatChart().curve(grid, mean=gp_mean, lower=lo, upper=hi,
                              points=(ttm, ytm), prior=nss).show()

        **Dated frames** — pass ``frames={date: {...}}`` to get a date slider
        that morphs the curve.  Each frame dict may hold ``mean``/``lower``/
        ``upper``/``prior`` (aligned to ``x``) and ``points=(px, py)``.  Pin a
        ``base`` date to draw it as a dashed ghost reference and to anchor a
        linked :meth:`spread` panel::

            StatChart().curve(grid, frames=by_date, base="2026-06-01").spread(
                grid, frames=by_date, base="2026-06-01")

        Parameters
        ----------
        x : array-like
            The x grid (e.g. time-to-maturity) shared by mean/band/prior.
        mean, lower, upper, prior : array-like, optional
            Curve, band bounds, and second reference line — aligned to ``x``.
            Ignored when ``frames`` is given.
        points : (array-like, array-like), optional
            Observed ``(x, y)`` dots (e.g. per-bond ttm + yield).
        color, band_color, point_color, prior_color : str, optional
            Colours.  ``band_color`` defaults to the curve colour at low alpha.
        frames : dict {date: dict}, optional
            Per-date arrays for the slider.  See above.
        base : str, optional
            A key in ``frames`` to pin as a dashed ghost reference.
        """
        gx = self._num_list(x)
        # Defaults pulled from the active theme so curves look native (ft / midnight /
        # …): mean = palette[0], prior = palette[1] (a distinct hue, not the gray
        # base-ghost), points = the theme's ink colour.
        curve_color = color or self._next_color()
        prior_col = prior_color or self._next_color()
        ink = self._theme.get("chart", {}).get("layout", {}).get("textColor")
        point_col = point_color or ink or curve_color
        frame_array = None
        base_idx = None
        if frames is not None:
            frame_dates, frame_array = self._frames_from(
                frames, ("mean", "lower", "upper", "prior")
            )
            if base is not None and base in frame_dates:
                base_idx = frame_dates.index(base)
            self._register_slider(frame_dates, base_idx, slider_label, curve_color)
            if trades:
                for i, d in enumerate(frame_dates):
                    tr = trades.get(d) or trades.get(str(d))
                    frame_array[i]["trade"] = self._trade_marker(tr) if tr else None

        self._panels.append({
            "type": "curve",
            "name": name or "Curve",
            "x": gx,
            "color": curve_color,
            "band_color": band_color or self._rgba(curve_color, 0.18),
            "point_color": point_col,
            "prior_color": prior_col,
            "prior_name": prior_name,
            "mean_name": mean_name,
            "point_name": point_name,
            "base_name": base_name,
            "band_label": band_label,
            "x_label": x_label,
            "y_label": y_label,
            "resid_label": resid_label,
            # static arrays
            "mean": self._num_list(mean) if mean is not None else None,
            "lower": self._num_list(lower) if lower is not None else None,
            "upper": self._num_list(upper) if upper is not None else None,
            "px": self._num_list(points[0]) if points is not None else None,
            "py": self._num_list(points[1]) if points is not None else None,
            "plabels": [str(x) for x in point_labels] if point_labels is not None else None,
            "pcolors": [str(x) for x in point_colors] if point_colors is not None else None,
            "psizes": self._num_list(point_sizes) if point_sizes is not None else None,
            "extras": [{"y": self._num_list(e["y"]), "name": str(e.get("name", "")), "color": str(e.get("color", "#888888"))}
                       for e in extras] if extras else None,
            "plegend": [{"t": str(k), "c": str(v)} for k, v in point_legend.items()] if point_legend else None,
            "prior": self._num_list(prior) if prior is not None else None,
            # framed
            "frames": frame_array,
            "base_idx": base_idx,
        })
        return self

    def spread(
        self,
        x,
        frames: Optional[dict] = None,
        base: Optional[str] = None,
        static_y=None,
        band: bool = False,
        name: str = "spread",
        color: Optional[str] = None,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        slider_label: str = "date",
    ) -> "StatChart":
        """Add a difference panel — ``active − base`` per x, with a zero line.

        Designed to sit below a :meth:`curve` panel sharing the same date
        slider, so dragging the slider redraws the spread live against the
        pinned base date.

        Static mode: pass ``static_y`` (the precomputed difference aligned to
        ``x``) for a one-off A−B comparison with no slider.
        """
        gx = self._num_list(x)
        accent = color or self._next_color()
        frame_array = None
        base_idx = None
        if frames is not None:
            frame_dates, frame_array = self._frames_from(
                frames, ("mean", "lower", "upper")
            )
            if base is not None and base in frame_dates:
                base_idx = frame_dates.index(base)
            self._register_slider(frame_dates, base_idx, slider_label, accent)

        self._panels.append({
            "type": "spread",
            "name": name,
            "x": gx,
            "color": accent,
            "band": bool(band),
            "x_label": x_label,
            "y_label": y_label,
            "frames": frame_array,
            "base_idx": base_idx,
            "static": self._num_list(static_y) if static_y is not None else None,
        })
        return self

    # ── Theme helpers ─────────────────────────────────────────────────────

    def _is_dark(self) -> bool:
        bg = (self._theme.get("chart", {})
              .get("layout", {})
              .get("background", {})
              .get("color", "#1e1e1e"))
        if bg.startswith("#") and len(bg) >= 7:
            r, g, b = (int(bg[i:i+2], 16) for i in (1, 3, 5))
            return (r * 0.299 + g * 0.587 + b * 0.114) < 140
        return self._theme_name in DARK_THEMES

    def _font_family(self) -> str:
        return (self._theme.get("chart", {})
                .get("layout", {})
                .get("fontFamily",
                     "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"))

    # ── Build HTML ────────────────────────────────────────────────────────

    def _build_html(self) -> str:
        if not self._panels:
            return "<html><body>No panels</body></html>"

        n = len(self._panels)
        cols = self._cols or min(n, 3)
        dark = self._is_dark()

        bg = (self._theme.get("chart", {})
              .get("layout", {})
              .get("background", {})
              .get("color", "#1e1e1e"))

        # ── Chrome colours: pull from the SAME theme dict Chart uses, so
        #    StatChart matches the LWC charts (grid, ink, crosshair) per theme.
        #    Generic dark/light values are only fallbacks when a theme omits a key.
        _chart = self._theme.get("chart", {})
        _layout = _chart.get("layout", {})
        _ink = _layout.get("textColor")  # e.g. ft "#33302e", dark "#d1d4dc"

        def _ink_alpha(a, fallback):
            return self._rgba(_ink, a) if _ink else fallback

        text_color  = _ink_alpha(0.82, "rgba(255,255,255,0.75)" if dark else "rgba(0,0,0,0.65)")
        sub_color   = _ink_alpha(0.48, "rgba(255,255,255,0.45)" if dark else "rgba(0,0,0,0.40)")
        title_color = _ink_alpha(0.95, "rgba(255,255,255,0.88)" if dark else "rgba(0,0,0,0.78)")
        ref_color   = _ink_alpha(0.55, "rgba(255,255,255,0.55)" if dark else "rgba(0,0,0,0.55)")
        # axis derived from ink (not rightPriceScale.borderColor — some themes set
        # that to "transparent", which would make StatChart's axes vanish).
        axis_color  = _ink_alpha(0.22, "rgba(255,255,255,0.18)" if dark else "rgba(0,0,0,0.18)")
        grid_color  = (_chart.get("grid", {}).get("horzLines", {}).get("color")
                       or ("rgba(255,255,255,0.06)" if dark else "rgba(0,0,0,0.07)"))
        cross_color = (_chart.get("crosshair", {}).get("vertLine", {}).get("color")
                       or ("rgba(255,255,255,0.30)" if dark else "rgba(0,0,0,0.25)"))

        custom_bg_css = self._theme.get("background_css", "")
        bg_css = custom_bg_css if custom_bg_css else f"background:{bg};"
        bg_svg = self._theme.get("background_svg", "")
        font = self._font_family()
        logo_invert = "filter:invert(1);" if not dark else ""
        # trade-marker colours: the theme's natural up/down (buy/sell) hues; mono font so digits
        # in tickers don't shrink (serif themes like 'ft' use oldstyle figures).
        long_c = self._theme.get("candlestick", {}).get("upColor", "#16a34a")
        short_c = self._theme.get("candlestick", {}).get("downColor", "#dc2626")
        mark_font = "'SF Mono','Consolas','Menlo',monospace"

        panels_json = json.dumps(self._panels, separators=(",", ":"))

        # Theme stat overrides (bar_stroke, fit line width)
        stat_cfg = self._theme.get("stat", {})
        bar_stroke = stat_cfg.get("bar_stroke", "")
        bar_stroke_w = stat_cfg.get("bar_stroke_width", 0)
        bar_fill_override = stat_cfg.get("bar_fill", "")
        fit_lw = stat_cfg.get("fit_line_width", 2)
        font_style = stat_cfg.get("font_style", "normal")
        theme_fit_color = stat_cfg.get("fit_color", "")
        fit_default_color = theme_fit_color or ("rgba(255,255,255,0.9)" if dark else "rgba(0,0,0,0.75)")
        pct_color = stat_cfg.get("percentile_color", "")

        title_html = ""
        if self._title:
            safe = _html.escape(self._title)
            title_html = (
                f'<div id="mt" style="color:{title_color};font-family:{font};'
                f'font-size:15px;font-weight:600;font-style:{font_style};padding:14px 0 6px 20px;'
                f'letter-spacing:0.2px">{safe}</div>'
            )

        logo_html = ""
        if self._logo:
            logo_html = (
                f'<img id="signum-logo" '
                f'src="data:image/svg+xml;base64,{_LOGO_B64}" '
                f'width="30" height="30" alt="Signum" '
                f'style="position:fixed;right:12px;bottom:6px;'
                f'opacity:0.7;pointer-events:none;{logo_invert}">'
            )

        # ── Date scrubber — styled to match Chart's MA/threshold sliders
        #    (centered, monospace label, fixed-width range, accent thumb).
        slider_html = ""
        slider_meta = "null"
        if self._slider:
            sl = self._slider
            ndates = len(sl["dates"])
            base_i = sl["base"]
            init_i = ndates - 1  # start on the most-recent curve
            # Accent = the theme's up/"buy" colour — exactly what Chart's
            # threshold/MA sliders use (e.g. ft #00847b green), pulled from
            # THEMES, not approximated.
            accent = self._theme.get("candlestick", {}).get("upColor", "#26a69a")
            lbl_c = "rgba(255,255,255,0.88)" if dark else "rgba(0,0,0,0.78)"
            cnt_c = "rgba(255,255,255,0.48)" if dark else "rgba(0,0,0,0.42)"
            mono = "'SF Mono','Consolas',monospace"
            cur_txt = _html.escape(sl["dates"][init_i])
            base_html = ""
            if base_i is not None:
                base_html = (
                    f'<span id="sl-base" style="color:{cnt_c};font:11px/1 sans-serif;'
                    f'min-width:96px;text-align:right">base {_html.escape(sl["dates"][base_i])}</span>'
                )
            slider_html = (
                f'<div id="sl-wrap" style="display:flex;align-items:center;justify-content:center;'
                f'gap:10px;background:transparent;padding:6px 16px;white-space:nowrap;height:36px">'
                f'<span id="sl-label" style="color:{lbl_c};font:11px/1 {mono}">'
                f'{_html.escape(sl["label"])}</span>'
                f'<input id="sl-range" type="range" min="0" max="{ndates - 1}" step="1" '
                f'value="{init_i}" style="width:240px;cursor:pointer;accent-color:{accent}">'
                f'<span id="sl-date" style="color:{accent};font:11px/1 {mono};'
                f'min-width:80px">{cur_txt}</span>'
                f'{base_html}'
                f'</div>'
            )
            slider_meta = (
                "{dates:" + json.dumps(sl["dates"]) +
                ",base:" + ("null" if base_i is None else str(base_i)) +
                ",init:" + str(init_i) + "}"
            )

        # Height accounting: title ≈ 40px, slider ≈ 36px
        top_px = (40 if self._title else 0) + (36 if self._slider else 0)
        grid_h = f"calc(100vh - {top_px}px)"
        grid_w = f"{self._width}px" if self._width else "100%"

        return f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{{bg_css}font-family:{font};font-style:{font_style};overflow:hidden;position:relative;border-radius:12px}}
#grid{{display:grid;grid-template-columns:repeat({cols},1fr);
  gap:10px;padding:4px 16px 10px 16px;width:{grid_w};height:{grid_h}}}
.cell{{position:relative;width:100%;height:100%;min-height:0}}
canvas{{display:block}}
</style></head><body>
{bg_svg}{title_html}{slider_html}
<div id="grid"></div>
{logo_html}
<script>(function(){{
"use strict";
const P={panels_json};
const grid=document.getElementById("grid");
const SLIDER={slider_meta};
let AF=SLIDER?SLIDER.init:0;        /* active frame index */
const REDRAW=[];                    /* per-panel redraw fns, fired on slider move */
const XLINK={{x:null}};             /* shared x zoom for curve+spread, which share the grid */
const TC="{text_color}",SC="{sub_color}",GC="{grid_color}",
      AC="{axis_color}",RC="{ref_color}";
const FONT="{font}";
const LONG_C="{long_c}",SHORT_C="{short_c}",MARK_FONT="{mark_font}";
const BAR_STROKE="{bar_stroke}",BAR_SW={bar_stroke_w},FIT_LW={fit_lw};
const BAR_FILL="{bar_fill_override}";
const FIT_DEFAULT="{fit_default_color}";
const PCT_C="{pct_color}";

/* ── nice axis step ────────────────────────────────────── */
function niceStep(range,maxT){{
  const r=range/maxT,m=Math.pow(10,Math.floor(Math.log10(r))),n=r/m;
  return (n<=1.5?1:n<=3?2:n<=7?5:10)*m;
}}
function fmt(v){{
  if(Math.abs(v)>=1e6)return(v/1e6).toFixed(1)+"M";
  if(Math.abs(v)>=1e3)return(v/1e3).toFixed(1)+"k";
  if(Math.abs(v)>=100)return v.toFixed(0);
  if(Math.abs(v)>=10)return v.toFixed(1);
  if(Math.abs(v)>=1)return v.toFixed(2);
  return v.toPrecision(3);
}}
function fmtV(v){{   /* value-label: integers bare (counts), else fmt() */
  if(v==null)return"";
  if(Math.abs(v-Math.round(v))<1e-9)return String(Math.round(v));
  return fmt(v);
}}

/* ── marker shapes (scatter) ───────────────────────────── */
function pathMarker(ctx,cx,cy,r,sym){{
  ctx.beginPath();
  if(sym==="diamond"){{ctx.moveTo(cx,cy-r);ctx.lineTo(cx+r,cy);ctx.lineTo(cx,cy+r);ctx.lineTo(cx-r,cy);ctx.closePath();}}
  else if(sym==="square"){{ctx.rect(cx-r,cy-r,2*r,2*r);}}
  else if(sym==="triangle"){{ctx.moveTo(cx,cy-r);ctx.lineTo(cx+r*0.87,cy+r*0.55);ctx.lineTo(cx-r*0.87,cy+r*0.55);ctx.closePath();}}
  else if(sym==="triangle-down"){{ctx.moveTo(cx,cy+r);ctx.lineTo(cx+r*0.87,cy-r*0.55);ctx.lineTo(cx-r*0.87,cy-r*0.55);ctx.closePath();}}
  else{{ctx.arc(cx,cy,r,0,Math.PI*2);}}                       /* circle (default) */
}}
function lineMarker(ctx,cx,cy,r,sym){{
  ctx.beginPath();
  if(sym==="cross"){{ctx.moveTo(cx-r,cy);ctx.lineTo(cx+r,cy);ctx.moveTo(cx,cy-r);ctx.lineTo(cx,cy+r);}}
  else{{ctx.moveTo(cx-r,cy-r);ctx.lineTo(cx+r,cy+r);ctx.moveTo(cx+r,cy-r);ctx.lineTo(cx-r,cy+r);}}   /* x */
}}
function drawPoint(ctx,cx,cy,r,sym,col,ol){{
  if(sym==="x"||sym==="cross"){{
    ctx.strokeStyle=col;ctx.lineWidth=Math.max(1.4,r*0.42);ctx.lineCap="round";
    lineMarker(ctx,cx,cy,r,sym);ctx.stroke();
  }}else{{
    ctx.fillStyle=col;pathMarker(ctx,cx,cy,r,sym);ctx.fill();
    if(ol){{ctx.strokeStyle=ol;ctx.lineWidth=2;ctx.stroke();}}
  }}
}}

/* ── tooltip style ──────────────────────────────────────── */
const TT_BG="{("rgba(30,30,30,0.92)" if dark else "rgba(255,255,255,0.94)")}";
const TT_FG="{("rgba(255,255,255,0.9)" if dark else "rgba(0,0,0,0.8)")}";
const TT_BD="{("rgba(255,255,255,0.15)" if dark else "rgba(0,0,0,0.12)")}";
const CH_C="{cross_color}";

/* ── render each panel ─────────────────────────────────── */
P.forEach(function(p){{
  const cell=document.createElement("div");cell.className="cell";
  grid.appendChild(cell);
  /* main chart canvas */
  const cvs=document.createElement("canvas");cell.appendChild(cvs);
  /* overlay canvas for crosshair (avoids redrawing the main chart) */
  const ovl=document.createElement("canvas");
  ovl.style.cssText="position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none";
  cell.appendChild(ovl);
  /* tooltip div */
  const tip=document.createElement("div");
  tip.style.cssText="position:absolute;display:none;pointer-events:none;"
    +"padding:6px 10px;border-radius:4px;font:11px 'SF Mono','Consolas','Monaco','Menlo',monospace;"
    +"color:"+TT_FG+";background:"+TT_BG+";border:1px solid "+TT_BD+";"
    +"box-shadow:0 2px 8px rgba(0,0,0,0.3);z-index:10;white-space:nowrap;"
    +"line-height:1.5";
  cell.appendChild(tip);

  /* ── shared layout values (set during draw, used by hover) ── */
  let _pad,_pW,_pH,_xMin,_xMax,_yMin,_yMax,_maxC,_W,_H;
  let VIEW=null;   /* zoomed/panned x0,x1,y0,y1; null = fit to data */
  const sx=v=>_pad.left+(v-_xMin)/(_xMax-_xMin)*_pW;
  const sy=v=>_pad.top+_pH-(v-_yMin)/(_yMax-_yMin)*_pH;

  function draw(){{
    const dpr=window.devicePixelRatio||1;
    _W=cell.clientWidth;_H=cell.clientHeight;
    if(_W===0||_H===0)return;
    cvs.width=_W*dpr;cvs.height=_H*dpr;
    cvs.style.width=_W+"px";cvs.style.height=_H+"px";
    ovl.width=_W*dpr;ovl.height=_H*dpr;
    ovl.style.width=_W+"px";ovl.style.height=_H+"px";
    const ctx=cvs.getContext("2d");ctx.scale(dpr,dpr);

    /* ── distribution ───────────────────────────────────── */
    if(p.type==="distribution"){{
      const C=p.counts,E=p.edges,total=C.reduce((a,b)=>a+b,0);
      _maxC=Math.max(...C);
      if(p.kde&&p.kde.length){{
        for(let k=0;k<p.kde.length;k++)if(p.kde[k][1]>_maxC)_maxC=p.kde[k][1];
      }}
      _xMin=E[0];_xMax=E[E.length-1];
      _pad={{top:30,right:14,bottom:34,left:46}};
      _pW=_W-_pad.left-_pad.right;_pH=_H-_pad.top-_pad.bottom;
      _yMin=0;_yMax=_maxC;
      const sxD=v=>_pad.left+(v-_xMin)/(_xMax-_xMin)*_pW;
      const syD=v=>_pad.top+_pH-(v/_maxC)*_pH;

      /* y grid + labels */
      const yS=niceStep(_maxC,5);
      ctx.strokeStyle=GC;ctx.lineWidth=1;
      ctx.fillStyle=SC;ctx.font="10px "+FONT;
      ctx.textAlign="right";ctx.textBaseline="middle";
      for(let yv=0;yv<=_maxC;yv+=yS){{
        const y=syD(yv);
        ctx.beginPath();ctx.moveTo(_pad.left,y);ctx.lineTo(_W-_pad.right,y);ctx.stroke();
        ctx.fillText(fmt(yv),_pad.left-6,y);
      }}

      /* bars — priority: explicit user color > theme bar_fill > palette */
      const bW=_pW/C.length,gap=Math.max(0.5,bW*0.06);
      p._barW=bW;p._gap=gap;p._total=total;
      const barColor=p.color_explicit?p.color:(BAR_FILL||p.color);
      ctx.fillStyle=barColor;
      const doStroke=BAR_STROKE&&!p.color_explicit;
      C.forEach(function(c,i){{
        if(!c)return;
        const x=_pad.left+i*bW+gap,w=bW-gap*2;
        ctx.fillRect(x,_pad.top+_pH-c/_maxC*_pH,w,c/_maxC*_pH);
        if(doStroke){{
          ctx.strokeStyle=BAR_STROKE;ctx.lineWidth=BAR_SW;
          ctx.strokeRect(x,_pad.top+_pH-c/_maxC*_pH,w,c/_maxC*_pH);
        }}
      }});

      /* x grid + labels */
      const xS=niceStep(_xMax-_xMin,Math.min(8,Math.floor(_pW/55)));
      const x0=Math.ceil(_xMin/xS)*xS;
      ctx.fillStyle=SC;ctx.font="10px "+FONT;
      ctx.textAlign="center";ctx.textBaseline="top";
      ctx.strokeStyle=GC;ctx.lineWidth=1;
      for(let xv=x0;xv<=_xMax;xv+=xS){{
        const x=sxD(xv);
        ctx.beginPath();ctx.moveTo(x,_pad.top);ctx.lineTo(x,_pad.top+_pH);ctx.stroke();
        ctx.fillText(fmt(xv),x,_pad.top+_pH+6);
      }}

      /* axes */
      ctx.strokeStyle=AC;ctx.lineWidth=1;
      ctx.beginPath();ctx.moveTo(_pad.left,_pad.top+_pH);ctx.lineTo(_W-_pad.right,_pad.top+_pH);ctx.stroke();
      ctx.beginPath();ctx.moveTo(_pad.left,_pad.top);ctx.lineTo(_pad.left,_pad.top+_pH);ctx.stroke();

      /* mean (dashed) */
      if(p.show_mean){{
        const mx=sxD(p.mean);
        ctx.save();ctx.strokeStyle=RC;ctx.lineWidth=1.5;ctx.setLineDash([6,4]);
        ctx.beginPath();ctx.moveTo(mx,_pad.top);ctx.lineTo(mx,_pad.top+_pH);ctx.stroke();
        ctx.restore();
      }}
      /* median (dotted) */
      if(p.show_median){{
        const mx=sxD(p.median);
        ctx.save();ctx.strokeStyle=RC;ctx.lineWidth=1;ctx.setLineDash([2,3]);
        ctx.beginPath();ctx.moveTo(mx,_pad.top);ctx.lineTo(mx,_pad.top+_pH);ctx.stroke();
        ctx.restore();
      }}
      /* KDE fit curve */
      if(p.kde&&p.kde.length){{
        ctx.save();
        ctx.strokeStyle=p.fit_color||FIT_DEFAULT;
        ctx.lineWidth=FIT_LW;
        ctx.lineJoin="round";ctx.lineCap="round";
        ctx.beginPath();
        for(let k=0;k<p.kde.length;k++){{
          const kx=sxD(p.kde[k][0]),ky=syD(p.kde[k][1]);
          if(k===0)ctx.moveTo(kx,ky);else ctx.lineTo(kx,ky);
        }}
        ctx.stroke();ctx.restore();
      }}
      /* percentile lines */
      if(p.percentiles&&p.percentiles.length){{
        const pColors=["#4CAF50","#8BC34A","#FFC107","#FF9800","#f44336","#E91E63","#9C27B0","#3F51B5"];
        p.percentiles.forEach(function(pc,pi){{
          const px=sxD(pc.v);
          const clr=PCT_C||pColors[pi%pColors.length];
          ctx.save();
          ctx.strokeStyle=clr;ctx.lineWidth=1.2;
          ctx.setLineDash([5,3]);
          ctx.beginPath();ctx.moveTo(px,_pad.top);ctx.lineTo(px,_pad.top+_pH);ctx.stroke();
          ctx.fillStyle=clr;
          ctx.font="bold 9px "+FONT;ctx.textAlign="center";ctx.textBaseline="bottom";
          ctx.fillText(pc.p+"%",px,_pad.top-1);
          ctx.restore();
        }});
      }}
      /* panel title */
      ctx.fillStyle=TC;ctx.font="bold 12px "+FONT;
      ctx.textAlign="center";ctx.textBaseline="top";
      ctx.fillText(p.name,_W/2,6);
    }}

    /* ── scatter ─────────────────────────────────────────── */
    if(p.type==="scatter"){{
      const X=p.x,Y=p.y;
      _xMin=Infinity;_xMax=-Infinity;_yMin=Infinity;_yMax=-Infinity;
      for(let i=0;i<X.length;i++){{
        if(X[i]<_xMin)_xMin=X[i];if(X[i]>_xMax)_xMax=X[i];
        if(Y[i]<_yMin)_yMin=Y[i];if(Y[i]>_yMax)_yMax=Y[i];
      }}
      if(p.curves){{for(const cv of p.curves){{for(let i=0;i<cv.x.length;i++){{
        const xv=cv.x[i];if(xv<_xMin)_xMin=xv;if(xv>_xMax)_xMax=xv;
        const lo=cv.lower?cv.lower[i]:cv.y[i],hi=cv.upper?cv.upper[i]:cv.y[i];
        if(lo!=null&&lo<_yMin)_yMin=lo;if(hi!=null&&hi>_yMax)_yMax=hi;
      }}}}}}
      const xP=(_xMax-_xMin)*0.05||1,yP=(_yMax-_yMin)*0.05||1;
      _xMin-=xP;_xMax+=xP;_yMin-=yP;_yMax+=yP;
      if(VIEW){{_xMin=VIEW.x0;_xMax=VIEW.x1;_yMin=VIEW.y0;_yMax=VIEW.y1;}}

      const CB=p.colorbar;
      _pad={{top:30,right:CB?72:14,bottom:34,left:52}};
      _pW=_W-_pad.left-_pad.right;_pH=_H-_pad.top-_pad.bottom;

      /* grid */
      const yS=niceStep(_yMax-_yMin,5);
      ctx.strokeStyle=GC;ctx.lineWidth=1;
      ctx.fillStyle=SC;ctx.font="10px "+FONT;ctx.textAlign="right";ctx.textBaseline="middle";
      for(let yv=Math.ceil(_yMin/yS)*yS;yv<=_yMax;yv+=yS){{
        const y=sy(yv);
        ctx.beginPath();ctx.moveTo(_pad.left,y);ctx.lineTo(_W-_pad.right,y);ctx.stroke();
        ctx.fillText(fmt(yv),_pad.left-6,y);
      }}
      const xS=niceStep(_xMax-_xMin,Math.min(8,Math.floor(_pW/55)));
      ctx.textAlign="center";ctx.textBaseline="top";
      for(let xv=Math.ceil(_xMin/xS)*xS;xv<=_xMax;xv+=xS){{
        const x=sx(xv);
        ctx.beginPath();ctx.moveTo(x,_pad.top);ctx.lineTo(x,_pad.top+_pH);ctx.stroke();
        ctx.fillText(fmt(xv),x,_pad.top+_pH+6);
      }}

      /* axes */
      ctx.strokeStyle=AC;ctx.lineWidth=1;
      ctx.beginPath();ctx.moveTo(_pad.left,_pad.top+_pH);ctx.lineTo(_W-_pad.right,_pad.top+_pH);ctx.stroke();
      ctx.beginPath();ctx.moveTo(_pad.left,_pad.top);ctx.lineTo(_pad.left,_pad.top+_pH);ctx.stroke();

      /* fitted curves (band + line), drawn under the markers - clipped, since a
         zoomed view leaves data outside the plot */
      ctx.save();ctx.beginPath();ctx.rect(_pad.left,_pad.top,_pW,_pH);ctx.clip();
      if(p.curves){{
        for(const cv of p.curves){{
          if(cv.lower&&cv.upper){{
            ctx.fillStyle=cv.band;ctx.beginPath();let s=false;
            for(let i=0;i<cv.x.length;i++){{if(cv.upper[i]==null)continue;const xx=sx(cv.x[i]),yy=sy(cv.upper[i]);if(!s){{ctx.moveTo(xx,yy);s=true;}}else ctx.lineTo(xx,yy);}}
            for(let i=cv.x.length-1;i>=0;i--){{if(cv.lower[i]==null)continue;ctx.lineTo(sx(cv.x[i]),sy(cv.lower[i]));}}
            ctx.closePath();ctx.fill();
          }}
          ctx.save();ctx.strokeStyle=cv.color;ctx.lineWidth=2.4;ctx.lineJoin="round";ctx.lineCap="round";
          if(cv.dash)ctx.setLineDash([6,4]);
          ctx.beginPath();let st=false;
          for(let i=0;i<cv.x.length;i++){{if(cv.y[i]==null)continue;const xx=sx(cv.x[i]),yy=sy(cv.y[i]);if(!st){{ctx.moveTo(xx,yy);st=true;}}else ctx.lineTo(xx,yy);}}
          ctx.stroke();ctx.restore();
        }}
      }}

      /* markers (per-point colour / size / shape / outline) */
      for(let i=0;i<X.length;i++){{
        const col=p.pcolors?p.pcolors[i]:p.color;
        const r=p.sizes?p.sizes[i]:p.size;
        const sym=p.symbols?p.symbols[i]:p.symbol;
        const ol=p.outlines?p.outlines[i]:p.outline;
        drawPoint(ctx,sx(X[i]),sy(Y[i]),r,sym,col,ol);
      }}
      ctx.restore();

      /* colour bar */
      if(CB){{
        const bw=12,bh=_pH*0.68,bx=_W-_pad.right+18,by=_pad.top+(_pH-bh)/2;
        const g=ctx.createLinearGradient(0,by+bh,0,by);
        for(let s=0;s<CB.stops.length;s++)g.addColorStop(CB.stops[s][0],CB.stops[s][1]);
        ctx.fillStyle=g;ctx.fillRect(bx,by,bw,bh);
        ctx.strokeStyle=AC;ctx.lineWidth=1;ctx.strokeRect(bx,by,bw,bh);
        ctx.fillStyle=SC;ctx.font="9px "+FONT;ctx.textAlign="left";ctx.textBaseline="middle";
        ctx.fillText(fmt(CB.hi),bx+bw+5,by);
        ctx.fillText(fmt((CB.lo+CB.hi)/2),bx+bw+5,by+bh/2);
        ctx.fillText(fmt(CB.lo),bx+bw+5,by+bh);
        if(CB.title){{ctx.save();ctx.fillStyle=TC;ctx.font="10px "+FONT;ctx.textAlign="center";ctx.textBaseline="alphabetic";
          ctx.translate(bx+bw+40,by+bh/2);ctx.rotate(-Math.PI/2);ctx.fillText(CB.title,0,0);ctx.restore();}}
      }}

      /* symbol legend (top-left) */
      if(p.legend&&p.legend.length){{
        ctx.save();ctx.font="10px "+FONT;ctx.textBaseline="middle";ctx.textAlign="left";
        let lx=_pad.left+8;const ly=_pad.top+12;
        for(const it of p.legend){{
          drawPoint(ctx,lx+5,ly,4.5,it.symbol,it.color||TC,null);
          ctx.fillStyle=TC;ctx.fillText(it.label,lx+16,ly);
          lx+=16+ctx.measureText(it.label).width+16;
        }}
        ctx.restore();
      }}

      /* curve legend (line swatches, row below the symbol legend) */
      if(p.curves&&p.curves.some(function(c){{return c.name;}})){{
        ctx.save();ctx.font="10px "+FONT;ctx.textBaseline="middle";ctx.textAlign="left";
        let lx=_pad.left+8;const ly=_pad.top+((p.legend&&p.legend.length)?28:12);
        for(const cv of p.curves){{
          if(!cv.name)continue;
          ctx.strokeStyle=cv.color;ctx.lineWidth=2.4;if(cv.dash)ctx.setLineDash([6,4]);
          ctx.beginPath();ctx.moveTo(lx,ly);ctx.lineTo(lx+16,ly);ctx.stroke();ctx.setLineDash([]);
          ctx.fillStyle=TC;ctx.fillText(cv.name,lx+20,ly);
          lx+=20+ctx.measureText(cv.name).width+14;
        }}
        ctx.restore();
      }}

      /* axis labels */
      ctx.fillStyle=SC;ctx.font="10px "+FONT;
      if(p.x_label){{ctx.textAlign="center";ctx.textBaseline="bottom";ctx.fillText(p.x_label,_pad.left+_pW/2,_H-3);}}
      if(p.y_label){{ctx.save();ctx.translate(12,_pad.top+_pH/2);ctx.rotate(-Math.PI/2);ctx.textAlign="center";ctx.textBaseline="top";ctx.fillText(p.y_label,0,0);ctx.restore();}}

      /* title */
      ctx.fillStyle=TC;ctx.font="bold 12px "+FONT;
      ctx.textAlign="center";ctx.textBaseline="top";
      ctx.fillText(p.name,_W/2,6);
    }}

    /* ── bars (categorical / binned, grouped or stacked) ─── */
    if(p.type==="bars"){{
      const L=p.labels,S=p.series,nL=L.length,nS=S.length,stack=p.mode==="stack";
      /* y-range: stack ⇒ per-slot sums; group ⇒ individual values (allow ±) */
      let yMax=-Infinity,yMin=0;
      if(stack){{
        for(let i=0;i<nL;i++){{let s=0;for(let j=0;j<nS;j++){{const v=S[j].values[i];if(v!=null)s+=v;}}if(s>yMax)yMax=s;}}
      }}else{{
        for(let j=0;j<nS;j++)for(let i=0;i<nL;i++){{const v=S[j].values[i];if(v==null)continue;if(v>yMax)yMax=v;if(v<yMin)yMin=v;}}
      }}
      if(!isFinite(yMax))yMax=1;
      if(yMax===yMin)yMax+=1;
      yMax+=(yMax-yMin)*0.12;                 /* headroom for value labels */
      _yMin=yMin;_yMax=yMax;_maxC=yMax;
      _pad={{top:30,right:14,bottom:48,left:50}};
      _pW=_W-_pad.left-_pad.right;_pH=_H-_pad.top-_pad.bottom;
      const syB=v=>_pad.top+_pH-(v-_yMin)/(_yMax-_yMin)*_pH;
      const y0=syB(0);
      const slotW=_pW/nL,innerPad=slotW*0.12,groupW=slotW-innerPad*2;
      const barW=stack?groupW:groupW/nS,bgap=stack?0:Math.min(3,barW*0.18);
      /* store geometry for hover */
      p._nL=nL;p._nS=nS;p._stack=stack;p._y0=y0;
      p._slotW=slotW;p._innerPad=innerPad;p._groupW=groupW;p._barW=barW;p._bgap=bgap;

      /* y grid + labels */
      const yS=niceStep(_yMax-_yMin,5);
      ctx.strokeStyle=GC;ctx.lineWidth=1;ctx.fillStyle=SC;ctx.font="10px "+FONT;
      ctx.textAlign="right";ctx.textBaseline="middle";
      for(let yv=Math.ceil(_yMin/yS)*yS;yv<=_yMax;yv+=yS){{
        const y=syB(yv);ctx.beginPath();ctx.moveTo(_pad.left,y);ctx.lineTo(_W-_pad.right,y);ctx.stroke();
        ctx.fillText(fmt(yv),_pad.left-6,y);
      }}
      /* axes (x at y=0 baseline) */
      ctx.strokeStyle=AC;ctx.lineWidth=1;
      ctx.beginPath();ctx.moveTo(_pad.left,y0);ctx.lineTo(_W-_pad.right,y0);ctx.stroke();
      ctx.beginPath();ctx.moveTo(_pad.left,_pad.top);ctx.lineTo(_pad.left,_pad.top+_pH);ctx.stroke();

      /* bars */
      for(let i=0;i<nL;i++){{
        const sx0=_pad.left+i*slotW+innerPad;
        if(stack){{
          let acc=0;
          for(let j=0;j<nS;j++){{
            const v=S[j].values[i];if(v==null)continue;
            const yTop=syB(acc+v),yBot=syB(acc);
            ctx.fillStyle=S[j].color;ctx.fillRect(sx0,yTop,groupW,yBot-yTop);
            if(BAR_STROKE){{ctx.strokeStyle=BAR_STROKE;ctx.lineWidth=BAR_SW;ctx.strokeRect(sx0,yTop,groupW,yBot-yTop);}}
            acc+=v;
          }}
          if(p.show_values&&acc!==0){{
            ctx.fillStyle=TC;ctx.font="600 9px "+FONT;ctx.textAlign="center";ctx.textBaseline="bottom";
            ctx.fillText(fmtV(acc),sx0+groupW/2,syB(acc)-3);
          }}
        }}else{{
          for(let j=0;j<nS;j++){{
            const v=S[j].values[i];if(v==null)continue;
            const bx=sx0+j*barW+bgap/2,bw=barW-bgap;
            const yv=syB(v),top=Math.min(yv,y0),h=Math.abs(yv-y0);
            ctx.fillStyle=S[j].color;ctx.fillRect(bx,top,bw,h);
            if(BAR_STROKE){{ctx.strokeStyle=BAR_STROKE;ctx.lineWidth=BAR_SW;ctx.strokeRect(bx,top,bw,h);}}
            if(p.show_values){{
              ctx.fillStyle=TC;ctx.font="600 9px "+FONT;ctx.textAlign="center";
              ctx.textBaseline=v<0?"top":"bottom";
              ctx.fillText(fmtV(v),bx+bw/2,v<0?yv+3:yv-3);
            }}
          }}
        }}
      }}

      /* x labels — rotate when a label would overflow its slot */
      ctx.fillStyle=SC;ctx.font="10px "+FONT;
      let maxLW=0;for(let i=0;i<nL;i++)maxLW=Math.max(maxLW,ctx.measureText(L[i]).width);
      const rot=maxLW>slotW*0.92;
      for(let i=0;i<nL;i++){{
        const cx=_pad.left+i*slotW+slotW/2;
        if(rot){{
          ctx.save();ctx.translate(cx,_pad.top+_pH+8);ctx.rotate(-Math.PI/5);
          ctx.textAlign="right";ctx.textBaseline="middle";ctx.fillText(L[i],0,0);ctx.restore();
        }}else{{
          ctx.textAlign="center";ctx.textBaseline="top";ctx.fillText(L[i],cx,_pad.top+_pH+7);
        }}
      }}

      /* axis labels */
      ctx.fillStyle=SC;ctx.font="10px "+FONT;
      if(p.x_label){{ctx.textAlign="center";ctx.textBaseline="bottom";ctx.fillText(p.x_label,_pad.left+_pW/2,_H-3);}}
      if(p.y_label){{ctx.save();ctx.translate(12,_pad.top+_pH/2);ctx.rotate(-Math.PI/2);ctx.textAlign="center";ctx.textBaseline="top";ctx.fillText(p.y_label,0,0);ctx.restore();}}

      /* legend (multi-series only) */
      if(!p.single&&nS>1){{
        ctx.save();ctx.font="10px "+FONT;ctx.textBaseline="middle";ctx.textAlign="left";
        let lx=_pad.left+8;const ly=_pad.top+12;
        for(let j=0;j<nS;j++){{
          ctx.fillStyle=S[j].color;ctx.fillRect(lx,ly-4,12,8);
          ctx.fillStyle=TC;ctx.fillText(S[j].name,lx+16,ly);
          lx+=16+ctx.measureText(S[j].name).width+16;
        }}
        ctx.restore();
      }}
      /* title */
      ctx.fillStyle=TC;ctx.font="bold 12px "+FONT;ctx.textAlign="center";ctx.textBaseline="top";
      ctx.fillText(p.name,_W/2,6);
    }}

    /* ── curve (points + fitted line + confidence band) ──── */
    if(p.type==="curve"){{
      const GX=p.x;
      const fr=p.frames?p.frames[AF]:null;
      const mean = fr?fr.mean:p.mean;
      const lower= fr?fr.lower:p.lower;
      const upper= fr?fr.upper:p.upper;
      const prior= fr?fr.prior:p.prior;
      const PX   = fr?fr.px:p.px;
      const PY   = fr?fr.py:p.py;
      const PLAB = fr?fr.plabels:p.plabels;
      const PCOL = fr?fr.pcolors:p.pcolors;
      const PSZ  = fr?fr.psizes:p.psizes;
      const EXTRAS = p.extras;   /* extra reference lines: y/name/color */
      const baseM=(p.frames&&p.base_idx!=null)?p.frames[p.base_idx].mean:null;
      p._cur={{x:GX,mean:mean,lower:lower,upper:upper,px:PX,py:PY,plabels:PLAB}};

      _xMin=Infinity;_xMax=-Infinity;_yMin=Infinity;_yMax=-Infinity;
      for(let i=0;i<GX.length;i++){{if(GX[i]<_xMin)_xMin=GX[i];if(GX[i]>_xMax)_xMax=GX[i];}}
      function ext(a){{if(!a)return;for(let i=0;i<a.length;i++){{const v=a[i];if(v==null)continue;if(v<_yMin)_yMin=v;if(v>_yMax)_yMax=v;}}}}
      ext(mean);ext(lower);ext(upper);ext(prior);ext(baseM);ext(PY);
      if(PX){{for(let i=0;i<PX.length;i++){{if(PX[i]<_xMin)_xMin=PX[i];if(PX[i]>_xMax)_xMax=PX[i];}}}}
      if(!isFinite(_yMin)){{_yMin=0;_yMax=1;}}
      const yP=(_yMax-_yMin)*0.10||1,xP=(_xMax-_xMin)*0.03||0.5;
      _xMax+=xP;_yMin-=yP;_yMax+=yP;
      if(XLINK.x){{_xMin=XLINK.x[0];_xMax=XLINK.x[1];}}
      if(VIEW){{_yMin=VIEW.y0;_yMax=VIEW.y1;}}

      _pad={{top:32,right:16,bottom:40,left:56}};
      _pW=_W-_pad.left-_pad.right;_pH=_H-_pad.top-_pad.bottom;

      /* grid + labels */
      const yS=niceStep(_yMax-_yMin,5);
      ctx.strokeStyle=GC;ctx.lineWidth=1;ctx.fillStyle=SC;ctx.font="10px "+FONT;
      ctx.textAlign="right";ctx.textBaseline="middle";
      for(let yv=Math.ceil(_yMin/yS)*yS;yv<=_yMax;yv+=yS){{
        const y=sy(yv);ctx.beginPath();ctx.moveTo(_pad.left,y);ctx.lineTo(_W-_pad.right,y);ctx.stroke();
        ctx.fillText(fmt(yv),_pad.left-6,y);
      }}
      const xS=niceStep(_xMax-_xMin,Math.min(8,Math.floor(_pW/60)));
      ctx.textAlign="center";ctx.textBaseline="top";
      for(let xv=Math.ceil(_xMin/xS)*xS;xv<=_xMax;xv+=xS){{
        const x=sx(xv);ctx.beginPath();ctx.moveTo(x,_pad.top);ctx.lineTo(x,_pad.top+_pH);ctx.stroke();
        ctx.fillText(fmt(xv),x,_pad.top+_pH+6);
      }}
      /* axes */
      ctx.strokeStyle=AC;ctx.lineWidth=1;
      ctx.beginPath();ctx.moveTo(_pad.left,_pad.top+_pH);ctx.lineTo(_W-_pad.right,_pad.top+_pH);ctx.stroke();
      ctx.beginPath();ctx.moveTo(_pad.left,_pad.top);ctx.lineTo(_pad.left,_pad.top+_pH);ctx.stroke();

      /* polyline helper */
      function pline(Y,dash){{ctx.save();if(dash)ctx.setLineDash(dash);ctx.beginPath();let s=false;
        for(let i=0;i<GX.length;i++){{if(Y[i]==null)continue;const X=sx(GX[i]),yy=sy(Y[i]);if(!s){{ctx.moveTo(X,yy);s=true;}}else ctx.lineTo(X,yy);}}
        ctx.stroke();ctx.restore();}}

      ctx.save();ctx.beginPath();ctx.rect(_pad.left,_pad.top,_pW,_pH);ctx.clip();   /* zoomed views leave data outside */
      /* confidence band (fill between lower & upper) */
      if(lower&&upper){{
        ctx.fillStyle=p.band_color;ctx.beginPath();let s=false;
        for(let i=0;i<GX.length;i++){{if(upper[i]==null)continue;const X=sx(GX[i]),yy=sy(upper[i]);if(!s){{ctx.moveTo(X,yy);s=true;}}else ctx.lineTo(X,yy);}}
        for(let i=GX.length-1;i>=0;i--){{if(lower[i]==null)continue;ctx.lineTo(sx(GX[i]),sy(lower[i]));}}
        ctx.closePath();ctx.fill();
      }}
      /* pinned base-date ghost */
      if(baseM){{ctx.strokeStyle=RC;ctx.lineWidth=1.5;pline(baseM,[5,4]);}}
      /* second curve / prior */
      if(prior){{ctx.strokeStyle=p.prior_color||RC;ctx.lineWidth=1.6;pline(prior,[7,4]);}}
      /* extra reference lines */
      if(EXTRAS){{for(const e of EXTRAS){{if(!e.y)continue;ctx.strokeStyle=e.color;ctx.lineWidth=1.6;pline(e.y,[4,3]);}}}}
      /* fitted mean */
      if(mean){{ctx.strokeStyle=p.color;ctx.lineWidth=2.6;ctx.lineJoin="round";ctx.lineCap="round";pline(mean,null);}}
      /* observed points (per-point colour when pcolors given, e.g. by instrument type) */
      if(PX&&PY){{
        for(let i=0;i<PX.length;i++){{if(PY[i]==null)continue;
          ctx.fillStyle=(PCOL&&PCOL[i])?PCOL[i]:p.point_color;
          ctx.beginPath();ctx.arc(sx(PX[i]),sy(PY[i]),(PSZ&&PSZ[i]!=null?PSZ[i]:3.2),0,Math.PI*2);ctx.fill();}}}}

      /* trade markers — N LONG (green ^) / SHORT (red v) per frame, each with a drop-line to the curve */
      if(fr&&fr.trade){{
        const T=fr.trade;
        /* linear-interp the fitted curve y at an x (the drop-line's curve endpoint) */
        function cyAt(xv){{
          if(!mean)return null;let lo=-1;
          for(let i=0;i<GX.length;i++){{if(GX[i]<=xv)lo=i;else break;}}
          if(lo<0)return mean[0];if(lo>=GX.length-1)return mean[GX.length-1];
          const a=mean[lo],b=mean[lo+1];if(a==null||b==null)return a==null?b:a;
          return a+(b-a)*((xv-GX[lo])/(GX[lo+1]-GX[lo]));
        }}
        function tri(cx,cy,col,up,lbl){{
          ctx.save();ctx.fillStyle=col;ctx.strokeStyle="rgba(0,0,0,0.45)";ctx.lineWidth=0.8;ctx.beginPath();const s=5;
          if(up){{ctx.moveTo(cx,cy-s);ctx.lineTo(cx-s,cy+s);ctx.lineTo(cx+s,cy+s);}}
          else{{ctx.moveTo(cx,cy+s);ctx.lineTo(cx-s,cy-s);ctx.lineTo(cx+s,cy-s);}}
          ctx.closePath();ctx.fill();ctx.stroke();
          if(lbl){{ctx.fillStyle=col;ctx.font="600 9px "+MARK_FONT;ctx.textAlign="center";
            ctx.textBaseline=up?"top":"bottom";ctx.fillText(lbl,cx,up?cy+s+3:cy-s-3);}}
          ctx.restore();
        }}
        function drawSide(arr,col,up){{
          if(!arr)return;
          for(let i=0;i<arr.length;i++){{
            const m=arr[i];if(!m||m[0]==null||m[1]==null)continue;
            const cyv=cyAt(m[0]);                                  /* dashed vertical drop-line to the curve */
            if(cyv!=null){{ctx.save();ctx.strokeStyle=col;ctx.globalAlpha=0.5;ctx.lineWidth=1;ctx.setLineDash([3,3]);
              ctx.beginPath();ctx.moveTo(sx(m[0]),sy(m[1]));ctx.lineTo(sx(m[0]),sy(cyv));ctx.stroke();ctx.restore();}}
            tri(sx(m[0]),sy(m[1]),col,up,m[2]);
          }}
        }}
        drawSide(T.long,LONG_C,true);
        drawSide(T.short,SHORT_C,false);
      }}
      ctx.restore();

      /* axis labels */
      ctx.fillStyle=SC;ctx.font="10px "+FONT;
      if(p.x_label){{ctx.textAlign="center";ctx.textBaseline="bottom";ctx.fillText(p.x_label,_pad.left+_pW/2,_H-4);}}
      if(p.y_label){{ctx.save();ctx.translate(13,_pad.top+_pH/2);ctx.rotate(-Math.PI/2);ctx.textAlign="center";ctx.textBaseline="top";ctx.fillText(p.y_label,0,0);ctx.restore();}}

      /* legend (top-left, below title) */
      (function(){{
        const ptItems=(p.plegend&&PX&&PY)?p.plegend.map(e=>({{c:e.c,t:e.t,dot:true}}))
                     :((PX&&PY)?[{{c:p.point_color,t:p.point_name,dot:true}}]:[]);
        const items=[
          mean?{{c:p.color,t:p.mean_name}}:null,
          (lower&&upper)?{{c:p.band_color,t:p.band_label,box:true}}:null,
          prior?{{c:p.prior_color||RC,t:p.prior_name,dash:true}}:null,
          baseM?{{c:RC,t:p.base_name,dash:true}}:null,
          ...(EXTRAS?EXTRAS.map(e=>({{c:e.color,t:e.name,dash:true}})):[]),
          ...ptItems,
        ].filter(Boolean);
        if(!items.length)return;
        ctx.save();ctx.font="10px "+FONT;ctx.textBaseline="middle";ctx.textAlign="left";
        let lx=_pad.left+8;const ly=_pad.top+12;
        for(const it of items){{
          ctx.strokeStyle=it.c;ctx.fillStyle=it.c;ctx.lineWidth=2.4;
          if(it.box){{ctx.globalAlpha=1;ctx.fillRect(lx,ly-4,14,8);}}
          else if(it.dot){{ctx.beginPath();ctx.arc(lx+7,ly,3,0,Math.PI*2);ctx.fill();}}
          else{{ctx.save();if(it.dash)ctx.setLineDash([5,3]);ctx.beginPath();ctx.moveTo(lx,ly);ctx.lineTo(lx+15,ly);ctx.stroke();ctx.restore();}}
          ctx.fillStyle=TC;ctx.fillText(it.t,lx+21,ly);
          lx+=21+ctx.measureText(it.t).width+18;
        }}
        ctx.restore();
      }})();

      /* title */
      ctx.fillStyle=TC;ctx.font="bold 12px "+FONT;ctx.textAlign="center";ctx.textBaseline="top";
      ctx.fillText(p.name,_W/2,6);
    }}

    /* ── spread (active − base, per x, vs zero line) ─────── */
    if(p.type==="spread"){{
      const GX=p.x;let YV;
      if(p.frames){{
        const a=p.frames[AF].mean,bi=(p.base_idx!=null?p.base_idx:0),b=p.frames[bi].mean;
        YV=GX.map(function(_,i){{return (a&&b&&a[i]!=null&&b[i]!=null)?(a[i]-b[i]):null;}});
      }} else YV=p.static||[];
      p._cur={{x:GX,y:YV}};

      _xMin=Infinity;_xMax=-Infinity;let m=1e-9;
      for(let i=0;i<GX.length;i++){{if(GX[i]<_xMin)_xMin=GX[i];if(GX[i]>_xMax)_xMax=GX[i];const v=YV[i];if(v!=null&&Math.abs(v)>m)m=Math.abs(v);}}
      const xP=(_xMax-_xMin)*0.03||0.5;_xMax+=xP;_yMin=-m*1.25;_yMax=m*1.25;
      if(XLINK.x){{_xMin=XLINK.x[0];_xMax=XLINK.x[1];}}
      if(VIEW){{_yMin=VIEW.y0;_yMax=VIEW.y1;}}

      _pad={{top:30,right:16,bottom:40,left:56}};
      _pW=_W-_pad.left-_pad.right;_pH=_H-_pad.top-_pad.bottom;

      const yS=niceStep(_yMax-_yMin,4);
      ctx.strokeStyle=GC;ctx.lineWidth=1;ctx.fillStyle=SC;ctx.font="10px "+FONT;
      ctx.textAlign="right";ctx.textBaseline="middle";
      for(let yv=Math.ceil(_yMin/yS)*yS;yv<=_yMax;yv+=yS){{const y=sy(yv);ctx.beginPath();ctx.moveTo(_pad.left,y);ctx.lineTo(_W-_pad.right,y);ctx.stroke();ctx.fillText(fmt(yv),_pad.left-6,y);}}
      const xS=niceStep(_xMax-_xMin,Math.min(8,Math.floor(_pW/60)));
      ctx.textAlign="center";ctx.textBaseline="top";
      for(let xv=Math.ceil(_xMin/xS)*xS;xv<=_xMax;xv+=xS){{const x=sx(xv);ctx.beginPath();ctx.moveTo(x,_pad.top);ctx.lineTo(x,_pad.top+_pH);ctx.stroke();ctx.fillText(fmt(xv),x,_pad.top+_pH+6);}}

      const y0=sy(0);
      ctx.save();ctx.beginPath();ctx.rect(_pad.left,_pad.top,_pW,_pH);ctx.clip();
      /* fill between spread and zero */
      const pts=[];for(let i=0;i<GX.length;i++){{if(YV[i]!=null)pts.push([sx(GX[i]),sy(YV[i])]);}}
      if(pts.length){{ctx.save();ctx.globalAlpha=0.16;ctx.fillStyle=p.color;ctx.beginPath();
        ctx.moveTo(pts[0][0],y0);for(const q of pts)ctx.lineTo(q[0],q[1]);
        ctx.lineTo(pts[pts.length-1][0],y0);ctx.closePath();ctx.fill();ctx.restore();}}
      /* zero line */
      ctx.save();ctx.strokeStyle=AC;ctx.lineWidth=1.2;ctx.setLineDash([4,3]);ctx.beginPath();ctx.moveTo(_pad.left,y0);ctx.lineTo(_W-_pad.right,y0);ctx.stroke();ctx.restore();
      /* y axis */
      ctx.strokeStyle=AC;ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(_pad.left,_pad.top);ctx.lineTo(_pad.left,_pad.top+_pH);ctx.stroke();
      /* spread line */
      ctx.strokeStyle=p.color;ctx.lineWidth=2.2;ctx.lineJoin="round";ctx.lineCap="round";ctx.beginPath();let s2=false;
      for(let i=0;i<GX.length;i++){{if(YV[i]==null)continue;const X=sx(GX[i]),yy=sy(YV[i]);if(!s2){{ctx.moveTo(X,yy);s2=true;}}else ctx.lineTo(X,yy);}}
      ctx.stroke();
      ctx.restore();

      ctx.fillStyle=SC;ctx.font="10px "+FONT;
      if(p.x_label){{ctx.textAlign="center";ctx.textBaseline="bottom";ctx.fillText(p.x_label,_pad.left+_pW/2,_H-4);}}
      if(p.y_label){{ctx.save();ctx.translate(13,_pad.top+_pH/2);ctx.rotate(-Math.PI/2);ctx.textAlign="center";ctx.textBaseline="top";ctx.fillText(p.y_label,0,0);ctx.restore();}}
      ctx.fillStyle=TC;ctx.font="bold 12px "+FONT;ctx.textAlign="center";ctx.textBaseline="top";ctx.fillText(p.name,_W/2,6);
    }}
  }}

  draw();
  REDRAW.push(draw);
  new ResizeObserver(function(){{draw()}}).observe(cell);

  /* ── Crosshair + Tooltip interaction ─────────────────── */
  function onHover(e){{
    const rect=cvs.getBoundingClientRect();
    const mx=e.clientX-rect.left, my=e.clientY-rect.top;
    const dpr=window.devicePixelRatio||1;
    const oc=ovl.getContext("2d");
    oc.setTransform(dpr,0,0,dpr,0,0);
    oc.clearRect(0,0,_W,_H);

    if(!_pad||mx<_pad.left||mx>_W-_pad.right||my<_pad.top||my>_pad.top+_pH){{
      tip.style.display="none";return;
    }}

    /* ── crosshair lines ──────────────────────────────── */
    oc.save();
    oc.strokeStyle=CH_C;oc.lineWidth=1;oc.setLineDash([4,3]);
    oc.beginPath();oc.moveTo(mx,_pad.top);oc.lineTo(mx,_pad.top+_pH);oc.stroke();
    oc.beginPath();oc.moveTo(_pad.left,my);oc.lineTo(_W-_pad.right,my);oc.stroke();
    oc.restore();

    /* ── distribution hover ───────────────────────────── */
    if(p.type==="distribution"){{
      const C=p.counts,E=p.edges;
      const bW=p._barW||(_pW/C.length);
      const idx=Math.floor((mx-_pad.left)/bW);
      if(idx<0||idx>=C.length){{tip.style.display="none";return;}}
      const lo=E[idx],hi=E[idx+1],cnt=C[idx],pct=(cnt/p._total*100).toFixed(1);
      /* highlight bar */
      const gap=p._gap||1;
      const bx=_pad.left+idx*bW+gap,bw=bW-gap*2;
      const bh=cnt/_maxC*_pH;
      oc.save();oc.fillStyle=p.color;oc.globalAlpha=0.35;
      oc.fillRect(bx,_pad.top+_pH-bh,bw,bh);oc.restore();
      /* dot on bar top */
      oc.save();oc.fillStyle=p.color;
      oc.beginPath();oc.arc(bx+bw/2,_pad.top+_pH-bh,4,0,Math.PI*2);oc.fill();
      oc.restore();
      tip.innerHTML="<b>"+fmt(lo)+" — "+fmt(hi)+"</b><br>"
        +"Count: "+cnt+"<br>"+pct+"% of total";
    }}

    /* ── scatter hover (nearest point) ────────────────── */
    if(p.type==="scatter"){{
      const X=p.x,Y=p.y;let best=-1,bd=Infinity;
      for(let i=0;i<X.length;i++){{
        const dx=sx(X[i])-mx,dy=sy(Y[i])-my;
        const d=dx*dx+dy*dy;
        if(d<bd){{bd=d;best=i;}}
      }}
      if(best<0||Math.sqrt(bd)>34){{tip.style.display="none";return;}}
      const px=sx(X[best]),py=sy(Y[best]);
      const r=p.sizes?p.sizes[best]:p.size;
      const col=p.pcolors?p.pcolors[best]:p.color;
      /* snap crosshair to point, then ring it */
      oc.clearRect(0,0,_W,_H);
      oc.save();oc.strokeStyle=CH_C;oc.lineWidth=1;oc.setLineDash([4,3]);
      oc.beginPath();oc.moveTo(px,_pad.top);oc.lineTo(px,_pad.top+_pH);oc.stroke();
      oc.beginPath();oc.moveTo(_pad.left,py);oc.lineTo(_W-_pad.right,py);oc.stroke();
      oc.restore();
      oc.save();oc.strokeStyle=col;oc.lineWidth=2;
      oc.beginPath();oc.arc(px,py,r+5,0,Math.PI*2);oc.stroke();
      oc.restore();
      let h="";
      if(p.plabels&&p.plabels[best])h+="<b>"+p.plabels[best]+"</b>";
      if(p.hover&&p.hover.length){{
        for(const f of p.hover)h+=(h?"<br>":"")+"<span style='color:"+SC+"'>"+f.t+"</span> "+f.v[best];
      }}else{{
        h+=(h?"<br>":"")+"<b>"+(p.x_label||"x")+":</b> "+fmt(X[best])+"<br><b>"+(p.y_label||"y")+":</b> "+fmt(Y[best]);
      }}
      tip.innerHTML=h;
    }}

    /* ── bars hover (slot + bar/segment under cursor) ───── */
    if(p.type==="bars"){{
      const L=p.labels,S=p.series,nL=p._nL,nS=p._nS,stack=p._stack;
      const slotW=p._slotW,innerPad=p._innerPad,groupW=p._groupW,barW=p._barW,bgap=p._bgap,y0=p._y0;
      const i=Math.floor((mx-_pad.left)/slotW);
      if(i<0||i>=nL){{tip.style.display="none";return;}}
      const sx0=_pad.left+i*slotW+innerPad;
      const syB=v=>_pad.top+_pH-(v-_yMin)/(_yMax-_yMin)*_pH;
      let h2="<b>"+L[i]+"</b>";
      if(stack){{
        let acc=0,tot=0,jHit=-1;
        for(let j=0;j<nS;j++){{const v=S[j].values[i];if(v!=null)tot+=v;}}
        for(let j=0;j<nS;j++){{
          const v=S[j].values[i];if(v==null)continue;
          const yTop=syB(acc+v),yBot=syB(acc);
          if(my>=yTop&&my<=yBot){{jHit=j;
            oc.save();oc.strokeStyle=S[j].color;oc.lineWidth=2;oc.strokeRect(sx0,yTop,groupW,yBot-yTop);oc.restore();}}
          acc+=v;
        }}
        for(let j=0;j<nS;j++){{const v=S[j].values[i];if(v==null)continue;const hl=(j===jHit)?"font-weight:600":"";
          h2+="<br><span style='color:"+S[j].color+"'>&#9632;</span> "
            +(p.single?"":S[j].name+": ")+"<span style='"+hl+"'>"+fmtV(v)+"</span>";}}
        if(nS>1)h2+="<br><b>total:</b> "+fmtV(tot);
      }}else{{
        const jHit=Math.floor((mx-sx0)/barW);
        for(let j=0;j<nS;j++){{
          const v=S[j].values[i];if(v==null||j!==jHit)continue;
          const bx=sx0+j*barW+bgap/2,bw=barW-bgap,yv=syB(v),top=Math.min(yv,y0),hh=Math.abs(yv-y0);
          oc.save();oc.strokeStyle=S[j].color;oc.lineWidth=2;oc.strokeRect(bx,top,bw,hh);oc.restore();
        }}
        for(let j=0;j<nS;j++){{const v=S[j].values[i];if(v==null)continue;const hl=(j===jHit)?"font-weight:600":"";
          h2+="<br><span style='color:"+S[j].color+"'>&#9632;</span> "
            +(p.single?"":S[j].name+": ")+"<span style='"+hl+"'>"+fmtV(v)+"</span>";}}
      }}
      tip.innerHTML=h2;
    }}

    /* ── curve hover (nearest point, else fit+band at x) ── */
    if(p.type==="curve"&&p._cur){{
      const cu=p._cur,GX=cu.x;
      let best=-1,bd=Infinity;
      if(cu.px&&cu.py){{for(let i=0;i<cu.px.length;i++){{if(cu.py[i]==null)continue;const dx=sx(cu.px[i])-mx,dy=sy(cu.py[i])-my,d=dx*dx+dy*dy;if(d<bd){{bd=d;best=i;}}}}}}
      const XL=p.x_label||"x",YL=p.y_label||"y";
      if(best>=0&&Math.sqrt(bd)<26){{
        const X=sx(cu.px[best]),Y=sy(cu.py[best]);
        oc.save();oc.strokeStyle=p.point_color;oc.lineWidth=2;oc.beginPath();oc.arc(X,Y,7,0,Math.PI*2);oc.stroke();oc.restore();
        let h=(cu.plabels&&cu.plabels[best]?"<b>"+cu.plabels[best]+"</b><br>":"")+"<b>"+XL+":</b> "+fmt(cu.px[best])+"<br><b>"+YL+":</b> "+fmt(cu.py[best]);
        if(cu.mean){{let gi=0,gd=Infinity;for(let i=0;i<GX.length;i++){{const dd=Math.abs(GX[i]-cu.px[best]);if(dd<gd){{gd=dd;gi=i;}}}}
          if(cu.mean[gi]!=null)h+="<br><b>"+(p.resid_label||"resid")+":</b> "+fmt(cu.py[best]-cu.mean[gi]);}}
        tip.innerHTML=h;
      }}else{{
        let gi=0,gd=Infinity;for(let i=0;i<GX.length;i++){{const dd=Math.abs(sx(GX[i])-mx);if(dd<gd){{gd=dd;gi=i;}}}}
        let h="<b>"+XL+":</b> "+fmt(GX[gi]);
        if(cu.mean&&cu.mean[gi]!=null)h+="<br><b>"+(p.mean_name||"fit")+":</b> "+fmt(cu.mean[gi]);
        if(cu.lower&&cu.upper&&cu.lower[gi]!=null)h+="<br><b>band:</b> "+fmt(cu.lower[gi])+" – "+fmt(cu.upper[gi]);
        if(cu.mean&&cu.mean[gi]!=null){{oc.save();oc.fillStyle=p.color;oc.beginPath();oc.arc(sx(GX[gi]),sy(cu.mean[gi]),4,0,Math.PI*2);oc.fill();oc.restore();}}
        tip.innerHTML=h;
      }}
    }}

    /* ── spread hover ─────────────────────────────────── */
    if(p.type==="spread"&&p._cur){{
      const cu=p._cur,GX=cu.x;
      let gi=0,gd=Infinity;for(let i=0;i<GX.length;i++){{const dd=Math.abs(sx(GX[i])-mx);if(dd<gd){{gd=dd;gi=i;}}}}
      let h="<b>x:</b> "+fmt(GX[gi]);
      if(cu.y[gi]!=null){{h+="<br><b>&Delta;:</b> "+fmt(cu.y[gi]);
        oc.save();oc.fillStyle=p.color;oc.beginPath();oc.arc(sx(GX[gi]),sy(cu.y[gi]),4,0,Math.PI*2);oc.fill();oc.restore();}}
      tip.innerHTML=h;
    }}

    /* ── position tooltip ─────────────────────────────── */
    tip.style.display="block";
    let tx=mx+14,ty=my-10;
    const tw=tip.offsetWidth,th=tip.offsetHeight;
    if(tx+tw>_W-4)tx=mx-tw-10;
    if(ty<_pad.top)ty=_pad.top+4;
    if(ty+th>_pad.top+_pH)ty=_pad.top+_pH-th-4;
    tip.style.left=tx+"px";tip.style.top=ty+"px";
  }}
  function onLeave(){{
    const dpr=window.devicePixelRatio||1;
    const oc=ovl.getContext("2d");oc.setTransform(dpr,0,0,dpr,0,0);
    oc.clearRect(0,0,_W,_H);tip.style.display="none";
  }}
  cvs.style.position="relative";cvs.style.zIndex="1";
  ovl.style.zIndex="2";ovl.style.pointerEvents="none";
  /* invisible hit area over cell */
  const hit=document.createElement("div");
  hit.style.cssText="position:absolute;top:0;left:0;width:100%;height:100%;z-index:3;cursor:crosshair";
  cell.appendChild(hit);
  hit.addEventListener("mousemove",onHover);
  hit.addEventListener("mouseleave",onLeave);

  /* ── zoom & pan, the way the price charts do it: wheel in the plot zooms
     both axes about the cursor; wheel over an axis strip zooms that axis
     alone; drag in the plot pans; drag along an axis stretches it;
     double-click resets to the data. ── */
  const ZOOM=(p.type==="scatter"||p.type==="curve"||p.type==="spread");
  const LINKED=(p.type!=="scatter");   /* curve+spread share one x view */
  if(ZOOM){{
    const inPlot=(x,y)=>x>=_pad.left&&x<=_pad.left+_pW&&y>=_pad.top&&y<=_pad.top+_pH;
    const zone=(x,y)=>inPlot(x,y)?"plot":(y>_pad.top+_pH&&x>=_pad.left?"x":(x<_pad.left&&y<=_pad.top+_pH?"y":""));
    const cur=()=>({{x0:_xMin,x1:_xMax,y0:_yMin,y1:_yMax}});   /* what is on screen now */
    const apply=(v,touchX)=>{{
      if(LINKED){{ VIEW={{y0:v.y0,y1:v.y1}}; if(touchX){{XLINK.x=[v.x0,v.x1];REDRAW.forEach(function(f){{f();}});return;}} }}
      else VIEW=v;
      draw();
    }};
    const dx=px=>_xMin+(px-_pad.left)/_pW*(_xMax-_xMin);
    const dy=py=>_yMax-(py-_pad.top)/_pH*(_yMax-_yMin);
    const scale=(v,fx,fy,ax,ay)=>({{x0:ax+(v.x0-ax)*fx,x1:ax+(v.x1-ax)*fx,y0:ay+(v.y0-ay)*fy,y1:ay+(v.y1-ay)*fy}});
    const pos=e=>{{const r=hit.getBoundingClientRect();return [e.clientX-r.left,e.clientY-r.top];}};
    hit.addEventListener("wheel",function(e){{
      if(!_pad)return;const [x,y]=pos(e);const z=zone(x,y);if(!z)return;
      e.preventDefault();
      const f=e.deltaY>0?1.12:1/1.12;
      apply(scale(cur(),z==="y"?1:f,z==="x"?1:f,dx(x),dy(y)),z!=="y");
    }},{{passive:false}});
    let drag=null;
    hit.addEventListener("pointerdown",function(e){{
      if(!_pad)return;const [x,y]=pos(e);const z=zone(x,y);if(!z)return;
      drag={{z:z,x:x,y:y,v:cur()}};
      try{{hit.setPointerCapture(e.pointerId);}}catch(err){{}}
      e.preventDefault();
    }});
    hit.addEventListener("pointermove",function(e){{
      if(!_pad)return;const [x,y]=pos(e);
      if(!drag){{const z=zone(x,y);hit.style.cursor=z==="x"?"ew-resize":z==="y"?"ns-resize":"crosshair";return;}}
      const v=drag.v;
      if(drag.z==="plot"){{
        const mx=(x-drag.x)*(v.x1-v.x0)/_pW,my=(y-drag.y)*(v.y1-v.y0)/_pH;
        apply({{x0:v.x0-mx,x1:v.x1-mx,y0:v.y0+my,y1:v.y1+my}},true);
      }}else if(drag.z==="x"){{
        apply(scale(v,Math.exp(-(x-drag.x)/200),1,(v.x0+v.x1)/2,0),true);
      }}else{{
        apply(scale(v,1,Math.exp((y-drag.y)/200),0,(v.y0+v.y1)/2),false);
      }}
    }});
    const endDrag=function(){{drag=null;}};
    hit.addEventListener("pointerup",endDrag);
    hit.addEventListener("pointercancel",endDrag);
    hit.addEventListener("dblclick",function(){{VIEW=null;if(LINKED){{XLINK.x=null;REDRAW.forEach(function(f){{f();}});}}else draw();}});
  }}
}});

/* ── date scrubber: redraw every framed panel on move ── */
if(SLIDER){{
  const r=document.getElementById("sl-range"),dl=document.getElementById("sl-date");
  if(r){{r.addEventListener("input",function(){{AF=+r.value;if(dl)dl.textContent=SLIDER.dates[AF];REDRAW.forEach(function(f){{f();}});}});}}
}}
}})();
</script></body></html>"""

    # ── Display Methods (same pipeline as Chart) ──────────────────────────

    def _repr_html_(self) -> str:
        import base64
        html = self._build_html()
        b64 = base64.b64encode(html.encode("utf-8")).decode("ascii")
        h = self._height + (40 if self._title else 10) + (36 if self._slider else 0)
        uid = f"sc{id(self)}"
        return (
            f'<div id="{uid}" style="width:100%;height:{h}px;'
            f'border-radius:12px;overflow:hidden;"></div><script>'
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
        """Display in a Jupyter notebook."""
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
        """Return a Dash ``html.Iframe`` component."""
        from dash import html
        default_style = {
            "width": "100%",
            "height": f"{self._height + (40 if self._title else 10) + (36 if self._slider else 0)}px",
            "border": "none",
            "borderRadius": "4px",
        }
        if style:
            default_style.update(style)
        return html.Iframe(
            id=id or "signum-stat",
            srcDoc=self._build_html(),
            style=default_style,
        )

    def to_streamlit(self, height: Optional[int] = None):
        """Render inside a Streamlit app."""
        import streamlit.components.v1 as components
        components.html(
            self._build_html(),
            height=height or self._height + (40 if self._title else 10) + (36 if self._slider else 0),
        )
