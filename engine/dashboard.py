"""Dashboard - Multi-pane synchronized financial charts.

Creates vertically stacked chart panes with synchronized time axes
and crosshairs — the LC equivalent of Plotly make_subplots(shared_xaxes=True).

Pattern (matches TKAN / Cogilator multi-panel layouts):
    from signum import Chart, Dashboard

    price = Chart(height=350).candlestick(df).volume(df).signals(sig_df)
    equity = Chart(height=200).area(equity_df, name="Equity")
    ivol   = Chart(height=150).line(ivol_df, name="IVol", color="orange")

    dash = Dashboard(
        panes=[price, equity, ivol],
        titles=["LVC Price + Trades", "Equity Curve", "Implied Volatility"],
    )
    dash.show()
"""

import html as _html
import json
import math
from typing import Dict, List, Optional

import pandas as pd

from .chart import Chart, _get_lc_js, _LOGO_B64
from .themes import THEMES


class Dashboard:
    """Vertically stacked, time-synced chart panes."""

    def __init__(
        self,
        panes: Optional[List[Chart]] = None,
        theme: Optional[str] = None,
        titles: Optional[List[str]] = None,
        gap: int = 2,
        logo: bool = True,
    ):
        self._panes: List[Chart] = panes or []
        self._titles: List[str] = titles or []
        self._logo = logo
        self._theme_explicit = theme is not None
        # Infer theme from first pane when not given explicitly
        if theme:
            self._theme_name = theme.lower()
        elif self._panes:
            self._theme_name = self._panes[0]._theme_name
        else:
            self._theme_name = "dark"
        self._theme = THEMES.get(self._theme_name, THEMES["dark"])
        self._gap = gap
        # Push explicit theme down to all panes
        if self._theme_explicit:
            for pane in self._panes:
                pane._theme_name = self._theme_name
                pane._theme = self._theme
        self._threshold_config = None
        self._bg_image_config: Optional[Dict] = None

    @property
    def theme(self) -> str:
        return self._theme_name

    @theme.setter
    def theme(self, value: str):
        """Change the theme for the entire dashboard and all its panes."""
        self._theme_name = value.lower()
        self._theme = THEMES.get(self._theme_name, THEMES["dark"])
        self._theme_explicit = True
        for pane in self._panes:
            pane._theme_name = self._theme_name
            pane._theme = self._theme

    def add(self, pane: Chart, title: Optional[str] = None) -> "Dashboard":
        """Append a chart pane with an optional title."""
        if self._theme_explicit:
            pane._theme_name = self._theme_name
            pane._theme = self._theme
        self._panes.append(pane)
        if title:
            # Pad the titles list to match the pane index
            while len(self._titles) < len(self._panes) - 1:
                self._titles.append("")
            self._titles.append(title)
        return self

    def threshold_control(
        self,
        df: pd.DataFrame,
        df2: Optional[pd.DataFrame] = None,
        signal_mode: str = "above",
        threshold: float = 0.0,
        min_val: float = -0.05,
        max_val: float = 0.05,
        step: float = 0.001,
        value_col: Optional[str] = None,
        value_col2: Optional[str] = None,
        price_pane: int = 0,
        buy_color: Optional[str] = None,
        sell_color: Optional[str] = None,
        daily_returns: Optional[pd.Series] = None,
        bh_returns: Optional[pd.Series] = None,
        equity_pane: Optional[int] = None,
        invert: bool = False,
        threshold2: Optional[float] = None,
    ) -> "Dashboard":
        """Add an interactive threshold slider to the dashboard.

        Parameters
        ----------
        df : DataFrame with time + value columns (primary signal series).
        df2 : Optional second signal DataFrame — required for ``signal_mode="crossover"``.
        signal_mode : How the entry condition is evaluated against θ.
            Operator strings (natural syntax):
                ``">="``  — long when signal ≥ θ  (default)
                ``">"``   — long when signal > θ
                ``"<="``  — long when signal ≤ θ
                ``"<"``   — long when signal < θ
            Single-threshold named modes:
                ``"above"``     — alias for ``">="``
                ``"below"``     — alias for ``"<="``
                ``"rising"``    — long when Δsignal ≥ θ  (positive slope)
                ``"crossover"`` — long when (signal − signal2) ≥ θ  (needs *df2*)
            Band modes (two sliders, require *threshold2*):
                ``"within"``  / ``"between"`` / ``"band"`` —
                    long when θ_lo ≤ signal ≤ θ_hi
                ``"outside"`` / ``"beyond"`` —
                    long when signal < θ_lo  OR  signal > θ_hi
        threshold : lower bound (θ_lo) for band modes, or the single threshold for all other modes.
        threshold2 : upper bound (θ_hi) for band modes only.  Ignored for other modes.
            Defaults to ``threshold + 20 % of (max_val − min_val)`` when *None*.
        min_val, max_val, step : slider range and granularity.
        price_pane : pane index to add buy/sell arrows and background shading to (default 0).
        daily_returns : daily strategy returns aligned 1:1 with *df* rows.
        bh_returns : daily buy-and-hold returns for the comparison equity curve.
        equity_pane : pane index that contains the equity area series to update live.
        invert : **Deprecated** — use ``signal_mode="below"`` instead.
        """
        # Backward-compat: invert=True → signal_mode="below"
        if invert and signal_mode == "above":
            signal_mode = "below"

        def _parse_df(frame, vcol_hint):
            tc_col = Chart._detect_time_col(frame)
            frame = frame.copy()
            if tc_col == "__index__":
                frame["time"] = frame.index
            elif tc_col != "time":
                frame["time"] = frame[tc_col]
            if pd.api.types.is_datetime64_any_dtype(frame["time"]):
                frame["time"] = frame["time"].dt.strftime("%Y-%m-%d")
            else:
                frame["time"] = pd.to_datetime(frame["time"]).dt.strftime("%Y-%m-%d")
            vc = vcol_hint
            if not vc:
                for name in ("value", "close", "Close", "VALUE", "price", "Price"):
                    if name in frame.columns:
                        vc = name
                        break
            if not vc:
                for col in frame.columns:
                    if col != "time" and pd.api.types.is_numeric_dtype(frame[col]):
                        vc = col
                        break
            return (
                frame[["time", vc]]
                .rename(columns={vc: "value"})
                .dropna(subset=["value"])
                .to_dict("records")
            )

        data  = _parse_df(df, value_col)
        data2 = _parse_df(df2, value_col2) if (df2 is not None and signal_mode == "crossover") else None

        up_clr = buy_color  or self._theme.get("candlestick", {}).get("upColor",   "#26a69a")
        dn_clr = sell_color or self._theme.get("candlestick", {}).get("downColor", "#ef5350")
        decimals = max(0, -int(math.floor(math.log10(step)))) if step > 0 else 3

        ret_data = None
        bh_data  = None
        if daily_returns is not None:
            ret_series = daily_returns.fillna(0)
            n = len(data)
            vals = (ret_series.values if len(ret_series) == n
                    else ret_series.reindex(pd.RangeIndex(n)).fillna(0).values)
            times = [r["time"] for r in data]
            ret_data = [{"time": t, "ret": float(v)} for t, v in zip(times, vals)]
        if bh_returns is not None:
            bh_series = bh_returns.fillna(0)
            n = len(data)
            vals = (bh_series.values if len(bh_series) == n
                    else bh_series.reindex(pd.RangeIndex(n)).fillna(0).values)
            times = [r["time"] for r in data]
            bh_data = [{"time": t, "ret": float(v)} for t, v in zip(times, vals)]

        # Pad slider bounds slightly beyond data range for full selection
        _pad = step * 3 if step > 0 else (max_val - min_val) * 0.02
        min_val = round(min_val - _pad, decimals)
        max_val = round(max_val + _pad, decimals)

        self._threshold_config = {
            "data":        data,
            "data2":       data2,
            "signal_mode": signal_mode,
            "threshold":   threshold,
            "min_val":     min_val,
            "max_val":     max_val,
            "step":        step,
            "decimals":    decimals,
            "price_pane":  price_pane,
            "buy_color":   up_clr,
            "sell_color":  dn_clr,
            "ret_data":    ret_data,
            "bh_data":     bh_data,
            "equity_pane": equity_pane,
            "threshold2":  threshold2,
        }
        return self

    def background_image(
        self,
        url: str,
        blur: int = 0,
        tint: str = "rgba(6,6,20,0.40)",
        glass_blur: int = 16,
        glass_tint: str = "rgba(10,10,26,0.55)",
    ) -> "Dashboard":
        """Set a custom background image with a frosted-glass panel over it.

        The chart canvas becomes transparent; the image is rendered as the
        body background and the charts float as frosted-glass cards on top.

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

    @property
    def total_height(self) -> int:
        title_h = sum(24 for i in range(len(self._panes)) if self._get_title(i))
        slider_h = 36 if self._threshold_config else 0
        # Count deduplicated smoothing sliders (one row per unique label)
        _sm_labels = set()
        for pane in self._panes:
            for sc in getattr(pane, "_smoothing_configs", []):
                _sm_labels.add(sc.get("label", ""))
        slider_h += len(_sm_labels) * 36
        return sum(p._height for p in self._panes) + self._gap * max(0, len(self._panes) - 1) + title_h + slider_h

    def _get_title(self, idx: int) -> str:
        if idx < len(self._titles):
            return self._titles[idx]
        return ""

    # ── Build ─────────────────────────────────────────────────────────────

    def _build_html(self) -> str:
        if not self._panes:
            return "<html><body>No panes</body></html>"

        bg = (
            self._theme.get("chart", {})
            .get("layout", {})
            .get("background", {})
            .get("color", "#1e1e1e")
        )
        is_dark = self._theme_name in ("dark", "midnight", "distfit")
        title_color = "rgba(255,255,255,0.55)" if is_dark else "rgba(0,0,0,0.55)"
        title_font = "11px -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"

        # Logo invert for light backgrounds
        _logo_invert = ""
        if bg.startswith("#") and len(bg) >= 7:
            _bg_hex = bg.lstrip("#")
            _r, _g, _b = (int(_bg_hex[i:i+2], 16) for i in (0, 2, 4))
            _logo_invert = "filter:invert(1);" if (_r * 0.299 + _g * 0.587 + _b * 0.114) > 150 else ""
        elif self._theme.get("background_css") or self._theme.get("background_svg"):
            if self._theme_name not in ("dark", "midnight", "distfit"):
                _logo_invert = "filter:invert(1);"

        lc_js = _get_lc_js()

        # Build pane divs and per-pane JS blocks
        pane_divs = []
        pane_scripts = []
        _pane_smoothing_configs = []   # collected from child panes

        n_panes = len(self._panes)
        for i, pane in enumerate(self._panes):
            div_id = f"pane{i}"
            chart_var = f"chart{i}"
            prefix = f"p{i}_"
            is_last = (i == n_panes - 1)

            # Optional title above the pane
            title = self._get_title(i)
            if title:
                pane_divs.append(
                    f'<div style="color:{title_color};font:{title_font};'
                    f'padding:6px 0 2px 8px;letter-spacing:0.3px;'
                    f'text-transform:uppercase">{title}</div>'
                )

            # Build stats legend overlay if set on the pane
            _stats_html = ""
            if getattr(pane, "_stats_legend", None):
                _sl = pane._stats_legend
                _pos = _sl.get("position", "top-left")
                _v, _h = ("top:8px" if "top" in _pos else "bottom:8px"), ("left:8px" if "left" in _pos else "right:8px")
                _rows = "".join(
                    f'<tr><td style="color:rgba(255,255,255,0.55);padding-right:10px;white-space:nowrap">'
                    f'{_html.escape(str(k))}</td>'
                    f'<td style="color:rgba(255,255,255,0.92);font-weight:600;text-align:right">'
                    f'{_html.escape(str(v))}</td></tr>'
                    for k, v in _sl["metrics"].items()
                )
                _stats_html = (
                    f'<div onclick="var t=this.querySelector(\'table\');'
                    f't.style.display=t.style.display===\'none\'?\'table\':\'none\';'
                    f'this.querySelector(\'span\').textContent=t.style.display===\'none\'?\'▸ STATS\':\'▾ STATS\'" '
                    f'style="position:absolute;{_v};{_h};z-index:6;'
                    f'background:rgba(10,10,26,0.62);'
                    f'backdrop-filter:blur(18px) saturate(180%);-webkit-backdrop-filter:blur(18px) saturate(180%);'
                    f'border:1px solid rgba(255,255,255,0.10);'
                    f'border-radius:10px;padding:6px 12px;pointer-events:auto;cursor:pointer;user-select:none;">' 
                    f'<div style="color:rgba(255,255,255,0.4);font:9px/1.8 \'SF Mono\',monospace;'
                    f'text-align:right;letter-spacing:0.5px"><span>▾ STATS</span></div>'
                    f'<table style="border-collapse:collapse;font:11px/1.6 \'SF Mono\',monospace">'
                    f'{_rows}</table></div>'
                )
            pane_divs.append(
                f'<div id="{div_id}" style="position:relative;width:100%;height:{pane._height}px;'
                f'margin-bottom:{self._gap}px">{_stats_html}</div>'
            )

            # Build chart options — hide time axis labels on non-bottom panes
            chart_opts = pane._build_chart_options()
            # Dashboard panes use fixed container divs, not autoSize
            chart_opts.pop("autoSize", None)
            # If dashboard has a bg image, make all pane canvases transparent
            if self._bg_image_config:
                lo = chart_opts.get("layout", {})
                lo["background"] = {"type": "solid", "color": "rgba(0,0,0,0)"}
                chart_opts["layout"] = lo
            if not is_last:
                ts = chart_opts.get("timeScale", {})
                ts["visible"] = False
                chart_opts["timeScale"] = ts
            else:
                ts = chart_opts.get("timeScale", {})
                ts["visible"] = True
                chart_opts["timeScale"] = ts
            opts = json.dumps(chart_opts, separators=(",", ":"))
            series_js = pane._build_series_js(var_prefix=prefix, chart_var=chart_var)

            # Track the first series reference for crosshair sync
            first_var = f"{prefix}s0" if pane._series else "null"
            pane_scripts.append(
                f"const {chart_var} = LightweightCharts.createChart("
                f"document.getElementById('{div_id}'), {opts});\n"
                f"    charts.push({chart_var});\n"
                f"    {series_js}\n"
                f"    firstSeries.push({first_var});"
            )

            # ── Collect smoothing configs from this pane (with namespaced var names) ──
            n_pane_series = len(pane._series)
            for sc_idx, sc in enumerate(getattr(pane, "_smoothing_configs", [])):
                target_idx = sc["series_index"] if sc["series_index"] >= 0 else n_pane_series + sc["series_index"]
                _pane_smoothing_configs.append({**sc, "_svar": f"{prefix}s{target_idx}", "_pane": i, "_local_idx": sc_idx})

        divs_html = "\n".join(pane_divs)
        scripts_body = "\n    ".join(pane_scripts)

        # ── Threshold slider (dashboard-level) ────────────────────────────
        slider_html = ""
        slider_js = ""
        if self._threshold_config:
            tc = self._threshold_config
            dec = tc["decimals"]
            lbl_c = "rgba(255,255,255,0.88)" if is_dark else "rgba(0,0,0,0.78)"
            cnt_c = "rgba(255,255,255,0.48)" if is_dark else "rgba(0,0,0,0.42)"
            thr0 = tc["threshold"]
            pp = tc["price_pane"]
            price_chart = f"chart{pp}"
            price_s0 = f"p{pp}_s0"
            tdata_json = json.dumps(tc["data"], separators=(",", ":"))

            bc = tc["buy_color"].lstrip("#")
            sr, sg, sb = int(bc[0:2], 16), int(bc[2:4], 16), int(bc[4:6], 16)
            shade_fill = f"rgba({sr},{sg},{sb},0.18)"
            shade_line = "transparent"
            dc = tc["sell_color"].lstrip("#")
            sell_r, sell_g, sell_b = int(dc[0:2], 16), int(dc[2:4], 16), int(dc[4:6], 16)

            # ── Signal mode — condition strings and JS precompute snippet ───
            sm = tc.get("signal_mode", "above")

            # Normalise operator aliases → canonical names
            _op_alias = {
                ">=": "above", ">": "above_strict", "<=": "below", "<": "below_strict",
                "between": "within", "band": "within",
                "beyond": "outside",
            }
            sm = _op_alias.get(sm, sm)

            is_band_mode = False
            if sm == "rising":
                long_idx_cond  = "_delta[i] >= thr"
                short_idx_cond = "false"
                long_pos_cond  = "_delta[i] >= thr"
                short_pos_cond = "false"
                eq_pos_expr    = "_delta[i] >= thr ? 1 : 0"
                precompute_js  = "    const _delta = _td.map((d,i) => i===0 ? 0 : d.value - _td[i-1].value);"
                update_baseline = True
                invert_colors  = False
                show_neg_line  = False
                thr_lbl_pfx    = "\\u0394\\u00a0\\u2265\\u00a0"
                thr_lbl_init   = f"\u0394\u00a0\u2265\u00a0{thr0:.{dec}f}"
            elif sm == "crossover":
                _data2_json    = json.dumps(tc.get("data2") or [], separators=(",", ":"))
                long_idx_cond  = "_spread[i] >= thr"
                short_idx_cond = "false"
                long_pos_cond  = "_spread[i] >= thr"
                short_pos_cond = "false"
                eq_pos_expr    = "_spread[i] >= thr ? 1 : 0"
                precompute_js  = (
                    f"    const _td2 = {_data2_json};\n"
                    f"    const _spread = _td.map((d,i) => d.value - (_td2[i] ? _td2[i].value : 0));"
                )
                update_baseline = True
                invert_colors  = False
                show_neg_line  = False
                thr_lbl_pfx    = "spread\\u00a0\\u2265\\u00a0"
                thr_lbl_init   = f"spread\u00a0\u2265\u00a0{thr0:.{dec}f}"
            elif sm in ("below", "below_strict"):
                _op_sym = "<=" if sm == "below" else "<"
                _js_op  = "<="  if sm == "below" else "<"
                long_idx_cond  = f"_td[i].value {_js_op} thr"
                short_idx_cond = "false"
                long_pos_cond  = f"sig {_js_op} thr"
                short_pos_cond = "false"
                eq_pos_expr    = f"sig {_js_op} thr ? 1 : 0"
                precompute_js  = ""
                update_baseline = True
                invert_colors  = True
                show_neg_line  = False
                thr_lbl_pfx    = f"\\u03b8\\u00a0{_op_sym}\\u00a0"
                thr_lbl_init   = f"\u03b8\u00a0{_op_sym}\u00a0{thr0:.{dec}f}"
            elif sm == "above_strict":
                long_idx_cond  = "_td[i].value > thr"
                short_idx_cond = "false"
                long_pos_cond  = "sig > thr"
                short_pos_cond = "false"
                eq_pos_expr    = "sig > thr ? 1 : 0"
                precompute_js  = ""
                update_baseline = True
                invert_colors  = False
                show_neg_line  = False
                thr_lbl_pfx    = "\\u03b8\\u00a0>\\u00a0"
                thr_lbl_init   = f"\u03b8\u00a0>\u00a0{thr0:.{dec}f}"
            elif sm == "within":
                long_idx_cond  = "_td[i].value >= thr"   # dummy — band JS overrides
                short_idx_cond = "false"
                long_pos_cond  = "sig >= lo && sig <= hi"
                short_pos_cond = "false"
                eq_pos_expr    = "(sig >= lo && sig <= hi) ? 1 : 0"
                precompute_js  = ""
                update_baseline = False
                invert_colors  = False
                show_neg_line  = False
                is_band_mode   = True
                thr_lbl_pfx    = "\\u03b8_lo"          # unused for band
                thr_lbl_init   = f"\u03b8_lo\u00a0{thr0:.{dec}f}"
            elif sm == "outside":
                long_idx_cond  = "_td[i].value < thr"    # dummy — band JS overrides
                short_idx_cond = "false"
                long_pos_cond  = "sig < lo || sig > hi"
                short_pos_cond = "false"
                eq_pos_expr    = "(sig < lo || sig > hi) ? 1 : 0"
                precompute_js  = ""
                update_baseline = False
                invert_colors  = False
                show_neg_line  = False
                is_band_mode   = True
                thr_lbl_pfx    = "\\u03b8_lo"          # unused for band
                thr_lbl_init   = f"\u03b8_lo\u00a0{thr0:.{dec}f}"
            else:  # "above" / ">=" (default)
                long_idx_cond  = "_td[i].value >= thr"
                short_idx_cond = "false"
                long_pos_cond  = "sig >= thr"
                short_pos_cond = "false"
                eq_pos_expr    = "sig >= thr ? 1 : 0"
                precompute_js  = ""
                update_baseline = True
                invert_colors  = False
                show_neg_line  = False
                thr_lbl_pfx    = "\\u03b8\\u00a0\\u2265\\u00a0"
                thr_lbl_init   = f"\u03b8\u00a0\u2265\u00a0{thr0:.{dec}f}"

            # Pre-resolved baseline colors (used only when update_baseline=True)
            top_line_color = "rgba(120,120,120,0.4)"  if invert_colors else tc["buy_color"]
            top_fill1      = "rgba(100,100,100,0.06)" if invert_colors else f"rgba({sr},{sg},{sb},0.28)"
            top_fill2      = "rgba(100,100,100,0.01)" if invert_colors else f"rgba({sr},{sg},{sb},0.05)"
            bot_line_color = tc["buy_color"]           if invert_colors else "rgba(100,100,100,0.28)"
            bot_fill1      = f"rgba({sr},{sg},{sb},0.08)" if invert_colors else "rgba(100,100,100,0.01)"
            bot_fill2      = "transparent"                  if invert_colors else "rgba(100,100,100,0.06)"

            slider_html = (
                f'<div id="th-bar" style="display:flex;align-items:center;justify-content:center;'
                f'gap:10px;background:transparent;padding:6px 16px;white-space:nowrap;height:36px">'
                f'<span id="th-label" style="color:{lbl_c};'
                f"font:11px/1 'SF Mono','Consolas',monospace;min-width:72px\">"
                f'{thr_lbl_init}</span>'
                f'<input id="th-slider" type="range" '
                f'min="{tc["min_val"]}" max="{tc["max_val"]}" step="{tc["step"]}" '
                f'value="{thr0}" '
                f'style="width:220px;cursor:pointer;accent-color:{tc["buy_color"]}">'
                f'<span id="th-count" style="color:{cnt_c};font:11px/1 sans-serif;'
                f'min-width:60px;text-align:right"></span>'
                f'</div>'
            )
            if not is_band_mode:
                divs_html += "\n" + slider_html

            slider_js = "\n".join([
                "",
                "    // ── Dashboard threshold slider ───────────────────────────────",
                f"    const _td = {tdata_json};",
                f'    const _tBuy = "{tc["buy_color"]}";',
                f'    const _tSell = "{tc["sell_color"]}";',
                f"    const _tDec = {dec};",
                f'    const _thrLblPfx = "{thr_lbl_pfx}";',
                "",
            ] + ([precompute_js, ""] if precompute_js else []) + [
                "    // Long shade (green) on price pane; short shade (red) when signal <= -thr",
                f"    const _shadeSeries = {price_chart}.addSeries(LightweightCharts.AreaSeries, {{",
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
                f'    {price_chart}.priceScale("_thShade").applyOptions({{visible:false,scaleMargins:{{top:0,bottom:0}}}});',
                f"    const _shadeSellSeries = {price_chart}.addSeries(LightweightCharts.AreaSeries, {{",
                '        priceScaleId: "_thShadeSell",',
                "        lineWidth: 1,",
                '        lineColor: "transparent",',
                "        lineType: 2,",
                '        topColor: "transparent",',
                '        bottomColor: "transparent",',
                "        crosshairMarkerVisible: false,",
                "        pointMarkersVisible: false,",
                "        lastValueVisible: false,",
                "        priceLineVisible: false,",
                "    });",
                f'    {price_chart}.priceScale("_thShadeSell").applyOptions({{visible:false,scaleMargins:{{top:0,bottom:0}}}});',
                "",
                "    // Signal line pane — colour by threshold position (above/below modes only)",
                "    const _thresholdLines = [];",
                "    const _thresholdNegLines = [];",
                "    const _sigSeries = [];",
                "    for (let ci = 0; ci < charts.length; ci++) {",
                f"        if (ci === {pp}) continue;",
                f"        if (ci === {tc['equity_pane'] if tc.get('equity_pane') is not None else -1}) continue;",
                "        const fs = firstSeries[ci];",
                "        if (fs) {",
                "            try {",
            ] + ([] if not update_baseline else ([
                "                // Override baseline: in-market side = colored, out = grey",
                "                fs.applyOptions({",
                f'                    baseValue: {{type:"price",price:{thr0}}},',
                f'                    topLineColor: "{top_line_color}",',
                f'                    topFillColor1: "{top_fill1}",',
                f'                    topFillColor2: "{top_fill2}",',
                f'                    bottomLineColor: "{bot_line_color}",',
                f'                    bottomFillColor1: "{bot_fill1}",',
                f'                    bottomFillColor2: "{bot_fill2}",',
                "                });",
                "                _sigSeries.push(fs);",
                f'                const pl = fs.createPriceLine({{price:{thr0},title:"\\u03b8",color:"{tc["buy_color"]}",lineWidth:1,lineStyle:2,axisLabelVisible:true}});',
                "                _thresholdLines.push(pl);",
            ] + ([] if not show_neg_line else [
                f'                const plN = fs.createPriceLine({{price:{-thr0},title:"-\\u03b8",color:"{tc["sell_color"]}",lineWidth:1,lineStyle:2,axisLabelVisible:true}});',
                "                _thresholdNegLines.push(plN);",
            ]))) + [
                "            } catch(e) {}",
                "        }",
                "    }",
                "",
                "    function _buildShade(thr) {",
                "        const d = [];",
                "        for (let i = 0; i < _td.length; i++) {",
                f"            const above = {long_idx_cond};",
                "            d.push({time: _td[i].time, value: above ? 1 : 0});",
                "        }",
                "        return d;",
                "    }",
                "    function _buildShadeSell(thr) {",
                "        const d = [];",
                "        for (let i = 0; i < _td.length; i++) {",
                f"            const below = {short_idx_cond};",
                "            d.push({time: _td[i].time, value: below ? 1 : 0});",
                "        }",
                "        return d;",
                "    }",
                "",
                "    function _mkrs(thr) {",
                "        const m = []; let longOn=false, shortOn=false;",
                "        for (let i = 0; i < _td.length; i++) {",
                "            const sig = _td[i].value;",
                f"            const goLong  = {long_pos_cond};",
                f"            const goShort = {short_pos_cond};",
                "            if (goLong && !longOn) {",
                '                m.push({time:_td[i].time,position:"belowBar",shape:"arrowUp",color:_tBuy,text:""});',
                "                longOn=true; if(shortOn) shortOn=false;",
                "            } else if (!goLong && longOn) {",
                '                m.push({time:_td[i].time,position:"aboveBar",shape:"arrowDown",color:_tSell,text:""});',
                "                longOn=false;",
                "            }",
                "            if (!goLong && goShort && !shortOn) {",
                '                m.push({time:_td[i].time,position:"aboveBar",shape:"arrowDown",color:_tSell,text:"S"});',
                "                shortOn=true;",
                "            } else if (!goShort && shortOn) {",
                '                m.push({time:_td[i].time,position:"belowBar",shape:"circle",color:"rgba(150,150,150,0.8)",text:"C"});',
                "                shortOn=false;",
                "            }",
                "        }",
                "        return m;",
                "    }",
                "",
                f"    const _thP = LightweightCharts.createSeriesMarkers({price_s0}, []);  // markers hidden",
                f"    _shadeSeries.setData(_buildShade({thr0}));",
                f"    _shadeSellSeries.setData(_buildShadeSell({thr0}));",
                "",
            ] + ([] if not tc.get("ret_data") else [
                "    // ── Live equity + stats ──────────────────────────────────────",
                f"    const _rets = {json.dumps(tc['ret_data'], separators=(',', ':'))};",
                f"    const _bhRets = {json.dumps(tc.get('bh_data') or [], separators=(',', ':'))};",
                f"    const _eqPaneIdx = {tc['equity_pane'] if tc.get('equity_pane') is not None else -1};",
                "    const _eqSeries  = _eqPaneIdx >= 0 ? firstSeries[_eqPaneIdx] : null;",
                "    // second series in equity pane = b&h line (index 1)",
                f"    const _bhSeries  = _eqPaneIdx >= 0 ? (typeof p{tc['equity_pane'] if tc.get('equity_pane') is not None else 0}_s1 !== 'undefined' ? p{tc['equity_pane'] if tc.get('equity_pane') is not None else 0}_s1 : null) : null;",
                "",
                "    // Stats overlay element — injected into equity pane's container div",
                "    let _statsEl = null;",
                "    let _statsTblEl = null;",
                "    let _statsCollapsed = false;",
                "    if (_eqPaneIdx >= 0) {",
                f"        const _epDiv = document.getElementById('pane' + _eqPaneIdx);",
                "        _statsEl = document.createElement('div');",
                f"        _statsEl.style.cssText = 'position:absolute;top:8px;left:8px;z-index:6;"
                f"background:{'rgba(10,10,26,0.62)' if is_dark else 'rgba(255,255,255,0.68)'};"
                f"backdrop-filter:blur(18px) saturate(180%);-webkit-backdrop-filter:blur(18px) saturate(180%);"
                f"border:1px solid rgba(255,255,255,0.10);border-radius:10px;padding:6px 12px;"
                f"pointer-events:auto;cursor:pointer;user-select:none;"
                f"font:11px/1.6 \\'SF Mono\\',\\'Consolas\\',monospace;';",
                f"        const _statsHdr = document.createElement('div');",
                f"        _statsHdr.style.cssText = 'color:rgba(255,255,255,0.4);font:9px/1.8 \\'SF Mono\\',monospace;text-align:right;letter-spacing:0.5px';",
                f"        _statsHdr.textContent = '\\u25be STATS';",
                "        _statsTblEl = document.createElement('table');",
                "        _statsTblEl.style.cssText = 'border-collapse:collapse';",
                "        _statsEl.appendChild(_statsHdr);",
                "        _statsEl.appendChild(_statsTblEl);",
                "        _statsEl.addEventListener('click', function() {",
                "            _statsCollapsed = !_statsCollapsed;",
                "            _statsTblEl.style.display = _statsCollapsed ? 'none' : 'table';",
                "            _statsHdr.textContent = _statsCollapsed ? '\\u25b8 STATS' : '\\u25be STATS';",
                "        });",
                "        if (_epDiv) { _epDiv.style.position='relative'; _epDiv.appendChild(_statsEl); }",
                "    }",
                "",
                "    function _buildEquity(thr) {",
                "        let pos = 0, eq = 100, bh = 100;",
                "        const eqData = [], bhData = [];",
                "        let n=0,sumR=0,sumR2=0,peak=100,maxDD=0;",
                "        let wins=0,tradesN=0,nLong=0,nShort=0;",
                "        const hasBh = _bhRets.length === _rets.length;",
                "        for (let i = 0; i < _rets.length; i++) {",
                "            const sig = _td[i] ? _td[i].value : 0;",
                f"            pos = {eq_pos_expr};",
                "            const r = _rets[i].ret * pos;",
                "            eq *= (1 + r);",
                "            if (hasBh) bh *= (1 + _bhRets[i].ret);",
                "            eqData.push({time: _rets[i].time, value: eq});",
                "            if (hasBh) bhData.push({time: _bhRets[i].time, value: bh});",
                "            if (pos !== 0) { sumR += r; sumR2 += r*r; n++; if(r>0) wins++; tradesN++; }",
                "            if (pos === 1) nLong++; else if (pos === -1) nShort++;",
                "            if (eq > peak) peak = eq;",
                "            const dd = (eq - peak) / peak;",
                "            if (dd < maxDD) maxDD = dd;",
                "        }",
                "        const annR = n > 0 ? sumR / _rets.length * 252 : 0;",
                "        const annV = n > 0 ? Math.sqrt((sumR2/_rets.length - Math.pow(sumR/_rets.length,2)) * 252) : 1;",
                "        const sharpe = annV > 0 ? annR / annV : 0;",
                "        const wr = tradesN > 0 ? wins/tradesN : 0;",
                "        const timeMkt = n / _rets.length;",
                "        const longPct = nLong / _rets.length;",
                "        const shortPct = nShort / _rets.length;",
                "        // Re-index to 100",
                "        const s = 100 / eqData[0].value;",
                "        eqData.forEach(d => d.value *= s);",
                "        if (hasBh) { const bs = 100/bhData[0].value; bhData.forEach(d => d.value *= bs); }",
                "        const totalRet = eqData[eqData.length-1].value / 100 - 1;",
                "        const bhRet = hasBh ? bhData[bhData.length-1].value / 100 - 1 : null;",
                "        const nYears = _rets.length / 252;",
                "        const cagr = Math.pow(eqData[eqData.length-1].value/100, 1/nYears) - 1;",
                "        return {eqData, bhData, totalRet, bhRet, cagr, sharpe, maxDD, wr, timeMkt, longPct, shortPct};",
                "    }",
                "",
                "    function _fmtPct(v, showSign=true) {",
                "        return (showSign && v>0?'+':'') + (v*100).toFixed(1)+'%';",
                "    }",
                "",
                "    function _updateStats(res) {",
                "        if (!_statsEl || !_statsTblEl) return;",
                f"        const lbl = 'color:{'rgba(255,255,255,0.55)' if is_dark else 'rgba(0,0,0,0.45)'}';",
                f"        const val = 'color:{'rgba(255,255,255,0.92)' if is_dark else 'rgba(0,0,0,0.88)'};font-weight:600';",
                "        const rows = [",
                "            ['Return', _fmtPct(res.totalRet)],",
                "            res.bhRet !== null ? ['B&H', _fmtPct(res.bhRet)] : null,",
                "            ['CAGR',   _fmtPct(res.cagr)],",
                "            ['Sharpe', res.sharpe.toFixed(2)],",
                "            ['Max DD', _fmtPct(res.maxDD, false)],",
                "            ['Win Rate', _fmtPct(res.wr, false)],",
                "            ['In Mkt',  _fmtPct(res.timeMkt, false)],",
                "            res.shortPct > 0 ? ['  Long', _fmtPct(res.longPct, false)] : null,",
                "            res.shortPct > 0 ? ['  Short', _fmtPct(res.shortPct, false)] : null,",
                "        ].filter(Boolean);",
                "        _statsTblEl.innerHTML = rows.map(([k,v]) => `<tr><td style=\"${lbl};padding:1px 8px 1px 0;white-space:nowrap\">${k}</td><td style=\"${val};text-align:right\">${v}</td></tr>`).join('');",
                "    }",
                "",
                f"    const _eq0 = _buildEquity({thr0});",
                "    if (_eqSeries) _eqSeries.setData(_eq0.eqData);",
                "    if (_bhSeries && _eq0.bhData.length) _bhSeries.setData(_eq0.bhData);",
                "    _updateStats(_eq0);",
            ]) + [
                "    (function() {",
                f"        const m0 = _mkrs({thr0});",
                '        document.getElementById("th-count").textContent = m0.filter(x=>x.shape==="arrowUp").length + " signals";',
                "    })();",
                '    document.getElementById("th-slider").addEventListener("input", function() {',
                "        const thr = parseFloat(this.value);",
                '        document.getElementById("th-label").textContent = _thrLblPfx + thr.toFixed(_tDec);',
                "        const m = _mkrs(thr);",
                "        _thP.setMarkers([]);  // arrows hidden",
                "        _shadeSeries.setData(_buildShade(thr));",
                "        _shadeSellSeries.setData(_buildShadeSell(thr));",
                "        for (const pl of _thresholdLines) { pl.applyOptions({price: thr}); }",
                "        for (const pl of _thresholdNegLines) { pl.applyOptions({price: -thr}); }",
            ] + ([
                '        for (const ss of _sigSeries) { ss.applyOptions({baseValue:{type:"price",price:thr}}); }',
            ] if update_baseline else []) + [
                '        document.getElementById("th-count").textContent = m.filter(x=>x.shape==="arrowUp").length + " signals";',
                "        if (_rets && _rets.length) {",
                "            const eq = _buildEquity(thr);",
                "            if (_eqSeries) _eqSeries.setData(eq.eqData);",
                "            if (_bhSeries && eq.bhData.length) _bhSeries.setData(eq.bhData);",
                "            _updateStats(eq);",
                "            setTimeout(_alignScales, 50);",
                "        }",
                "    });",
            ])

            # ── Band mode override (two-slider HTML + JS) ─────────────────────
            if is_band_mode:
                _thr2 = tc.get("threshold2")
                if _thr2 is None:
                    _thr2 = round(thr0 + (tc["max_val"] - tc["min_val"]) * 0.2, dec)
                _thr2 = max(tc["min_val"], min(tc["max_val"], _thr2))
                _band_is_outside  = (sm == "outside")
                _band_shade_cond  = "v < lo || v > hi"  if _band_is_outside else "v >= lo && v <= hi"
                _band_pos_cond    = "sig < lo || sig > hi" if _band_is_outside else "sig >= lo && sig <= hi"
                _band_eq_expr     = "(sig < lo || sig > hi) ? 1 : 0" if _band_is_outside else "(sig >= lo && sig <= hi) ? 1 : 0"
                _lo_lbl           = f"\u03b8_lo\u00a0{thr0:.{dec}f}"
                _hi_lbl           = f"\u03b8_hi\u00a0{_thr2:.{dec}f}"
                # Custom series plugin: renders green/grey segments via Canvas2D
                divs_html += "\n" + (
                    f'<div id="th-bar" style="display:flex;align-items:center;justify-content:center;'
                    f'gap:8px;background:transparent;padding:6px 16px;white-space:nowrap;height:36px">'
                    f'<span id="th-lo-label" style="color:{lbl_c};'
                    f"font:11px/1 'SF Mono','Consolas',monospace;min-width:76px\">"
                    f'{_lo_lbl}</span>'
                    f'<input id="th-lo-slider" type="range" '
                    f'min="{tc["min_val"]}" max="{tc["max_val"]}" step="{tc["step"]}" '
                    f'value="{thr0}" '
                    f'style="width:155px;cursor:pointer;accent-color:{tc["buy_color"]}">'
                    f'<span id="th-hi-label" style="color:{lbl_c};'
                    f"font:11px/1 'SF Mono','Consolas',monospace;min-width:76px\">"
                    f'{_hi_lbl}</span>'
                    f'<input id="th-hi-slider" type="range" '
                    f'min="{tc["min_val"]}" max="{tc["max_val"]}" step="{tc["step"]}" '
                    f'value="{_thr2}" '
                    f'style="width:155px;cursor:pointer;accent-color:{tc["buy_color"]}">'
                    f'<span id="th-count" style="color:{cnt_c};font:11px/1 sans-serif;'
                    f'min-width:60px;text-align:right"></span>'
                    f'</div>'
                )
                slider_js = "\n".join([
                    "",
                    "    // \u2500\u2500 Dashboard band-threshold slider (two sliders) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500",
                    f"    const _td = {tdata_json};",
                    f'    const _tBuy = "{tc["buy_color"]}";',
                    f'    const _tSell = "{tc["sell_color"]}";',
                    f"    const _tDec = {dec};",
                    "",
                    f"    const _shadeSeries = {price_chart}.addSeries(LightweightCharts.AreaSeries, {{",
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
                    f'    {price_chart}.priceScale("_thShade").applyOptions({{visible:false,scaleMargins:{{top:0,bottom:0}}}});',
                    "",
                    "    // ── MaskedArea custom series plugin ──────────────────────",
                    "    function _createMaskedAreaPlugin() {",
                    "        let _d = null, _o = {};",
                    "        return {",
                    "            defaultOptions() { return {",
                    f"                activeLineColor: _tBuy,",
                    f"                activeTopColor: 'rgba(38,166,154,0.28)',",
                    f"                activeBottomColor: 'rgba(38,166,154,0.05)',",
                    f"                inactiveLineColor: 'rgba(150,150,150,0.55)',",
                    f"                inactiveTopColor: 'rgba(150,150,150,0.18)',",
                    f"                inactiveBottomColor: 'rgba(150,150,150,0.02)',",
                    "                lineWidth: 3,",
                    "            }; },",
                    "            renderer() {",
                    "                return {",
                    "                    draw(target, priceConverter) {",
                    "                        target.useBitmapCoordinateSpace(({context: ctx, bitmapSize, horizontalPixelRatio: hpx, verticalPixelRatio: vpx}) => {",
                    "                            if (!_d || !_d.bars.length) return;",
                    "                            const o = _o;",
                    "                            const pts = [];",
                    "                            for (const bar of _d.bars) {",
                    "                                const od = bar.originalData;",
                    "                                if (od.value == null) continue;",
                    "                                const y = priceConverter(od.value);",
                    "                                if (y == null) continue;",
                    "                                pts.push({x: Math.round(bar.x * hpx), y: Math.round(y * vpx), a: od.active !== false});",
                    "                            }",
                    "                            if (!pts.length) return;",
                    "                            const segs = []; let cur = {a: pts[0].a, p: [pts[0]]};",
                    "                            for (let i = 1; i < pts.length; i++) {",
                    "                                if (pts[i].a !== cur.a) { cur.p.push(pts[i]); segs.push(cur); cur = {a: pts[i].a, p: [pts[i]]}; }",
                    "                                else { cur.p.push(pts[i]); }",
                    "                            }",
                    "                            segs.push(cur);",
                    "                            const bottom = bitmapSize.height;",
                    "                            for (const s of segs) {",
                    "                                if (s.p.length < 2) continue;",
                    "                                const tC = s.a ? o.activeTopColor : o.inactiveTopColor;",
                    "                                const bC = s.a ? o.activeBottomColor : o.inactiveBottomColor;",
                    "                                const lC = s.a ? o.activeLineColor : o.inactiveLineColor;",
                    "                                const gr = ctx.createLinearGradient(0, 0, 0, bottom);",
                    "                                gr.addColorStop(0, tC); gr.addColorStop(1, bC);",
                    "                                ctx.beginPath();",
                    "                                ctx.moveTo(s.p[0].x, s.p[0].y);",
                    "                                for (let j = 1; j < s.p.length; j++) ctx.lineTo(s.p[j].x, s.p[j].y);",
                    "                                ctx.lineTo(s.p[s.p.length-1].x, bottom); ctx.lineTo(s.p[0].x, bottom);",
                    "                                ctx.closePath(); ctx.fillStyle = gr; ctx.fill();",
                    "                                ctx.beginPath();",
                    "                                ctx.moveTo(s.p[0].x, s.p[0].y);",
                    "                                for (let j = 1; j < s.p.length; j++) ctx.lineTo(s.p[j].x, s.p[j].y);",
                    "                                ctx.strokeStyle = lC; ctx.lineWidth = Math.max(1, (o.lineWidth || 2) * hpx); ctx.stroke();",
                    "                            }",
                    "                        });",
                    "                    }",
                    "                };",
                    "            },",
                    "            priceValueBuilder(d) { return [d.value]; },",
                    "            isWhitespace(d) { return d.value == null; },",
                    "            update(data, options) { _d = data; _o = options; }",
                    "        };",
                    "    }",
                    "",
                    "    const _thresholdLoLines = [];",
                    "    const _thresholdHiLines = [];",
                    "    const _maskedSeries = [];",
                    f"    for (let ci = 0; ci < charts.length; ci++) {{",
                    f"        if (ci === {pp}) continue;",
                    f"        if (ci === {tc['equity_pane'] if tc.get('equity_pane') is not None else -1}) continue;",
                    "        const fs = firstSeries[ci];",
                    "        if (fs) {",
                    "            fs.applyOptions({",
                    '                topLineColor: "rgba(0,0,0,0)", bottomLineColor: "rgba(0,0,0,0)",',
                    '                topFillColor1: "rgba(0,0,0,0)", topFillColor2: "rgba(0,0,0,0)",',
                    '                bottomFillColor1: "rgba(0,0,0,0)", bottomFillColor2: "rgba(0,0,0,0)",',
                    "                crosshairMarkerVisible: true, crosshairMarkerRadius: 5,",
                    "                crosshairMarkerBorderWidth: 2,",
                    "                crosshairMarkerBackgroundColor: _tBuy,",
                    "            });",
                    f'            const plo = fs.createPriceLine({{price:{thr0},title:"\\u03b8_lo",color:"#ef5350",lineWidth:1,lineStyle:2,axisLabelVisible:true}});',
                    f'            const phi = fs.createPriceLine({{price:{_thr2},title:"\\u03b8_hi",color:"{tc["buy_color"]}",lineWidth:1,lineStyle:2,axisLabelVisible:true}});',
                    "            _thresholdLoLines.push(plo);",
                    "            _thresholdHiLines.push(phi);",
                    "            const ms = charts[ci].addCustomSeries(_createMaskedAreaPlugin(), {",
                    "                crosshairMarkerVisible: false,",
                    "            });",
                    "            _maskedSeries.push(ms);",
                    "        }",
                    "    }",
                    "",
                    "    function _buildMask(lo, hi) {",
                    "        return _td.map(d => {",
                    f"            const v = d.value; const active = ({_band_shade_cond});",
                    "            return {time: d.time, value: v, active: active};",
                    "        });",
                    "    }",
                    f"    const _msk0 = _buildMask({thr0}, {_thr2});",
                    "    for (const s of _maskedSeries) s.setData(_msk0);",
                    "",
                    "    // Build time→active lookup for crosshair marker coloring",
                    "    let _activeMap = {};",
                    "    _msk0.forEach(d => { _activeMap[d.time] = d.active; });",
                    "",
                    "    // Collect signal-pane baseline series for marker recoloring",
                    "    const _sigBaselines = [];",
                    f"    for (let ci = 0; ci < charts.length; ci++) {{",
                    f"        if (ci === {pp}) continue;",
                    f"        if (ci === {tc['equity_pane'] if tc.get('equity_pane') is not None else -1}) continue;",
                    "        const fs = firstSeries[ci];",
                    "        if (fs) {",
                    "            _sigBaselines.push(fs);",
                    "            charts[ci].subscribeCrosshairMove(param => {",
                    "                if (!param.time) return;",
                    "                const isActive = _activeMap[param.time];",
                    "                const clr = isActive ? _tBuy : 'rgba(150,150,150,0.7)';",
                    "                fs.applyOptions({ crosshairMarkerBackgroundColor: clr });",
                    "            });",
                    "        }",
                    "    }",
                    "",
                    "    function _buildShade(lo, hi) {",
                    "        const d = [];",
                    "        for (let i = 0; i < _td.length; i++) {",
                    f"            const v = _td[i].value; const cond = {_band_shade_cond};",
                    "            d.push({time: _td[i].time, value: cond ? 1 : 0});",
                    "        }",
                    "        return d;",
                    "    }",
                    "",
                    "    function _mkrs(lo, hi) {",
                    "        const m = []; let longOn = false;",
                    "        for (let i = 0; i < _td.length; i++) {",
                    "            const sig = _td[i].value;",
                    f"            const goLong = {_band_pos_cond};",
                    "            if (goLong && !longOn) {",
                    '                m.push({time:_td[i].time,position:"belowBar",shape:"arrowUp",color:_tBuy,text:""});',
                    "                longOn=true;",
                    "            } else if (!goLong && longOn) {",
                    '                m.push({time:_td[i].time,position:"aboveBar",shape:"arrowDown",color:_tSell,text:""});',
                    "                longOn=false;",
                    "            }",
                    "        }",
                    "        return m;",
                    "    }",
                    "",
                ] + ([] if not tc.get("ret_data") else [
                    "    // \u2500\u2500 Live equity + stats \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500",
                    f"    const _rets = {json.dumps(tc['ret_data'], separators=(',', ':'))};",
                    f"    const _bhRets = {json.dumps(tc.get('bh_data') or [], separators=(',', ':'))};",
                    f"    const _eqPaneIdx = {tc['equity_pane'] if tc.get('equity_pane') is not None else -1};",
                    "    const _eqSeries  = _eqPaneIdx >= 0 ? firstSeries[_eqPaneIdx] : null;",
                    f"    const _bhSeries  = _eqPaneIdx >= 0 ? (typeof p{tc['equity_pane'] if tc.get('equity_pane') is not None else 0}_s1 !== 'undefined' ? p{tc['equity_pane'] if tc.get('equity_pane') is not None else 0}_s1 : null) : null;",
                    "",
                    "    let _statsEl = null;",
                    "    let _statsTblEl = null;",
                    "    let _statsCollapsed = false;",
                    "    if (_eqPaneIdx >= 0) {",
                    f"        const _epDiv = document.getElementById('pane' + _eqPaneIdx);",
                    "        _statsEl = document.createElement('div');",
                    f"        _statsEl.style.cssText = 'position:absolute;top:8px;left:8px;z-index:6;"
                    f"background:{'rgba(10,10,26,0.62)' if is_dark else 'rgba(255,255,255,0.68)'};"
                    f"backdrop-filter:blur(18px) saturate(180%);-webkit-backdrop-filter:blur(18px) saturate(180%);"
                    f"border:1px solid rgba(255,255,255,0.10);border-radius:10px;padding:6px 12px;"
                    f"pointer-events:auto;cursor:pointer;user-select:none;"
                    f"font:11px/1.6 \\'SF Mono\\',\\'Consolas\\',monospace;';",
                    f"        const _statsHdr = document.createElement('div');",
                    f"        _statsHdr.style.cssText = 'color:rgba(255,255,255,0.4);font:9px/1.8 \\'SF Mono\\',monospace;text-align:right;letter-spacing:0.5px';",
                    f"        _statsHdr.textContent = '\\u25be STATS';",
                    "        _statsTblEl = document.createElement('table');",
                    "        _statsTblEl.style.cssText = 'border-collapse:collapse';",
                    "        _statsEl.appendChild(_statsHdr);",
                    "        _statsEl.appendChild(_statsTblEl);",
                    "        _statsEl.addEventListener('click', function() {",
                    "            _statsCollapsed = !_statsCollapsed;",
                    "            _statsTblEl.style.display = _statsCollapsed ? 'none' : 'table';",
                    "            _statsHdr.textContent = _statsCollapsed ? '\\u25b8 STATS' : '\\u25be STATS';",
                    "        });",
                    "        if (_epDiv) { _epDiv.style.position='relative'; _epDiv.appendChild(_statsEl); }",
                    "    }",
                    "",
                    "    function _buildEquity(lo, hi) {",
                    "        let pos = 0, eq = 100, bh = 100;",
                    "        const eqData = [], bhData = [];",
                    "        let n=0,sumR=0,sumR2=0,peak=100,maxDD=0,wins=0,tradesN=0,nLong=0;",
                    "        const hasBh = _bhRets.length === _rets.length;",
                    "        for (let i = 0; i < _rets.length; i++) {",
                    "            const sig = _td[i] ? _td[i].value : 0;",
                    f"            pos = {_band_eq_expr};",
                    "            const r = _rets[i].ret * pos;",
                    "            eq *= (1 + r);",
                    "            if (hasBh) bh *= (1 + _bhRets[i].ret);",
                    "            eqData.push({time: _rets[i].time, value: eq});",
                    "            if (hasBh) bhData.push({time: _bhRets[i].time, value: bh});",
                    "            if (pos !== 0) { sumR += r; sumR2 += r*r; n++; if(r>0) wins++; tradesN++; }",
                    "            if (pos === 1) nLong++;",
                    "            if (eq > peak) peak = eq;",
                    "            const dd = (eq - peak) / peak;",
                    "            if (dd < maxDD) maxDD = dd;",
                    "        }",
                    "        const annR = n > 0 ? sumR / _rets.length * 252 : 0;",
                    "        const annV = n > 0 ? Math.sqrt((sumR2/_rets.length - Math.pow(sumR/_rets.length,2)) * 252) : 1;",
                    "        const sharpe = annV > 0 ? annR / annV : 0;",
                    "        const wr = tradesN > 0 ? wins/tradesN : 0;",
                    "        const timeMkt = n / _rets.length;",
                    "        const s = 100 / eqData[0].value;",
                    "        eqData.forEach(d => d.value *= s);",
                    "        if (hasBh) { const bs = 100/bhData[0].value; bhData.forEach(d => d.value *= bs); }",
                    "        const totalRet = eqData[eqData.length-1].value / 100 - 1;",
                    "        const bhRet = hasBh ? bhData[bhData.length-1].value / 100 - 1 : null;",
                    "        const nYears = _rets.length / 252;",
                    "        const cagr = Math.pow(eqData[eqData.length-1].value/100, 1/nYears) - 1;",
                    "        return {eqData, bhData, totalRet, bhRet, cagr, sharpe, maxDD, wr, timeMkt};",
                    "    }",
                    "",
                    "    function _fmtPct(v, showSign=true) {",
                    "        return (showSign && v>0?'+':'') + (v*100).toFixed(1)+'%';",
                    "    }",
                    "",
                    "    function _updateStats(res) {",
                    "        if (!_statsEl || !_statsTblEl) return;",
                    f"        const lbl = 'color:{'rgba(255,255,255,0.55)' if is_dark else 'rgba(0,0,0,0.45)'}';",
                    f"        const val = 'color:{'rgba(255,255,255,0.92)' if is_dark else 'rgba(0,0,0,0.88)'};font-weight:600';",
                    "        const rows = [",
                    "            ['Return', _fmtPct(res.totalRet)],",
                    "            res.bhRet !== null ? ['B&H', _fmtPct(res.bhRet)] : null,",
                    "            ['CAGR',   _fmtPct(res.cagr)],",
                    "            ['Sharpe', res.sharpe.toFixed(2)],",
                    "            ['Max DD', _fmtPct(res.maxDD, false)],",
                    "            ['Win Rate', _fmtPct(res.wr, false)],",
                    "            ['In Mkt',  _fmtPct(res.timeMkt, false)],",
                    "        ].filter(Boolean);",
                    "        _statsTblEl.innerHTML = rows.map(([k,v]) => `<tr><td style=\"${lbl};padding:1px 8px 1px 0;white-space:nowrap\">${k}</td><td style=\"${val};text-align:right\">${v}</td></tr>`).join('');",
                    "    }",
                    "",
                    f"    const _eq0 = _buildEquity({thr0}, {_thr2});",
                    "    if (_eqSeries) _eqSeries.setData(_eq0.eqData);",
                    "    if (_bhSeries && _eq0.bhData.length) _bhSeries.setData(_eq0.bhData);",
                    "    _updateStats(_eq0);",
                ]) + [
                    f"    const _thP = LightweightCharts.createSeriesMarkers({price_s0}, []);  // markers hidden",
                    f"    _shadeSeries.setData(_buildShade({thr0}, {_thr2}));",

                    "",
                    "    (function() {",
                    f"        const m0 = _mkrs({thr0}, {_thr2});",
                    '        document.getElementById("th-count").textContent = m0.filter(x=>x.shape==="arrowUp").length + " signals";',
                    "    })();",
                    f"    let _thLo = {thr0}, _thHi = {_thr2};",
                    "    const _loSlider = document.getElementById('th-lo-slider');",
                    "    const _hiSlider = document.getElementById('th-hi-slider');",
                    "",
                    "    function _th_update(lo, hi) {",
                    '        document.getElementById("th-lo-label").textContent = "\\u03b8_lo\\u00a0" + lo.toFixed(_tDec);',
                    '        document.getElementById("th-hi-label").textContent = "\\u03b8_hi\\u00a0" + hi.toFixed(_tDec);',
                    "        const m = _mkrs(lo, hi);",
                    "        _thP.setMarkers([]);  // arrows hidden",
                    "        _shadeSeries.setData(_buildShade(lo, hi));",
                    "        const _msk = _buildMask(lo, hi);",
                    "        for (const s of _maskedSeries) s.setData(_msk);",
                    "        _activeMap = {}; _msk.forEach(d => { _activeMap[d.time] = d.active; });",
                    "        for (const pl of _thresholdLoLines) pl.applyOptions({price: lo});",
                    "        for (const pl of _thresholdHiLines) pl.applyOptions({price: hi});",
                    '        document.getElementById("th-count").textContent = m.filter(x=>x.shape==="arrowUp").length + " signals";',
                ] + ([] if not tc.get("ret_data") else [
                    "        if (_rets && _rets.length) {",
                    "            const eq = _buildEquity(lo, hi);",
                    "            if (_eqSeries) _eqSeries.setData(eq.eqData);",
                    "            if (_bhSeries && eq.bhData.length) _bhSeries.setData(eq.bhData);",
                    "            _updateStats(eq);",
                    "        }",
                ]) + [
                    "    }",
                    "",
                    "    _loSlider.addEventListener('input', function() {",
                    "        _thLo = parseFloat(this.value);",
                    "        if (_thLo > _thHi) { _thHi = _thLo; _hiSlider.value = _thHi; }",
                    "        _th_update(_thLo, _thHi);",
                    "    });",
                    "    _hiSlider.addEventListener('input', function() {",
                    "        _thHi = parseFloat(this.value);",
                    "        if (_thHi < _thLo) { _thLo = _thHi; _loSlider.value = _thLo; }",
                    "        _th_update(_thLo, _thHi);",
                    "    });",
                ])

        # ── Smoothing sliders from child panes — deduplicated by label ─────
        # Group configs by label so same-label sliders share one control.
        from collections import defaultdict as _dd
        _sm_by_label = _dd(list)
        for sc in _pane_smoothing_configs:
            _sm_by_label[sc["label"]].append(sc)

        smoothing_html = ""
        smoothing_js   = ""
        _lbl_c = "rgba(255,255,255,0.88)" if is_dark else "rgba(0,0,0,0.78)"
        for global_idx, (lbl, group) in enumerate(_sm_by_label.items()):
            sc  = group[0]          # use first entry for slider config
            acc = sc["color"]
            sid = f"db-sm-slider-{global_idx}"
            lid = f"db-sm-label-{global_idx}"
            # All series vars that this label drives (one per pane in the group)
            all_svars = [g["_svar"] for g in group]

            if sc["mode"] == "variants":
                keys     = sc["variants_keys"]
                n_var    = len(keys)
                init_idx = sc["variants_init"]
                init_key = keys[init_idx]
                var_data_id = f"_dbSmVarData{global_idx}"
                var_keys_id = f"_dbSmVarKeys{global_idx}"
                smoothing_html += (
                    f'<div style="display:flex;align-items:center;justify-content:center;'
                    f'gap:10px;background:transparent;padding:4px 16px;white-space:nowrap;height:36px">'
                    f'<span id="{lid}" style="color:{_lbl_c};'
                    f"font:11px/1 'SF Mono','Consolas',monospace;min-width:72px\">"
                    f'{lbl} {init_key}</span>'
                    f'<input id="{sid}" type="range" min="0" max="{n_var - 1}" step="1" '
                    f'value="{init_idx}" style="width:220px;cursor:pointer;accent-color:{acc}">'
                    f'<span style="min-width:60px"></span>'
                    f'</div>\n'
                )
                # Build JS: one slider drives all series in the group (dedup by label)
                js_lines_v = [f"    // \u2500\u2500 Smoothing slider #{global_idx} (variants) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"]
                for gi, g in enumerate(group):
                    gdata_id = f"_dbSmVarData{global_idx}_{gi}"
                    init_idx_g = g["variants_init"]
                    js_lines_v.append(f"    const {gdata_id} = {json.dumps(g['variants_data'], separators=(',', ':'))};")
                    js_lines_v.append(f"    {g['_svar']}.setData({gdata_id}[{init_idx_g}]);")
                js_lines_v += [
                    f"    const {var_keys_id} = {json.dumps(keys, separators=(',', ':'))};",
                    f"    document.getElementById('{sid}').addEventListener('input', function() {{",
                    f"        const idx = parseInt(this.value);",
                    f"        document.getElementById('{lid}').textContent = '{lbl} ' + {var_keys_id}[idx];",
                ]
                for gi, g in enumerate(group):
                    gdata_id = f"_dbSmVarData{global_idx}_{gi}"
                    js_lines_v.append(f"        {g['_svar']}.setData({gdata_id}[idx]);")
                js_lines_v += ["    });", ""]
                smoothing_js += "\n".join(js_lines_v)
            else:
                # ── SMA/EMA mode: deduplicated — drive all series in group ──
                win_init = sc["window_init"]
                smoothing_html += (
                    f'<div style="display:flex;align-items:center;'
                    f'gap:10px;background:transparent;padding:4px 16px;white-space:nowrap;height:36px">'
                    f'<span id="{lid}" style="color:{_lbl_c};'
                    f"font:11px/1 'SF Mono','Consolas',monospace;min-width:100px\">"
                    f'{lbl} {win_init}</span>'
                    f'<input id="{sid}" type="range" '
                    f'min="{sc["window_min"]}" max="{sc["window_max"]}" step="{sc["window_step"]}" '
                    f'value="{win_init}" style="width:220px;cursor:pointer;accent-color:{acc}">'
                    f'</div>\n'
                )
                mode = sc["mode"]
                js_lines = []
                for gi, g in enumerate(group):
                    raw_id = f"_dbSmRaw{global_idx}_{gi}"
                    if mode == "ema":
                        cfn = (f"function _dbSmCmp{global_idx}_{gi}(rd,win){{"
                               "const k=2/(win+1);let e=null;const o=[];"
                               "for(const d of rd){e=e===null?d.value:d.value*k+e*(1-k);o.push({time:d.time,value:e});}"
                               "return o;}")
                    else:
                        cfn = (f"function _dbSmCmp{global_idx}_{gi}(rd,win){{"
                               "const o=[];let s=0,b=[];"
                               "for(const d of rd){b.push(d.value);s+=d.value;"
                               "if(b.length>win){s-=b.shift();}o.push({time:d.time,value:s/b.length});}"
                               "return o;}")
                    js_lines += [
                        f"    const {raw_id} = {json.dumps(g['raw_data'], separators=(',', ':'))};",
                        f"    {cfn}",
                        f"    {g['_svar']}.setData(_dbSmCmp{global_idx}_{gi}({raw_id},{win_init}));",
                    ]
                js_lines += [
                    f"    document.getElementById('{sid}').addEventListener('input', function() {{",
                    f"        const win = parseInt(this.value);",
                    f"        document.getElementById('{lid}').textContent = '{lbl} ' + win;",
                ]
                for gi, g in enumerate(group):
                    js_lines.append(f"        {g['_svar']}.setData(_dbSmCmp{global_idx}_{gi}(_dbSmRaw{global_idx}_{gi},win));")
                js_lines += ["    });", ""]
                smoothing_js += "\n".join(js_lines)

        if smoothing_html:
            divs_html += "\n" + smoothing_html
        if smoothing_js:
            slider_js = slider_js + "\n    " + smoothing_js.replace("\n", "\n    ")

        # Crosshair + time scale sync JS
        sync_js = """
    // ── Initial fit: chart[0] sets the master range; all others match by TIME ─
    charts[0].timeScale().fitContent();
    setTimeout(() => {
        const range = charts[0].timeScale().getVisibleRange();
        if (range) {
            for (let j = 0; j < charts.length; j++) {
                charts[j].timeScale().setVisibleRange(range);
            }
        }
    }, 300);
    // ── Force equal price-scale width so all panes align horizontally ─────────
    let _lastMaxW = 0, _alignStable = 0;
    function _alignScales() {
        const widths = charts.map(c => { try { return c.priceScale('right').width(); } catch(e) { return 0; } });
        const maxW = Math.max(60, ...widths);
        if (maxW !== _lastMaxW) {
            _lastMaxW = maxW;
            _alignStable = 0;
            charts.forEach(c => c.applyOptions({rightPriceScale:{minimumWidth:maxW}}));
        }
        if (++_alignStable < 20) requestAnimationFrame(_alignScales);
    }
    setTimeout(_alignScales, 60);

    // ── Sync time scales (scroll / zoom) — time range keeps dates identical ──
    let _syncing = false;
    function syncTimeScales(srcIdx) {
        charts[srcIdx].timeScale().subscribeVisibleTimeRangeChange(() => {
            if (_syncing) return;
            _syncing = true;
            const range = charts[srcIdx].timeScale().getVisibleRange();
            if (range) {
                for (let j = 0; j < charts.length; j++) {
                    if (j !== srcIdx) charts[j].timeScale().setVisibleRange(range);
                }
            }
            _syncing = false;
        });
    }
    for (let i = 0; i < charts.length; i++) syncTimeScales(i);

    // ── Crosshair sync across panes ────────────────────────────
    let _csSync = false;
    for (let i = 0; i < charts.length; i++) {
        charts[i].subscribeCrosshairMove(param => {
            if (_csSync) return;
            _csSync = true;
            if (!param.time || !param.point || param.point.x < 0 || param.point.y < 0) {
                for (let j = 0; j < charts.length; j++) {
                    if (j !== i) charts[j].clearCrosshairPosition();
                }
            } else {
                for (let j = 0; j < charts.length; j++) {
                    if (j !== i && firstSeries[j]) {
                        charts[j].setCrosshairPosition(
                            firstSeries[j].coordinateToPrice(param.point.y) ?? 0,
                            param.time, firstSeries[j]
                        );
                    }
                }
            }
            _csSync = false;
        });
    }

    // ── Resize ──────────────────────────────────────────────────
    window.addEventListener('resize', () => {
        charts[0].timeScale().fitContent();
        const r = charts[0].timeScale().getVisibleRange();
        if (r) for (let j = 1; j < charts.length; j++) charts[j].timeScale().setVisibleRange(r);
    });"""

        # Theme may override body background with CSS marble/texture
        custom_bg_css = self._theme.get("background_css", "")
        bg_css = custom_bg_css if custom_bg_css else f"background:{bg};"

        # SVG marble texture (rendered behind charts)
        bg_svg = self._theme.get("background_svg", "")

        # ── Background image glass overlay ───────────────────────────────
        _bgi = self._bg_image_config
        if _bgi:
            # Put image on body so backdrop-filter on #glass can blur it
            _url_safe = _bgi["url"].replace('"', "%22")
            bg_css = f'background:url("{_url_safe}") center/cover no-repeat;'
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

        return f"""<!DOCTYPE html>
<html><head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<script>{lc_js}</script>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{{bg_css}overflow-y:auto;overflow-x:hidden;position:relative;border-radius:12px;padding-bottom:36px}}
#signum-logo{{position:absolute;right:12px;bottom:6px;z-index:5;opacity:0.7;pointer-events:none;{_logo_invert}}}
</style>
</head><body>
{bg_svg}{_glass_open}{divs_html}
{_glass_close}{'<img id="signum-logo" src="data:image/svg+xml;base64,' + _LOGO_B64 + '" width="30" height="30" alt="Signum">' if self._logo else ''}
<script>
    const charts = [];
    const firstSeries = [];
    {scripts_body}
    {sync_js}
    {slider_js}
</script>
</body></html>"""

    # ── Display Methods ───────────────────────────────────────────────────

    def _repr_html_(self) -> str:
        import base64
        chart_html = self._build_html()
        b64 = base64.b64encode(chart_html.encode("utf-8")).decode("ascii")
        h = self.total_height + 40
        uid = f"fd{id(self)}"
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
        """Display in a Jupyter notebook."""
        try:
            from IPython.display import display, HTML
            display(HTML(self._repr_html_()))
        except ImportError:
            print("IPython not available. Use .save() or .render() instead.")

    def render(self) -> str:
        return self._build_html()

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            f.write(self._build_html())

    def to_dash(self, id: Optional[str] = None, style: Optional[dict] = None):
        from dash import html
        default_style = {
            "width": "100%",
            "height": f"{self.total_height + 40}px",
            "border": "none",
            "borderRadius": "4px",
        }
        if style:
            default_style.update(style)
        return html.Iframe(
            id=id or "forge-dashboard",
            srcDoc=self._build_html(),
            style=default_style,
        )

    def to_streamlit(self, height: Optional[int] = None):
        import streamlit.components.v1 as components
        components.html(self._build_html(), height=height or self.total_height + 40)
