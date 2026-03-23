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

import json
import math
from typing import List, Optional

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
        threshold: float = 0.0,
        min_val: float = -0.05,
        max_val: float = 0.05,
        step: float = 0.001,
        value_col: Optional[str] = None,
        price_pane: int = 0,
        buy_color: Optional[str] = None,
        sell_color: Optional[str] = None,
        daily_returns: Optional[pd.Series] = None,
        bh_returns: Optional[pd.Series] = None,
        equity_pane: Optional[int] = None,
    ) -> "Dashboard":
        """Add an interactive threshold slider to the dashboard.

        Adds a signal pane showing the signal line + moving threshold,
        shading on the price pane, and a slider below all panes.

        Parameters
        ----------
        df : DataFrame with time + value columns (signal series).
        threshold : initial threshold value.
        price_pane : index of the pane to add markers/shading to (default 0).
        daily_returns : optional pd.Series of daily strategy returns (same index as df).
            When provided, the slider will live-update an equity curve + stats legend.
        bh_returns : optional pd.Series of daily buy-and-hold returns for comparison.
        equity_pane : pane index that contains the equity/area series to update live.
        """
        time_col = Chart._detect_time_col(df)
        df = df.copy()
        if time_col == "__index__":
            df["time"] = df.index
        elif time_col != "time":
            df["time"] = df[time_col]
        if pd.api.types.is_datetime64_any_dtype(df["time"]):
            df["time"] = df["time"].dt.strftime("%Y-%m-%d")
        else:
            df["time"] = pd.to_datetime(df["time"]).dt.strftime("%Y-%m-%d")

        vcol = value_col
        if not vcol:
            for name in ("value", "close", "Close", "VALUE", "price", "Price"):
                if name in df.columns:
                    vcol = name
                    break
            if not vcol:
                for col in df.columns:
                    if col != "time" and pd.api.types.is_numeric_dtype(df[col]):
                        vcol = col
                        break
        data = (
            df[["time", vcol]]
            .rename(columns={vcol: "value"})
            .dropna(subset=["value"])
            .to_dict("records")
        )
        up_clr = buy_color or self._theme.get("candlestick", {}).get("upColor", "#26a69a")
        dn_clr = sell_color or self._theme.get("candlestick", {}).get("downColor", "#ef5350")
        decimals = max(0, -int(math.floor(math.log10(step)))) if step > 0 else 3

        # Prepare daily returns arrays for live equity computation (aligned to signal data)
        ret_data = None
        bh_data = None
        if daily_returns is not None:
            ret_series = daily_returns.fillna(0)
            # Align to the signal dates
            ret_df = pd.DataFrame({"time": df["time"], "ret": ret_series.values if len(ret_series) == len(df) else ret_series.reindex(pd.RangeIndex(len(df))).fillna(0).values})
            ret_data = ret_df[["time", "ret"]].to_dict("records")
        if bh_returns is not None:
            bh_series = bh_returns.fillna(0)
            bh_df_aligned = pd.DataFrame({"time": df["time"], "ret": bh_series.values if len(bh_series) == len(df) else bh_series.reindex(pd.RangeIndex(len(df))).fillna(0).values})
            bh_data = bh_df_aligned[["time", "ret"]].to_dict("records")

        self._threshold_config = {
            "data": data,
            "threshold": threshold,
            "min_val": min_val,
            "max_val": max_val,
            "step": step,
            "decimals": decimals,
            "price_pane": price_pane,
            "buy_color": up_clr,
            "sell_color": dn_clr,
            "ret_data": ret_data,
            "bh_data": bh_data,
            "equity_pane": equity_pane,
        }
        return self

    @property
    def total_height(self) -> int:
        title_h = sum(24 for i in range(len(self._panes)) if self._get_title(i))
        slider_h = 36 if self._threshold_config else 0
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

            pane_divs.append(
                f'<div id="{div_id}" style="width:100%;height:{pane._height}px;'
                f'margin-bottom:{self._gap}px"></div>'
            )

            # Build chart options — hide time axis labels on non-bottom panes
            chart_opts = pane._build_chart_options()
            # Dashboard panes use fixed container divs, not autoSize
            chart_opts.pop("autoSize", None)
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
            shade_fill = f"rgba({sr},{sg},{sb},0.10)"
            shade_line = f"rgba({sr},{sg},{sb},0.25)"

            slider_html = (
                f'<div id="th-bar" style="display:flex;align-items:center;justify-content:center;'
                f'gap:10px;background:transparent;padding:6px 16px;white-space:nowrap;height:36px">'
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
            divs_html += "\n" + slider_html

            slider_js = "\n".join([
                "",
                "    // ── Dashboard threshold slider ───────────────────────────────",
                f"    const _td = {tdata_json};",
                f'    const _tBuy = "{tc["buy_color"]}";',
                f'    const _tSell = "{tc["sell_color"]}";',
                f"    const _tDec = {dec};",
                "",
                "    // Shading area on price pane",
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
                "    // Signal line pane — grey out below threshold, colored above",
                "    const _thresholdLines = [];",
                "    const _sigSeries = [];",
                "    for (let ci = 0; ci < charts.length; ci++) {",
                f"        if (ci === {pp}) continue;",
                "        const fs = firstSeries[ci];",
                "        if (fs) {",
                "            try {",
                "                // Override baseline: above θ = colored, below θ = grey",
                "                fs.applyOptions({",
                f'                    baseValue: {{type:"price",price:{thr0}}},',
                f'                    topLineColor: "{tc["buy_color"]}",',
                f'                    topFillColor1: "rgba({sr},{sg},{sb},0.28)",',
                f'                    topFillColor2: "rgba({sr},{sg},{sb},0.05)",',
                '                    bottomLineColor: "rgba(120,120,120,0.4)",',
                '                    bottomFillColor1: "rgba(120,120,120,0.05)",',
                '                    bottomFillColor2: "rgba(120,120,120,0.18)",',
                "                });",
                "                _sigSeries.push(fs);",
                f'                const pl = fs.createPriceLine({{price:{thr0},title:"\\u03b8",color:"{tc["buy_color"]}",lineWidth:1,lineStyle:2,axisLabelVisible:true}});',
                "                _thresholdLines.push(pl);",
                "            } catch(e) {}",
                "        }",
                "    }",
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
                f"    const _thP = LightweightCharts.createSeriesMarkers({price_s0}, _mkrs({thr0}));",
                f"    _shadeSeries.setData(_buildShade({thr0}));",
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
                "    if (_eqPaneIdx >= 0) {",
                f"        const _epDiv = document.getElementById('pane' + _eqPaneIdx);",
                "        _statsEl = document.createElement('div');",
                f"        _statsEl.style.cssText = 'position:absolute;top:8px;left:8px;z-index:6;"
                f"background:{'rgba(0,0,0,0.52)' if is_dark else 'rgba(255,255,255,0.72)'};"
                f"backdrop-filter:blur(4px);border-radius:6px;padding:6px 10px;pointer-events:none;"
                f"font:11px/1.6 \\'SF Mono\\',\\'Consolas\\',monospace;';",
                "        if (_epDiv) { _epDiv.style.position='relative'; _epDiv.appendChild(_statsEl); }",
                "    }",
                "",
                "    function _buildEquity(thr) {",
                "        // position = 1 when signal >= thr, held until signal drops below",
                "        let pos = 0, eq = 100, bh = 100;",
                "        const eqData = [], bhData = [];",
                "        let totRet=0,n=0,sumR=0,sumR2=0,peak=100,maxDD=0;",
                "        let wins=0,tradesN=0;",
                "        const hasBh = _bhRets.length === _rets.length;",
                "        for (let i = 0; i < _rets.length; i++) {",
                "            const sig = _td[i] ? _td[i].value : 0;",
                "            pos = sig >= thr ? 1 : 0;",
                "            const r = _rets[i].ret * pos;",
                "            eq *= (1 + r);",
                "            if (hasBh) bh *= (1 + _bhRets[i].ret);",
                "            eqData.push({time: _rets[i].time, value: eq});",
                "            if (hasBh) bhData.push({time: _bhRets[i].time, value: bh});",
                "            if (pos) { sumR += r; sumR2 += r*r; n++; if(r>0) wins++; tradesN++; }",
                "            if (eq > peak) peak = eq;",
                "            const dd = (eq - peak) / peak;",
                "            if (dd < maxDD) maxDD = dd;",
                "        }",
                "        const annR = n > 0 ? sumR / _rets.length * 252 : 0;",
                "        const annV = n > 0 ? Math.sqrt((sumR2/_rets.length - Math.pow(sumR/_rets.length,2)) * 252) : 1;",
                "        const sharpe = annV > 0 ? annR / annV : 0;",
                "        const nYears = _rets.length / 252;",
                "        const cagr = Math.pow(eq/100, 1/nYears) - 1;",
                "        const totalRet = eq/100 - 1;",
                "        const bhRet = hasBh ? bh/100 - 1 : null;",
                "        const wr = tradesN > 0 ? wins/tradesN : 0;",
                "        const timeMkt = n / _rets.length;",
                "        // Re-index to 100",
                "        const s = 100 / eqData[0].value;",
                "        eqData.forEach(d => d.value *= s);",
                "        if (hasBh) { const bs = 100/bhData[0].value; bhData.forEach(d => d.value *= bs); }",
                "        return {eqData, bhData, totalRet, bhRet, cagr, sharpe, maxDD, wr, timeMkt};",
                "    }",
                "",
                "    function _fmtPct(v, showSign=true) {",
                "        return (showSign && v>0?'+':'') + (v*100).toFixed(1)+'%';",
                "    }",
                "",
                "    function _updateStats(res) {",
                "        if (!_statsEl) return;",
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
                "        _statsEl.innerHTML = '<table style=\"border-collapse:collapse\">' +",
                "            rows.map(([k,v]) => `<tr><td style=\"${lbl};padding:1px 8px 1px 0;white-space:nowrap\">${k}</td><td style=\"${val};text-align:right\">${v}</td></tr>`).join('')",
                "            + '</table>';",
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
                '        document.getElementById("th-label").textContent = "\\u03b8\\u00a0=\\u00a0" + thr.toFixed(_tDec);',
                "        const m = _mkrs(thr);",
                "        _thP.setMarkers(m);",
                "        _shadeSeries.setData(_buildShade(thr));",
                "        for (const pl of _thresholdLines) { pl.applyOptions({price: thr}); }",
                '        for (const ss of _sigSeries) { ss.applyOptions({baseValue:{type:"price",price:thr}}); }',
                '        document.getElementById("th-count").textContent = m.filter(x=>x.shape==="arrowUp").length + " signals";',
                "        if (_rets && _rets.length) {",
                "            const eq = _buildEquity(thr);",
                "            if (_eqSeries) _eqSeries.setData(eq.eqData);",
                "            if (_bhSeries && eq.bhData.length) _bhSeries.setData(eq.bhData);",
                "            _updateStats(eq);",
                "        }",
                "    });",
            ])

        # Crosshair + time scale sync JS
        sync_js = """
    // ── Initial fit: all charts fit their own content, then sync by LOGICAL RANGE ─
    for (let i = 0; i < charts.length; i++) charts[i].timeScale().fitContent();
    setTimeout(() => {
        const range = charts[0].timeScale().getVisibleLogicalRange();
        if (range) {
            for (let j = 1; j < charts.length; j++) {
                charts[j].timeScale().setVisibleLogicalRange(range);
            }
        }
    }, 50);

    // ── Sync time scales (scroll / zoom) — logical range keeps bar positions identical ──
    let _syncing = false;
    function syncTimeScales(srcIdx) {
        charts[srcIdx].timeScale().subscribeVisibleLogicalRangeChange(() => {
            if (_syncing) return;
            _syncing = true;
            const range = charts[srcIdx].timeScale().getVisibleLogicalRange();
            if (range) {
                for (let j = 0; j < charts.length; j++) {
                    if (j !== srcIdx) charts[j].timeScale().setVisibleLogicalRange(range);
                }
            }
            _syncing = false;
        });
    }
    for (let i = 0; i < charts.length; i++) syncTimeScales(i);

    // ── Sync crosshairs ─────────────────────────────────────────
    function syncCrosshair(srcIdx) {
        charts[srcIdx].subscribeCrosshairMove(param => {
            for (let j = 0; j < charts.length; j++) {
                if (j === srcIdx) continue;
                if (!param || !param.logical === null) {
                    charts[j].clearCrosshairPosition();
                } else if (firstSeries[j] && param.time) {
                    charts[j].setCrosshairPosition(NaN, param.time, firstSeries[j]);
                } else {
                    charts[j].clearCrosshairPosition();
                }
            }
        });
    }
    for (let i = 0; i < charts.length; i++) syncCrosshair(i);

    // ── Resize ──────────────────────────────────────────────────
    window.addEventListener('resize', () => {
        charts[0].timeScale().fitContent();
        const r = charts[0].timeScale().getVisibleLogicalRange();
        if (r) for (let j = 1; j < charts.length; j++) charts[j].timeScale().setVisibleLogicalRange(r);
    });"""

        # Theme may override body background with CSS marble/texture
        custom_bg_css = self._theme.get("background_css", "")
        bg_css = custom_bg_css if custom_bg_css else f"background:{bg};"

        # SVG marble texture (rendered behind charts)
        bg_svg = self._theme.get("background_svg", "")

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
{bg_svg}
{divs_html}
{'<img id="signum-logo" src="data:image/svg+xml;base64,' + _LOGO_B64 + '" width="30" height="30" alt="Signum">' if self._logo else ''}
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
