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
    ) -> "Dashboard":
        """Add an interactive threshold slider to the dashboard.

        Adds a signal pane showing the signal line + moving threshold,
        shading on the price pane, and a slider below all panes.

        Parameters
        ----------
        df : DataFrame with time + value columns (signal series).
        threshold : initial threshold value.
        price_pane : index of the pane to add markers/shading to (default 0).
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
