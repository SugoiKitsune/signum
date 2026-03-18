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
from typing import List, Optional

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
        # Infer theme from first pane when not given explicitly
        if theme:
            self._theme_name = theme
        elif self._panes:
            self._theme_name = self._panes[0]._theme_name
        else:
            self._theme_name = "dark"
        self._theme = THEMES.get(self._theme_name, THEMES["dark"])
        self._gap = gap

    def add(self, pane: Chart, title: Optional[str] = None) -> "Dashboard":
        """Append a chart pane with an optional title."""
        self._panes.append(pane)
        if title:
            # Pad the titles list to match the pane index
            while len(self._titles) < len(self._panes) - 1:
                self._titles.append("")
            self._titles.append(title)
        return self

    @property
    def total_height(self) -> int:
        title_h = sum(24 for i in range(len(self._panes)) if self._get_title(i))
        return sum(p._height for p in self._panes) + self._gap * max(0, len(self._panes) - 1) + title_h

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
        is_dark = self._theme_name in ("dark", "midnight")
        title_color = "rgba(255,255,255,0.55)" if is_dark else "rgba(0,0,0,0.55)"
        title_font = "11px -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"

        # Logo invert for light backgrounds
        _bg_hex = bg.lstrip("#")
        _r, _g, _b = (int(_bg_hex[i:i+2], 16) for i in (0, 2, 4))
        _logo_invert = "filter:invert(1);" if (_r * 0.299 + _g * 0.587 + _b * 0.114) > 150 else ""

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

        # Crosshair + time scale sync JS
        sync_js = """
    // ── Initial fit: use chart 0 as the master range ────────────
    charts[0].timeScale().fitContent();
    setTimeout(() => {
        const range = charts[0].timeScale().getVisibleLogicalRange();
        if (range) {
            for (let j = 1; j < charts.length; j++) {
                charts[j].timeScale().setVisibleLogicalRange(range);
            }
        }
    }, 0);

    // ── Sync time scales (scroll / zoom) ────────────────────────
    let _syncing = false;
    function syncTimeScales(srcIdx) {
        charts[srcIdx].timeScale().subscribeVisibleLogicalRangeChange(range => {
            if (_syncing || !range) return;
            _syncing = true;
            for (let j = 0; j < charts.length; j++) {
                if (j !== srcIdx) charts[j].timeScale().setVisibleLogicalRange(range);
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
                if (!param || !param.time) {
                    charts[j].clearCrosshairPosition();
                } else if (firstSeries[j]) {
                    charts[j].setCrosshairPosition(NaN, param.time, firstSeries[j]);
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

        return f"""<!DOCTYPE html>
<html><head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<script>{lc_js}</script>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:{bg};overflow-y:auto;overflow-x:hidden;position:relative}}
#signum-logo{{position:absolute;left:12px;bottom:4px;z-index:5;opacity:0.6;pointer-events:none;{_logo_invert}}}
</style>
</head><body>
{divs_html}
{'<img id="signum-logo" src="data:image/svg+xml;base64,' + _LOGO_B64 + '" width="36" height="36" alt="Signum">' if self._logo else ''}
<script>
    const charts = [];
    const firstSeries = [];
    {scripts_body}
    {sync_js}
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
            f'<div id="{uid}" style="width:100%;height:{h}px;border-radius:4px;overflow:hidden;">'
            f'</div><script>'
            f'(function(){{'
            f'var b=atob("{b64}");'
            f'var blob=new Blob([b],{{type:"text/html"}});'
            f'var url=URL.createObjectURL(blob);'
            f'var f=document.createElement("iframe");'
            f'f.src=url;f.style.width="100%";f.style.height="{h}px";'
            f'f.style.border="none";f.style.borderRadius="4px";'
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
