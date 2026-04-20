"""Dashboard - Multi-pane synchronized financial charts.

Creates vertically stacked chart panes with synchronized time axes
and crosshairs — the LC equivalent of Plotly make_subplots(shared_xaxes=True).
"""

import html as _html
import json
import math
import warnings as _w
from collections import defaultdict as _dd
from typing import Dict, List, Optional, Union

import pandas as pd

from .chart import Chart, _get_lc_js, _LOGO_B64
from .themes import THEMES

# ── Signal mode lookup table ──────────────────────────────────────────────────
# Each entry: (long_idx, short_idx, long_pos, short_pos, eq_pos,
#              precompute_js, update_baseline, invert_colors, show_neg_line,
#              lbl_pfx_js, lbl_pfx_py, is_band)
def _sm_above(op=">=", strict=False):
    js_op = ">" if strict else ">="
    sym = ">" if strict else "\u2265"
    return dict(
        long_idx=f"_td[i].value {js_op} thr", short_idx="false",
        long_pos=f"sig {js_op} thr", short_pos="false",
        eq_pos=f"sig {js_op} thr ? 1 : 0",
        precompute="", update_baseline=True, invert=False, neg_line=False,
        lbl_js=f"\\u03b8\\u00a0{op}\\u00a0", lbl_py=f"\u03b8\u00a0{op}\u00a0",
    )

_SIGNAL_MODES = {
    "above":        _sm_above("\u2265"),
    "above_strict": _sm_above(">", strict=True),
    "below": dict(
        long_idx="_td[i].value <= thr", short_idx="false",
        long_pos="sig <= thr", short_pos="false", eq_pos="sig <= thr ? 1 : 0",
        precompute="", update_baseline=True, invert=True, neg_line=False,
        lbl_js="\\u03b8\\u00a0<=\\u00a0", lbl_py="\u03b8\u00a0\u2264\u00a0",
    ),
    "below_strict": dict(
        long_idx="_td[i].value < thr", short_idx="false",
        long_pos="sig < thr", short_pos="false", eq_pos="sig < thr ? 1 : 0",
        precompute="", update_baseline=True, invert=True, neg_line=False,
        lbl_js="\\u03b8\\u00a0<\\u00a0", lbl_py="\u03b8\u00a0<\u00a0",
    ),
    "rising": dict(
        long_idx="_delta[i] >= thr", short_idx="false",
        long_pos="_delta[i] >= thr", short_pos="false", eq_pos="_delta[i] >= thr ? 1 : 0",
        precompute="const _delta = _td.map((d,i) => i===0 ? 0 : d.value - _td[i-1].value);",
        update_baseline=True, invert=False, neg_line=False,
        lbl_js="\\u0394\\u00a0\\u2265\\u00a0", lbl_py="\u0394\u00a0\u2265\u00a0",
    ),
    "within": dict(
        long_idx="_td[i].value >= thr", short_idx="false",
        long_pos="sig >= lo && sig <= hi", short_pos="false",
        eq_pos="(sig >= lo && sig <= hi) ? 1 : 0",
        precompute="", update_baseline=False, invert=False, neg_line=False,
        lbl_js="\\u03b8_lo", lbl_py="\u03b8_lo\u00a0", is_band=True,
        shade_cond="v >= lo && v <= hi",
    ),
    "outside": dict(
        long_idx="_td[i].value < thr", short_idx="false",
        long_pos="sig < lo || sig > hi", short_pos="false",
        eq_pos="(sig < lo || sig > hi) ? 1 : 0",
        precompute="", update_baseline=False, invert=False, neg_line=False,
        lbl_js="\\u03b8_lo", lbl_py="\u03b8_lo\u00a0", is_band=True,
        shade_cond="v < lo || v > hi",
    ),
}
_OP_ALIAS = {">=": "above", ">": "above_strict", "<=": "below", "<": "below_strict",
             "between": "within", "band": "within", "beyond": "outside"}


# ── Reusable JS snippets ─────────────────────────────────────────────────────

_JS_SYNC = """
    // ── Fit + sync ──
    charts[0].timeScale().fitContent();
    setTimeout(() => {
        const range = charts[0].timeScale().getVisibleRange();
        if (range) for (let j = 0; j < charts.length; j++) charts[j].timeScale().setVisibleRange(range);
    }, 300);
    let _lastMaxW = 0, _alignStable = 0;
    function _alignScales() {
        const widths = charts.map(c => { try { return c.priceScale('right').width(); } catch(e) { return 0; } });
        const maxW = Math.max(60, ...widths);
        if (maxW !== _lastMaxW) { _lastMaxW = maxW; _alignStable = 0; charts.forEach(c => c.applyOptions({rightPriceScale:{minimumWidth:maxW}})); }
        if (++_alignStable < 20) requestAnimationFrame(_alignScales);
    }
    setTimeout(_alignScales, 60);
    let _syncing = false;
    function syncTimeScales(srcIdx) {
        charts[srcIdx].timeScale().subscribeVisibleTimeRangeChange(() => {
            if (_syncing) return; _syncing = true;
            const range = charts[srcIdx].timeScale().getVisibleRange();
            if (range) for (let j = 0; j < charts.length; j++) { if (j !== srcIdx) charts[j].timeScale().setVisibleRange(range); }
            _syncing = false;
        });
    }
    for (let i = 0; i < charts.length; i++) syncTimeScales(i);
    let _csSync = false;
    for (let i = 0; i < charts.length; i++) {
        charts[i].subscribeCrosshairMove(param => {
            if (_csSync) return; _csSync = true;
            if (!param.time || !param.point || param.point.x < 0 || param.point.y < 0) {
                for (let j = 0; j < charts.length; j++) { if (j !== i) charts[j].clearCrosshairPosition(); }
            } else {
                for (let j = 0; j < charts.length; j++) {
                    if (j !== i && firstSeries[j]) charts[j].setCrosshairPosition(firstSeries[j].coordinateToPrice(param.point.y) ?? 0, param.time, firstSeries[j]);
                }
            }
            _csSync = false;
        });
    }
    window.addEventListener('resize', () => {
        charts[0].timeScale().fitContent();
        const r = charts[0].timeScale().getVisibleRange();
        if (r) for (let j = 1; j < charts.length; j++) charts[j].timeScale().setVisibleRange(r);
    });"""


def _js_shade_opts(scale_id, line_color="transparent", top_color="transparent", bottom_color="transparent"):
    return (f'priceScaleId:"{scale_id}",lineWidth:1,lineColor:"{line_color}",lineType:2,'
            f'topColor:"{top_color}",bottomColor:"{bottom_color}",'
            f'crosshairMarkerVisible:false,pointMarkersVisible:false,'
            f'lastValueVisible:false,priceLineVisible:false')


def _js_equity_block(eq_pos_expr, thr_params, rets_json, oc_json, is_moo, bh_json,
                     eq_pane_idx, bh_series_var, is_dark, init_args, has_short=True):
    """Generate the JS for equity computation, stats overlay, and initial call."""
    stats_bg = 'rgba(10,10,26,0.62)' if is_dark else 'rgba(255,255,255,0.68)'
    lbl_c = 'rgba(255,255,255,0.55)' if is_dark else 'rgba(0,0,0,0.45)'
    val_c = 'rgba(255,255,255,0.92)' if is_dark else 'rgba(0,0,0,0.88)'
    hdr_c = 'rgba(255,255,255,0.4)' if is_dark else 'rgba(0,0,0,0.35)'

    short_vars = "nShort=0," if has_short else ""
    short_track = "if (pos === -1) nShort++;" if has_short else ""
    short_stats = (
        "res.shortPct > 0 ? ['  Long', _fmtPct(res.longPct, false)] : null,"
        "res.shortPct > 0 ? ['  Short', _fmtPct(res.shortPct, false)] : null,"
    ) if has_short else ""
    short_ret = ",longPct:nLong/L,shortPct:nShort/L" if has_short else ""

    return f"""
    const _rets = {rets_json};
    const _ocRets = {oc_json};
    const _isMOO = {json.dumps(is_moo)};
    const _bhRets = {bh_json};
    const _eqPaneIdx = {eq_pane_idx};
    const _eqSeries = _eqPaneIdx >= 0 ? firstSeries[_eqPaneIdx] : null;
    const _bhSeries = _eqPaneIdx >= 0 ? (typeof {bh_series_var} !== 'undefined' ? {bh_series_var} : null) : null;
    let _statsEl=null, _statsTblEl=null, _statsCollapsed=false;
    if (_eqPaneIdx >= 0) {{
        const _epDiv = document.getElementById('pane' + _eqPaneIdx);
        _statsEl = document.createElement('div');
        _statsEl.style.cssText = 'position:absolute;top:8px;left:8px;z-index:6;background:{stats_bg};backdrop-filter:blur(18px) saturate(180%);-webkit-backdrop-filter:blur(18px) saturate(180%);border:1px solid rgba(255,255,255,0.10);border-radius:10px;padding:6px 12px;pointer-events:auto;cursor:pointer;user-select:none;font:11px/1.6 \\'SF Mono\\',\\'Consolas\\',monospace;';
        const _statsHdr = document.createElement('div');
        _statsHdr.style.cssText = 'color:{hdr_c};font:9px/1.8 \\'SF Mono\\',monospace;text-align:right;letter-spacing:0.5px';
        _statsHdr.textContent = '\\u25be STATS';
        _statsTblEl = document.createElement('table');
        _statsTblEl.style.cssText = 'border-collapse:collapse';
        _statsEl.appendChild(_statsHdr); _statsEl.appendChild(_statsTblEl);
        _statsEl.addEventListener('click', function() {{
            _statsCollapsed = !_statsCollapsed;
            _statsTblEl.style.display = _statsCollapsed ? 'none' : 'table';
            _statsHdr.textContent = _statsCollapsed ? '\\u25b8 STATS' : '\\u25be STATS';
        }});
        if (_epDiv) {{ _epDiv.style.position='relative'; _epDiv.appendChild(_statsEl); }}
    }}
    function _buildEquity({thr_params}) {{
        let pos=0,prevPos=0,eq=100,bh=100;
        if (_carryIn && _td.length > 0) {{
            const _initSig = _td[0] ? _td[0].value : 0;
            const sig = _initSig;
            prevPos = {eq_pos_expr};
        }}
        const eqData=[],bhData=[];
        let n=0,sumR=0,sumR2=0,peak=100,maxDD=0,wins=0,tradesN=0,nLong=0,{short_vars}_coldCtr=0;
        const hasBh = _bhRets.length === _rets.length;
        const L = _rets.length;
        for (let i = 0; i < L; i++) {{
            const sig = _td[i] ? _td[i].value : 0;
            pos = {eq_pos_expr};
            const inClip = !_eqClipStart || _rets[i].time >= _eqClipStart;
            if (!_carryIn && inClip && _coldCtr < _signalDelay) {{ pos = 0; _coldCtr++; }}
            const isEntry = pos > 0 && prevPos <= 0;
            const baseRet = (_isMOO && isEntry && _ocRets.length > 0) ? _ocRets[i].ret : _rets[i].ret;
            const r = baseRet * pos;
            if (inClip) {{
                eq *= (1 + r);
                if (hasBh) bh *= (1 + _bhRets[i].ret);
                eqData.push({{time: _rets[i].time, value: eq}});
                if (hasBh) bhData.push({{time: _bhRets[i].time, value: bh}});
                if (pos !== 0) {{ sumR += r; sumR2 += r*r; n++; if(r>0) wins++; tradesN++; }}
                if (pos === 1) nLong++;{short_track}
                if (eq > peak) peak = eq;
                const dd = (eq - peak) / peak;
                if (dd < maxDD) maxDD = dd;
            }}
            prevPos = pos;
        }}
        const wr = tradesN > 0 ? wins/tradesN : 0;
        const timeMkt = n / L;
        const totalRet = eqData[eqData.length-1].value / 100 - 1;
        const bhRet = hasBh ? bhData[bhData.length-1].value / 100 - 1 : null;
        const nYears = L / 252;
        const cagr = Math.pow(eqData[eqData.length-1].value/100, 1/nYears) - 1;
        const annV = L > 1 ? Math.sqrt((sumR2 - sumR*sumR/L) / (L-1) * 252) : 1;
        const sharpe = annV > 0 ? cagr / annV : 0;
        return {{eqData,bhData,totalRet,bhRet,cagr,sharpe,maxDD,wr,timeMkt{short_ret}}};
    }}
    function _fmtPct(v,showSign=true) {{ return (showSign && v>0?'+':'') + (v*100).toFixed(1)+'%'; }}
    function _updateStats(res) {{
        if (!_statsEl || !_statsTblEl) return;
        const lbl = 'color:{lbl_c}';
        const val = 'color:{val_c};font-weight:600';
        const rows = [
            ['Return', _fmtPct(res.totalRet)],
            res.bhRet !== null ? ['B&H', _fmtPct(res.bhRet)] : null,
            ['CAGR',   _fmtPct(res.cagr)],
            ['Sharpe', res.sharpe.toFixed(2)],
            ['Max DD', _fmtPct(res.maxDD, false)],
            ['Win Rate', _fmtPct(res.wr, false)],
            ['In Mkt',  _fmtPct(res.timeMkt, false)],
            {short_stats}
        ].filter(Boolean);
        _statsTblEl.innerHTML = rows.map(([k,v]) => `<tr><td style="${{lbl}};padding:1px 8px 1px 0;white-space:nowrap">${{k}}</td><td style="${{val}};text-align:right">${{v}}</td></tr>`).join('');
    }}
    const _eq0 = _buildEquity({init_args});
    if (_eqSeries) _eqSeries.setData(_eq0.eqData);
    if (_bhSeries && _eq0.bhData.length) _bhSeries.setData(_eq0.bhData);
    _updateStats(_eq0);"""


# ── MaskedArea plugin (for band modes) ────────────────────────────────────────
_JS_MASKED_AREA_PLUGIN = """
    function _createMaskedAreaPlugin() {
        let _d = null, _o = {};
        return {
            defaultOptions() { return {
                activeLineColor: _tBuy,
                activeTopColor: 'rgba(38,166,154,0.28)', activeBottomColor: 'rgba(38,166,154,0.05)',
                inactiveLineColor: 'rgba(150,150,150,0.55)',
                inactiveTopColor: 'rgba(150,150,150,0.18)', inactiveBottomColor: 'rgba(150,150,150,0.02)',
                lineWidth: 3,
            }; },
            renderer() {
                return { draw(target, priceConverter) {
                    target.useBitmapCoordinateSpace(({context: ctx, bitmapSize, horizontalPixelRatio: hpx, verticalPixelRatio: vpx}) => {
                        if (!_d || !_d.bars.length) return;
                        const o = _o, pts = [];
                        for (const bar of _d.bars) {
                            const od = bar.originalData;
                            if (od.value == null) continue;
                            const y = priceConverter(od.value);
                            if (y == null) continue;
                            pts.push({x: Math.round(bar.x * hpx), y: Math.round(y * vpx), a: od.active !== false});
                        }
                        if (!pts.length) return;
                        const segs = []; let cur = {a: pts[0].a, p: [pts[0]]};
                        for (let i = 1; i < pts.length; i++) {
                            if (pts[i].a !== cur.a) { cur.p.push(pts[i]); segs.push(cur); cur = {a: pts[i].a, p: [pts[i]]}; }
                            else { cur.p.push(pts[i]); }
                        }
                        segs.push(cur);
                        const bottom = bitmapSize.height;
                        for (const s of segs) {
                            if (s.p.length < 2) continue;
                            const tC = s.a ? o.activeTopColor : o.inactiveTopColor;
                            const bC = s.a ? o.activeBottomColor : o.inactiveBottomColor;
                            const lC = s.a ? o.activeLineColor : o.inactiveLineColor;
                            const gr = ctx.createLinearGradient(0, 0, 0, bottom);
                            gr.addColorStop(0, tC); gr.addColorStop(1, bC);
                            ctx.beginPath(); ctx.moveTo(s.p[0].x, s.p[0].y);
                            for (let j = 1; j < s.p.length; j++) ctx.lineTo(s.p[j].x, s.p[j].y);
                            ctx.lineTo(s.p[s.p.length-1].x, bottom); ctx.lineTo(s.p[0].x, bottom);
                            ctx.closePath(); ctx.fillStyle = gr; ctx.fill();
                            ctx.beginPath(); ctx.moveTo(s.p[0].x, s.p[0].y);
                            for (let j = 1; j < s.p.length; j++) ctx.lineTo(s.p[j].x, s.p[j].y);
                            ctx.strokeStyle = lC; ctx.lineWidth = Math.max(1, (o.lineWidth || 2) * hpx); ctx.stroke();
                        }
                    });
                }};
            },
            priceValueBuilder(d) { return [d.value]; },
            isWhitespace(d) { return d.value == null; },
            update(data, options) { _d = data; _o = options; }
        };
    }"""


class Dashboard:
    """Vertically stacked, time-synced chart panes."""

    @staticmethod
    def _normalize_execution(execution: Union[int, str]) -> Union[int, str]:
        if isinstance(execution, int):
            if execution < 0:
                raise ValueError(f"execution must be a non-negative integer, got {execution}.")
            return execution
        if isinstance(execution, str):
            _exec = execution.upper()
            if _exec == "NM":
                _w.warn("execution='NM' is deprecated; use execution='NO' (next open).", UserWarning, stacklevel=3)
                return "NO"
            if _exec == "NO":
                return "NO"
        raise ValueError(f"execution must be a non-negative integer or 'NO', got {execution!r}.")

    @staticmethod
    def _execution_badge_text(execution: Union[int, str]) -> str:
        _MAP = {"NO": "EXEC NO (NEXT OPEN)", 0: "EXEC 0 (SAME BAR)", 1: "EXEC 1 (NEXT CLOSE)"}
        return _MAP.get(execution, f"EXEC {execution}")

    def __init__(self, panes=None, theme=None, titles=None, gap=2, logo=True, execution=1, show_execution=None):
        self._panes: List[Chart] = panes or []
        self._titles: List[str] = titles or []
        self._logo = logo
        self._show_execution = show_execution  # None = auto (show only if threshold_control used)
        self._theme_explicit = theme is not None
        if theme:
            self._theme_name = theme.lower()
        elif self._panes:
            self._theme_name = self._panes[0]._theme_name
        else:
            self._theme_name = "dark"
        self._theme = THEMES.get(self._theme_name, THEMES["dark"])
        self._gap = gap
        if self._theme_explicit:
            for pane in self._panes:
                pane._theme_name = self._theme_name
                pane._theme = self._theme
        self._execution: Union[int, str] = self._normalize_execution(execution)
        self._threshold_config = None
        self._bg_image_config: Optional[Dict] = None

    @property
    def execution(self): return self._execution

    def set_execution(self, execution):
        self._execution = self._normalize_execution(execution)
        return self

    @property
    def theme(self): return self._theme_name

    @theme.setter
    def theme(self, value):
        self._theme_name = value.lower()
        self._theme = THEMES.get(self._theme_name, THEMES["dark"])
        self._theme_explicit = True
        for pane in self._panes:
            pane._theme_name = self._theme_name
            pane._theme = self._theme

    def add(self, pane, title=None):
        """Append a chart pane with an optional title."""
        if self._theme_explicit:
            pane._theme_name = self._theme_name
            pane._theme = self._theme
        self._panes.append(pane)
        if title:
            while len(self._titles) < len(self._panes) - 1:
                self._titles.append("")
            self._titles.append(title)
        return self

    # ── threshold_control ─────────────────────────────────────────────────

    def threshold_control(
        self, df, df2=None, signal_mode="above", threshold=0.0,
        min_val=-0.05, max_val=0.05, step=0.001,
        value_col=None, value_col2=None, price_pane=0,
        buy_color=None, sell_color=None,
        daily_returns=None, bh_returns=None, equity_pane=None,
        invert=False, threshold2=None,
        execution=None, open_returns=None, signal_delay=0,
        prices=None, strategy_color="#4fc3f7", bh_color="#555555",
        equity_base=1.0, equity_clip_start=None, carry_in=True,
    ) -> "Dashboard":
        """Add an interactive threshold slider.

        Parameters
        ----------
        df : Signal DataFrame.
        signal_mode : >=, >, <=, <, above, below, rising, crossover, within, outside
        execution : 0 (same bar), 1 (MOC, default), N (lag-N), 'NO' (next open / MOO)
        prices : DataFrame with open + close → auto-derives returns.
        equity_clip_start : ISO date to start equity from.
        carry_in : Assume position was already established before window.
        """
        if invert and signal_mode == "above":
            signal_mode = "below"

        # ── Auto-derive returns from OHLCV prices ────────────────────────
        _prices_cc: Optional[pd.Series] = None
        _prices_oc: Optional[pd.Series] = None
        if prices is not None:
            _pc = prices["close"] if "close" in prices.columns else prices.iloc[:, -1]
            _po = prices["open"] if "open" in prices.columns else None
            _prices_cc_raw = _pc.pct_change()
            _prices_oc_raw = (_pc / _po - 1) if _po is not None else None
            _sig_dt = pd.api.types.is_datetime64_any_dtype(df.index)
            _px_dt = pd.api.types.is_datetime64_any_dtype(_pc.index)
            if _sig_dt and _px_dt:
                _prices_cc = _prices_cc_raw.reindex(df.index).fillna(0)
                _prices_oc = _prices_oc_raw.reindex(df.index).fillna(0) if _prices_oc_raw is not None else None
            else:
                _prices_cc = _prices_cc_raw.fillna(0)
                _prices_oc = _prices_oc_raw.fillna(0) if _prices_oc_raw is not None else None
            if daily_returns is None: daily_returns = _prices_cc
            if open_returns is None and _prices_oc is not None: open_returns = _prices_oc
            if bh_returns is None: bh_returns = _prices_cc

        # ── Resolve execution mode ───────────────────────────────────────
        if execution is None:
            execution = self._execution
        else:
            execution = self._normalize_execution(execution)
            self._execution = execution

        if execution == "NO":
            _signal_delay, _exec = 1, "NO"
            if open_returns is None:
                raise ValueError("execution='NO' (MOO) requires open_returns or prices with 'open' column.")
            _cov = open_returns.replace(0, float("nan")).notna().mean()
            if _cov < 0.95:
                _w.warn(f"execution='NO': open_returns coverage is only {_cov:.1%}; missing bars earn 0.", UserWarning, stacklevel=2)
            daily_returns = _prices_cc if _prices_cc is not None else open_returns
        else:
            _signal_delay = execution + 1
            _exec = execution
        signal_delay = _signal_delay

        # ── Auto-inject equity line ──────────────────────────────────────
        if prices is not None and equity_pane is not None and equity_pane < len(self._panes):
            _cc, _oc = _prices_cc, _prices_oc
            if _cc is not None:
                _sig_vals = self._extract_signal_values(df, value_col)
                _gate = self._apply_gate(_sig_vals, signal_mode, threshold)
                _oc_arg = _oc if _oc is not None else _cc
                _r_init = Chart.apply_execution(_gate, _cc, execution=execution, open_returns=_oc_arg, carry_in=carry_in)
                _base = float(equity_base)
                _nav_str = _base * (1 + _r_init).cumprod()
                _nav_bh = _base * (1 + _cc).cumprod()
                if equity_clip_start:
                    _nav_str = _nav_str.loc[equity_clip_start:]
                    _nav_bh = _nav_bh.loc[equity_clip_start:]
                    _nav_str = _nav_str / _nav_str.iloc[0] * _base
                    _nav_bh = _nav_bh / _nav_bh.iloc[0] * _base
                _ep = self._panes[equity_pane]
                _ep.line(_nav_str.to_frame("value"), name="strategy", color=strategy_color)
                _ep.line(_nav_bh.to_frame("value"), name="B&H", color=bh_color)

        # ── Parse signal DataFrames ──────────────────────────────────────
        data = self._parse_df(df, value_col)
        data2 = self._parse_df(df2, value_col2) if (df2 is not None and signal_mode == "crossover") else None

        # ── Apply signal delay (pre-shift) ───────────────────────────────
        if signal_delay > 0 and len(data) > signal_delay:
            _vals = [r["value"] for r in data]
            _times = [r["time"] for r in data]
            _shifted = ([_vals[0]] * signal_delay + _vals[:-signal_delay]) if carry_in else ([None] * signal_delay + _vals[:-signal_delay])
            data = [{"time": t, "value": (v if v is not None else 0.0)} for t, v in zip(_times, _shifted)]

        # ── Build return data arrays ─────────────────────────────────────
        up_clr = buy_color or self._theme.get("candlestick", {}).get("upColor", "#26a69a")
        dn_clr = sell_color or self._theme.get("candlestick", {}).get("downColor", "#ef5350")
        decimals = max(0, -int(math.floor(math.log10(step)))) if step > 0 else 3

        times = [r["time"] for r in data]
        ret_data = self._build_ret_data(daily_returns, times) if daily_returns is not None else None
        bh_data = self._build_ret_data(bh_returns, times) if bh_returns is not None else None
        oc_data = self._build_ret_data(open_returns, times) if _exec == "NO" and open_returns is not None else None

        _pad = step * 3 if step > 0 else (max_val - min_val) * 0.02
        min_val = round(min_val - _pad, decimals)
        max_val = round(max_val + _pad, decimals)

        self._threshold_config = {
            "data": data, "data2": data2, "signal_mode": signal_mode,
            "threshold": threshold, "min_val": min_val, "max_val": max_val,
            "step": step, "decimals": decimals, "price_pane": price_pane,
            "buy_color": up_clr, "sell_color": dn_clr,
            "ret_data": ret_data, "bh_data": bh_data, "equity_pane": equity_pane,
            "threshold2": threshold2, "signal_delay": signal_delay,
            "execution": _exec, "oc_data": oc_data,
            "equity_clip_start": equity_clip_start, "carry_in": carry_in,
        }
        return self

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _extract_signal_values(df, value_col):
        _stc = Chart._detect_time_col(df)
        if _stc == "__index__":
            return df.iloc[:, 0] if value_col is None else df[value_col]
        _vc = value_col
        if not _vc:
            for _n in ("value", "close", "Close", "price", "Price"):
                if _n in df.columns and _n != _stc:
                    _vc = _n; break
            if not _vc:
                _vc = [c for c in df.columns if c != _stc][0]
        return df[_vc]

    @staticmethod
    def _apply_gate(sig_vals, signal_mode, threshold):
        sm = signal_mode
        if sm in (">=", "above"):  return (sig_vals >= threshold).astype(float)
        if sm == ">":              return (sig_vals > threshold).astype(float)
        if sm in ("<=", "below"):  return (sig_vals <= threshold).astype(float)
        if sm == "<":              return (sig_vals < threshold).astype(float)
        return (sig_vals >= threshold).astype(float)

    @staticmethod
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
                    vc = name; break
        if not vc:
            for col in frame.columns:
                if col != "time" and pd.api.types.is_numeric_dtype(frame[col]):
                    vc = col; break
        return frame[["time", vc]].rename(columns={vc: "value"}).dropna(subset=["value"]).to_dict("records")

    @staticmethod
    def _build_ret_data(ret_series, times):
        ret_series = ret_series.fillna(0)
        if hasattr(ret_series.index, "strftime"):
            lut = {d.strftime("%Y-%m-%d"): float(v) for d, v in zip(ret_series.index, ret_series.values)}
            return [{"time": t, "ret": lut.get(t, 0.0)} for t in times]
        if pd.api.types.is_integer_dtype(ret_series.index):
            vals = ret_series.values
            return [{"time": times[i], "ret": float(vals[i]) if i < len(vals) else 0.0} for i in range(len(times))]
        lut = {str(d)[:10]: float(v) for d, v in zip(ret_series.index, ret_series.values)}
        return [{"time": t, "ret": lut.get(t, 0.0)} for t in times]

    # ── Background image ─────────────────────────────────────────────────

    def background_image(self, url, blur=0, tint="rgba(6,6,20,0.40)",
                         glass_blur=16, glass_tint="rgba(10,10,26,0.55)"):
        """Set a custom background image with a frosted-glass panel."""
        self._bg_image_config = {"url": url, "blur": blur, "tint": tint,
                                 "glass_blur": glass_blur, "glass_tint": glass_tint}
        return self

    @property
    def total_height(self):
        title_h = sum(24 for i in range(len(self._panes)) if self._get_title(i))
        slider_h = 36 if self._threshold_config else 0
        _sm_labels = set()
        for pane in self._panes:
            for sc in getattr(pane, "_smoothing_configs", []):
                _sm_labels.add(sc.get("label", ""))
        slider_h += len(_sm_labels) * 36
        return sum(p._height for p in self._panes) + self._gap * max(0, len(self._panes) - 1) + title_h + slider_h

    def _get_title(self, idx):
        return self._titles[idx] if idx < len(self._titles) else ""

    # ── Build HTML ────────────────────────────────────────────────────────

    def _build_html(self) -> str:
        if not self._panes:
            return "<html><body>No panes</body></html>"

        bg = self._theme.get("chart", {}).get("layout", {}).get("background", {}).get("color", "#1e1e1e")
        is_dark = self._theme_name in ("dark", "midnight", "distfit")
        title_color = "rgba(255,255,255,0.55)" if is_dark else "rgba(0,0,0,0.55)"
        title_font = "11px -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"

        _logo_invert = ""
        if bg.startswith("#") and len(bg) >= 7:
            _bg_hex = bg.lstrip("#")
            _r, _g, _b = (int(_bg_hex[i:i+2], 16) for i in (0, 2, 4))
            _logo_invert = "filter:invert(1);" if (_r * 0.299 + _g * 0.587 + _b * 0.114) > 150 else ""
        elif self._theme.get("background_css") or self._theme.get("background_svg"):
            if self._theme_name not in ("dark", "midnight", "distfit"):
                _logo_invert = "filter:invert(1);"

        lc_js = _get_lc_js()
        pane_divs, pane_scripts = [], []
        _pane_smoothing_configs = []
        n_panes = len(self._panes)
        _show = self._show_execution if self._show_execution is not None else bool(self._threshold_config)
        _exec_hdr = self._execution_badge_text(self._execution) if _show else ""

        for i, pane in enumerate(self._panes):
            div_id, chart_var, prefix = f"pane{i}", f"chart{i}", f"p{i}_"
            is_last = (i == n_panes - 1)

            title = self._get_title(i)
            if title:
                _right = (f'<span style="opacity:0.92;font:{title_font};color:{title_color};'
                          f'letter-spacing:0.3px;text-transform:uppercase">{_html.escape(_exec_hdr)}</span>'
                          if i == 0 else "")
                pane_divs.append(
                    f'<div style="color:{title_color};font:{title_font};padding:6px 8px 2px 8px;'
                    f'letter-spacing:0.3px;text-transform:uppercase;display:flex;align-items:center;'
                    f'justify-content:space-between;gap:10px"><span>{_html.escape(title)}</span>{_right}</div>')

            _stats_html = ""
            if getattr(pane, "_stats_legend", None):
                _sl = pane._stats_legend
                _pos = _sl.get("position", "top-left")
                _v = "top:8px" if "top" in _pos else "bottom:8px"
                _h = "left:8px" if "left" in _pos else "right:8px"
                _rows = "".join(
                    f'<tr><td style="color:rgba(255,255,255,0.55);padding-right:10px;white-space:nowrap">'
                    f'{_html.escape(str(k))}</td><td style="color:rgba(255,255,255,0.92);font-weight:600;'
                    f'text-align:right">{_html.escape(str(v))}</td></tr>'
                    for k, v in _sl["metrics"].items())
                _stats_html = (
                    f'<div onclick="var t=this.querySelector(\'table\');'
                    f't.style.display=t.style.display===\'none\'?\'table\':\'none\';'
                    f'this.querySelector(\'span\').textContent=t.style.display===\'none\'?\'▸ STATS\':\'▾ STATS\'" '
                    f'style="position:absolute;{_v};{_h};z-index:6;background:rgba(10,10,26,0.62);'
                    f'backdrop-filter:blur(18px) saturate(180%);-webkit-backdrop-filter:blur(18px) saturate(180%);'
                    f'border:1px solid rgba(255,255,255,0.10);border-radius:10px;padding:6px 12px;'
                    f'pointer-events:auto;cursor:pointer;user-select:none;">'
                    f'<div style="color:rgba(255,255,255,0.4);font:9px/1.8 \'SF Mono\',monospace;'
                    f'text-align:right;letter-spacing:0.5px"><span>▾ STATS</span></div>'
                    f'<table style="border-collapse:collapse;font:11px/1.6 \'SF Mono\',monospace">{_rows}</table></div>')

            pane_divs.append(f'<div id="{div_id}" style="position:relative;width:100%;height:{pane._height}px;margin-bottom:{self._gap}px">{_stats_html}</div>')

            chart_opts = pane._build_chart_options()
            chart_opts.pop("autoSize", None)
            if self._bg_image_config:
                lo = chart_opts.get("layout", {})
                lo["background"] = {"type": "solid", "color": "rgba(0,0,0,0)"}
                chart_opts["layout"] = lo
            ts = chart_opts.get("timeScale", {})
            ts["visible"] = is_last
            chart_opts["timeScale"] = ts
            opts = json.dumps(chart_opts, separators=(",", ":"))
            fmt_js = pane._get_formatter_js()
            if fmt_js:
                opts = pane._inject_formatter_into_opts(opts, fmt_js)
            series_js = pane._build_series_js(var_prefix=prefix, chart_var=chart_var)
            first_var = f"{prefix}s0" if pane._series else "null"
            pane_scripts.append(
                f"const {chart_var} = LightweightCharts.createChart(document.getElementById('{div_id}'), {opts});\n"
                f"    charts.push({chart_var});\n    {series_js}\n    firstSeries.push({first_var});")

            n_pane_series = len(pane._series)
            for sc_idx, sc in enumerate(getattr(pane, "_smoothing_configs", [])):
                target_idx = sc["series_index"] if sc["series_index"] >= 0 else n_pane_series + sc["series_index"]
                _pane_smoothing_configs.append({**sc, "_svar": f"{prefix}s{target_idx}", "_pane": i, "_local_idx": sc_idx})

        divs_html = "\n".join(pane_divs)
        scripts_body = "\n    ".join(pane_scripts)

        # ── Threshold slider JS ──────────────────────────────────────────
        slider_html, slider_js = "", ""
        if self._threshold_config:
            slider_js = self._build_threshold_js(divs_html, is_dark)
            # divs_html may have been modified (slider HTML appended) — stored in _slider_divs_extra
            divs_html += getattr(self, "_slider_divs_extra", "")

        # ── Smoothing sliders ────────────────────────────────────────────
        sm_html, sm_js = self._build_smoothing_js(_pane_smoothing_configs, is_dark)
        if sm_html:
            divs_html += "\n" + sm_html
        if sm_js:
            slider_js += "\n    " + sm_js.replace("\n", "\n    ")

        # ── Background ───────────────────────────────────────────────────
        custom_bg_css = self._theme.get("background_css", "")
        bg_css = custom_bg_css if custom_bg_css else f"background:{bg};"
        bg_svg = self._theme.get("background_svg", "")
        _glass_open, _glass_close = "", ""
        _bgi = self._bg_image_config
        if _bgi:
            _url_safe = _bgi["url"].replace('"', "%22")
            bg_css = f'background:url("{_url_safe}") center/cover no-repeat;'
            _blur_div = (f'<div style="position:absolute;inset:0;z-index:0;'
                         f'backdrop-filter:blur({_bgi["blur"]}px);-webkit-backdrop-filter:blur({_bgi["blur"]}px);"></div>'
                         if _bgi["blur"] > 0 else "")
            _glass_open = (f'{_blur_div}<div id="bg-tint" style="position:absolute;inset:0;z-index:1;'
                           f'background:{_bgi["tint"]}"></div>'
                           f'<div id="glass" style="position:absolute;inset:0;z-index:2;'
                           f'backdrop-filter:blur({_bgi["glass_blur"]}px) saturate(180%);'
                           f'-webkit-backdrop-filter:blur({_bgi["glass_blur"]}px) saturate(180%);'
                           f'background:{_bgi["glass_tint"]};border-radius:12px;overflow:hidden;">')
            _glass_close = "</div>"

        _logo = (f'<img id="signum-logo" src="data:image/svg+xml;base64,{_LOGO_B64}" '
                 f'width="30" height="30" alt="Signum">') if self._logo else ''

        return f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<script>{lc_js}</script>
<style>*{{margin:0;padding:0;box-sizing:border-box}}body{{{bg_css}overflow-y:auto;overflow-x:hidden;position:relative;border-radius:12px;padding-bottom:36px}}
#signum-logo{{position:absolute;right:12px;bottom:6px;z-index:5;opacity:0.7;pointer-events:none;{_logo_invert}}}</style>
</head><body>{bg_svg}{_glass_open}{divs_html}{_glass_close}{_logo}
<script>
    const charts = [];
    const firstSeries = [];
    {scripts_body}
    {_JS_SYNC}
    {slider_js}
</script></body></html>"""

    # ── Threshold JS builder ─────────────────────────────────────────────

    def _build_threshold_js(self, divs_html, is_dark):
        tc = self._threshold_config
        dec = tc["decimals"]
        lbl_c = "rgba(255,255,255,0.88)" if is_dark else "rgba(0,0,0,0.78)"
        cnt_c = "rgba(255,255,255,0.48)" if is_dark else "rgba(0,0,0,0.42)"
        thr0 = tc["threshold"]
        pp = tc["price_pane"]
        price_chart, price_s0 = f"chart{pp}", f"p{pp}_s0"
        tdata_json = json.dumps(tc["data"], separators=(",", ":"))

        bc = tc["buy_color"].lstrip("#")
        sr, sg, sb = int(bc[0:2], 16), int(bc[2:4], 16), int(bc[4:6], 16)
        shade_fill = f"rgba({sr},{sg},{sb},0.18)"

        # Resolve signal mode
        sm = _OP_ALIAS.get(tc.get("signal_mode", "above"), tc.get("signal_mode", "above"))

        # Handle crossover specially (needs data2)
        if sm == "crossover":
            _data2_json = json.dumps(tc.get("data2") or [], separators=(",", ":"))
            mc = dict(
                long_idx="_spread[i] >= thr", short_idx="false",
                long_pos="_spread[i] >= thr", short_pos="false",
                eq_pos="_spread[i] >= thr ? 1 : 0",
                precompute=f"const _td2 = {_data2_json};\n    const _spread = _td.map((d,i) => d.value - (_td2[i] ? _td2[i].value : 0));",
                update_baseline=True, invert=False, neg_line=False,
                lbl_js="spread\\u00a0\\u2265\\u00a0", lbl_py="spread\u00a0\u2265\u00a0",
            )
        else:
            mc = _SIGNAL_MODES.get(sm, _SIGNAL_MODES["above"])

        is_band = mc.get("is_band", False)

        # Colors
        if mc["invert"]:
            top_lc, top_f1, top_f2 = "rgba(120,120,120,0.4)", "rgba(100,100,100,0.06)", "rgba(100,100,100,0.01)"
            bot_lc = tc["buy_color"]
            bot_f1, bot_f2 = f"rgba({sr},{sg},{sb},0.08)", "transparent"
        else:
            top_lc, top_f1 = tc["buy_color"], f"rgba({sr},{sg},{sb},0.28)"
            top_f2 = f"rgba({sr},{sg},{sb},0.05)"
            bot_lc, bot_f1, bot_f2 = "rgba(100,100,100,0.28)", "rgba(100,100,100,0.01)", "rgba(100,100,100,0.06)"

        # Common JS constants
        common_js = f"""
    const _td = {tdata_json};
    const _tBuy = "{tc['buy_color']}";
    const _tSell = "{tc['sell_color']}";
    const _tDec = {dec};
    const _eqClipStart = {json.dumps(tc.get('equity_clip_start'))};
    const _carryIn = {json.dumps(bool(tc.get('carry_in', True)))};
    const _signalDelay = {tc.get('signal_delay', 0)};"""
        if mc["precompute"]:
            common_js += f"\n    {mc['precompute']}"

        # Shade series
        shade_opts = _js_shade_opts("_thShade", "transparent", shade_fill, "transparent")
        shade_js = f"""
    const _shadeSeries = {price_chart}.addSeries(LightweightCharts.AreaSeries, {{{shade_opts}}});
    {price_chart}.priceScale("_thShade").applyOptions({{visible:false,scaleMargins:{{top:0,bottom:0}}}});"""

        if is_band:
            return self._build_band_js(tc, mc, common_js, shade_js, price_chart, price_s0,
                                       thr0, dec, lbl_c, cnt_c, sr, sg, sb, is_dark, pp)
        else:
            return self._build_single_js(tc, mc, common_js, shade_js, price_chart, price_s0,
                                         thr0, dec, lbl_c, cnt_c, top_lc, top_f1, top_f2,
                                         bot_lc, bot_f1, bot_f2, is_dark)

    def _build_single_js(self, tc, mc, common_js, shade_js, price_chart, price_s0,
                          thr0, dec, lbl_c, cnt_c, top_lc, top_f1, top_f2,
                          bot_lc, bot_f1, bot_f2, is_dark):
        thr_lbl_init = f"{mc['lbl_py']}{thr0:.{dec}f}"

        # Slider HTML
        self._slider_divs_extra = (
            f'\n<div id="th-bar" style="display:flex;align-items:center;justify-content:center;'
            f'gap:10px;background:transparent;padding:6px 16px;white-space:nowrap;height:36px">'
            f'<span id="th-label" style="color:{lbl_c};font:11px/1 \'SF Mono\',\'Consolas\',monospace;'
            f'min-width:72px">{thr_lbl_init}</span>'
            f'<input id="th-slider" type="range" min="{tc["min_val"]}" max="{tc["max_val"]}" '
            f'step="{tc["step"]}" value="{thr0}" style="width:220px;cursor:pointer;accent-color:{tc["buy_color"]}">'
            f'<span id="th-count" style="color:{cnt_c};font:11px/1 sans-serif;min-width:60px;text-align:right"></span></div>')

        # Sell shade series
        sell_shade_opts = _js_shade_opts("_thShadeSell")
        sell_shade_js = f"""
    const _shadeSellSeries = {price_chart}.addSeries(LightweightCharts.AreaSeries, {{{sell_shade_opts}}});
    {price_chart}.priceScale("_thShadeSell").applyOptions({{visible:false,scaleMargins:{{top:0,bottom:0}}}});"""

        # Baseline pane setup
        baseline_panes = [i for i, p in enumerate(self._panes) if any(s.get('type') == 'BaselineSeries' for s in p._series)]
        baseline_setup = ""
        if mc["update_baseline"]:
            baseline_setup = f"""
    const _thresholdLines = [], _thresholdNegLines = [], _sigSeries = [];
    const _baselinePanes = {baseline_panes};
    for (const ci of _baselinePanes) {{
        const fs = firstSeries[ci];
        if (fs) {{ try {{
            fs.applyOptions({{
                baseValue:{{type:"price",price:{thr0}}},
                topLineColor:"{top_lc}",topFillColor1:"{top_f1}",topFillColor2:"{top_f2}",
                bottomLineColor:"{bot_lc}",bottomFillColor1:"{bot_f1}",bottomFillColor2:"{bot_f2}",
            }});
            _sigSeries.push(fs);
            _thresholdLines.push(fs.createPriceLine({{price:{thr0},title:"\\u03b8",color:"{tc['buy_color']}",lineWidth:1,lineStyle:2,axisLabelVisible:true}}));
            {"_thresholdNegLines.push(fs.createPriceLine({price:" + str(-thr0) + ',title:"-\\u03b8",color:"' + tc['sell_color'] + '",lineWidth:1,lineStyle:2,axisLabelVisible:true}));' if mc['neg_line'] else ''}
        }} catch(e) {{}} }}
    }}"""
        else:
            baseline_setup = "\n    const _thresholdLines = [], _thresholdNegLines = [], _sigSeries = [];"

        # JS functions
        long_idx, short_idx = mc["long_idx"], mc["short_idx"]
        long_pos, short_pos, eq_pos = mc["long_pos"], mc["short_pos"], mc["eq_pos"]

        functions_js = f"""
    function _buildShade(thr) {{
        const d = [];
        for (let i = 0; i < _td.length; i++) {{
            d.push({{time: _td[i].time, value: ({long_idx}) ? 1 : 0}});
        }}
        return d;
    }}
    function _buildShadeSell(thr) {{
        const d = [];
        for (let i = 0; i < _td.length; i++) {{
            d.push({{time: _td[i].time, value: ({short_idx}) ? 1 : 0}});
        }}
        return d;
    }}
    function _mkrs(thr) {{
        const m = []; let longOn=false, shortOn=false;
        for (let i = 0; i < _td.length; i++) {{
            const sig = _td[i].value;
            const goLong = {long_pos}, goShort = {short_pos};
            if (goLong && !longOn) {{ m.push({{time:_td[i].time,position:"belowBar",shape:"arrowUp",color:_tBuy,text:""}}); longOn=true; if(shortOn) shortOn=false; }}
            else if (!goLong && longOn) {{ m.push({{time:_td[i].time,position:"aboveBar",shape:"arrowDown",color:_tSell,text:""}}); longOn=false; }}
            if (!goLong && goShort && !shortOn) {{ m.push({{time:_td[i].time,position:"aboveBar",shape:"arrowDown",color:_tSell,text:"S"}}); shortOn=true; }}
            else if (!goShort && shortOn) {{ m.push({{time:_td[i].time,position:"belowBar",shape:"circle",color:"rgba(150,150,150,0.8)",text:"C"}}); shortOn=false; }}
        }}
        return m;
    }}
    const _thP = LightweightCharts.createSeriesMarkers({price_s0}, []);
    _shadeSeries.setData(_buildShade({thr0}));
    _shadeSellSeries.setData(_buildShadeSell({thr0}));"""

        # Equity block
        equity_js = ""
        eq_pane = tc.get('equity_pane')
        if tc.get("ret_data"):
            eq_pane_idx = eq_pane if eq_pane is not None else -1
            bh_var = f"p{eq_pane if eq_pane is not None else 0}_s1"
            equity_js = _js_equity_block(
                eq_pos, "thr",
                json.dumps(tc['ret_data'], separators=(',', ':')),
                json.dumps(tc.get('oc_data') or [], separators=(',', ':')),
                tc.get('execution') == 'NO',
                json.dumps(tc.get('bh_data') or [], separators=(',', ':')),
                eq_pane_idx, bh_var, is_dark, str(thr0), has_short=True)

        # Slider event
        baseline_update = '        for (const ss of _sigSeries) { ss.applyOptions({baseValue:{type:"price",price:thr}}); }' if mc["update_baseline"] else ""
        event_js = f"""
    (function() {{
        const m0 = _mkrs({thr0});
        document.getElementById("th-count").textContent = m0.filter(x=>x.shape==="arrowUp").length + " signals";
    }})();
    document.getElementById("th-slider").addEventListener("input", function() {{
        const thr = parseFloat(this.value);
        document.getElementById("th-label").textContent = "{mc['lbl_js']}" + thr.toFixed(_tDec);
        const m = _mkrs(thr);
        _thP.setMarkers([]);
        _shadeSeries.setData(_buildShade(thr));
        _shadeSellSeries.setData(_buildShadeSell(thr));
        for (const pl of _thresholdLines) {{ pl.applyOptions({{price: thr}}); }}
        for (const pl of _thresholdNegLines) {{ pl.applyOptions({{price: -thr}}); }}
{baseline_update}
        document.getElementById("th-count").textContent = m.filter(x=>x.shape==="arrowUp").length + " signals";
        if (typeof _rets !== 'undefined' && _rets.length) {{
            const eq = _buildEquity(thr);
            if (_eqSeries) _eqSeries.setData(eq.eqData);
            if (_bhSeries && eq.bhData.length) _bhSeries.setData(eq.bhData);
            _updateStats(eq);
            setTimeout(_alignScales, 50);
        }}
    }});"""

        return common_js + shade_js + sell_shade_js + baseline_setup + functions_js + equity_js + event_js

    def _build_band_js(self, tc, mc, common_js, shade_js, price_chart, price_s0,
                        thr0, dec, lbl_c, cnt_c, sr, sg, sb, is_dark, pp):
        _thr2 = tc.get("threshold2")
        if _thr2 is None:
            _thr2 = round(thr0 + (tc["max_val"] - tc["min_val"]) * 0.2, dec)
        _thr2 = max(tc["min_val"], min(tc["max_val"], _thr2))
        shade_cond = mc["shade_cond"]
        band_pos = mc["long_pos"]
        band_eq = mc["eq_pos"]

        # Two-slider HTML
        self._slider_divs_extra = (
            f'\n<div id="th-bar" style="display:flex;align-items:center;justify-content:center;'
            f'gap:8px;background:transparent;padding:6px 16px;white-space:nowrap;height:36px">'
            f'<span id="th-lo-label" style="color:{lbl_c};font:11px/1 \'SF Mono\',\'Consolas\',monospace;'
            f'min-width:76px">\u03b8_lo\u00a0{thr0:.{dec}f}</span>'
            f'<input id="th-lo-slider" type="range" min="{tc["min_val"]}" max="{tc["max_val"]}" '
            f'step="{tc["step"]}" value="{thr0}" style="width:155px;cursor:pointer;accent-color:{tc["buy_color"]}">'
            f'<span id="th-hi-label" style="color:{lbl_c};font:11px/1 \'SF Mono\',\'Consolas\',monospace;'
            f'min-width:76px">\u03b8_hi\u00a0{_thr2:.{dec}f}</span>'
            f'<input id="th-hi-slider" type="range" min="{tc["min_val"]}" max="{tc["max_val"]}" '
            f'step="{tc["step"]}" value="{_thr2}" style="width:155px;cursor:pointer;accent-color:{tc["buy_color"]}">'
            f'<span id="th-count" style="color:{cnt_c};font:11px/1 sans-serif;min-width:60px;text-align:right"></span></div>')

        eq_pane = tc.get('equity_pane')
        eq_pane_idx = eq_pane if eq_pane is not None else -1

        # MaskedArea plugin + setup
        masked_setup = f"""
    const _thresholdLoLines = [], _thresholdHiLines = [], _maskedSeries = [];
    for (let ci = 0; ci < charts.length; ci++) {{
        if (ci === {pp} || ci === {eq_pane_idx}) continue;
        const fs = firstSeries[ci];
        if (fs) {{
            fs.applyOptions({{
                topLineColor:"rgba(0,0,0,0)",bottomLineColor:"rgba(0,0,0,0)",
                topFillColor1:"rgba(0,0,0,0)",topFillColor2:"rgba(0,0,0,0)",
                bottomFillColor1:"rgba(0,0,0,0)",bottomFillColor2:"rgba(0,0,0,0)",
                crosshairMarkerVisible:true,crosshairMarkerRadius:5,
                crosshairMarkerBorderWidth:2,crosshairMarkerBackgroundColor:_tBuy,
            }});
            _thresholdLoLines.push(fs.createPriceLine({{price:{thr0},title:"\\u03b8_lo",color:"#ef5350",lineWidth:1,lineStyle:2,axisLabelVisible:true}}));
            _thresholdHiLines.push(fs.createPriceLine({{price:{_thr2},title:"\\u03b8_hi",color:"{tc['buy_color']}",lineWidth:1,lineStyle:2,axisLabelVisible:true}}));
            _maskedSeries.push(charts[ci].addCustomSeries(_createMaskedAreaPlugin(), {{crosshairMarkerVisible:false}}));
        }}
    }}
    function _buildMask(lo, hi) {{ return _td.map(d => {{ const v = d.value; return {{time:d.time,value:v,active:({shade_cond})}}; }}); }}
    const _msk0 = _buildMask({thr0}, {_thr2});
    for (const s of _maskedSeries) s.setData(_msk0);
    let _activeMap = {{}}; _msk0.forEach(d => {{ _activeMap[d.time] = d.active; }});
    const _sigBaselines = [];
    for (let ci = 0; ci < charts.length; ci++) {{
        if (ci === {pp} || ci === {eq_pane_idx}) continue;
        const fs = firstSeries[ci];
        if (fs) {{
            _sigBaselines.push(fs);
            charts[ci].subscribeCrosshairMove(param => {{
                if (!param.time) return;
                fs.applyOptions({{ crosshairMarkerBackgroundColor: _activeMap[param.time] ? _tBuy : 'rgba(150,150,150,0.7)' }});
            }});
        }}
    }}"""

        functions_js = f"""
    function _buildShade(lo, hi) {{
        const d = [];
        for (let i = 0; i < _td.length; i++) {{
            const v = _td[i].value; const cond = {shade_cond};
            d.push({{time: _td[i].time, value: cond ? 1 : 0}});
        }}
        return d;
    }}
    function _mkrs(lo, hi) {{
        const m = []; let longOn = false;
        for (let i = 0; i < _td.length; i++) {{
            const sig = _td[i].value;
            const goLong = {band_pos};
            if (goLong && !longOn) {{ m.push({{time:_td[i].time,position:"belowBar",shape:"arrowUp",color:_tBuy,text:""}}); longOn=true; }}
            else if (!goLong && longOn) {{ m.push({{time:_td[i].time,position:"aboveBar",shape:"arrowDown",color:_tSell,text:""}}); longOn=false; }}
        }}
        return m;
    }}"""

        equity_js = ""
        if tc.get("ret_data"):
            bh_var = f"p{eq_pane if eq_pane is not None else 0}_s1"
            equity_js = _js_equity_block(
                band_eq, "lo, hi",
                json.dumps(tc['ret_data'], separators=(',', ':')),
                json.dumps(tc.get('oc_data') or [], separators=(',', ':')),
                tc.get('execution') == 'NO',
                json.dumps(tc.get('bh_data') or [], separators=(',', ':')),
                eq_pane_idx, bh_var, is_dark, f"{thr0}, {_thr2}", has_short=False)

        event_js = f"""
    const _thP = LightweightCharts.createSeriesMarkers({price_s0}, []);
    _shadeSeries.setData(_buildShade({thr0}, {_thr2}));
    (function() {{
        const m0 = _mkrs({thr0}, {_thr2});
        document.getElementById("th-count").textContent = m0.filter(x=>x.shape==="arrowUp").length + " signals";
    }})();
    let _thLo = {thr0}, _thHi = {_thr2};
    const _loSlider = document.getElementById('th-lo-slider');
    const _hiSlider = document.getElementById('th-hi-slider');
    function _th_update(lo, hi) {{
        document.getElementById("th-lo-label").textContent = "\\u03b8_lo\\u00a0" + lo.toFixed(_tDec);
        document.getElementById("th-hi-label").textContent = "\\u03b8_hi\\u00a0" + hi.toFixed(_tDec);
        _thP.setMarkers([]);
        _shadeSeries.setData(_buildShade(lo, hi));
        const _msk = _buildMask(lo, hi);
        for (const s of _maskedSeries) s.setData(_msk);
        _activeMap = {{}}; _msk.forEach(d => {{ _activeMap[d.time] = d.active; }});
        for (const pl of _thresholdLoLines) pl.applyOptions({{price: lo}});
        for (const pl of _thresholdHiLines) pl.applyOptions({{price: hi}});
        document.getElementById("th-count").textContent = _mkrs(lo, hi).filter(x=>x.shape==="arrowUp").length + " signals";
        {"if (typeof _rets !== 'undefined' && _rets.length) { const eq = _buildEquity(lo, hi); if (_eqSeries) _eqSeries.setData(eq.eqData); if (_bhSeries && eq.bhData.length) _bhSeries.setData(eq.bhData); _updateStats(eq); }" if tc.get('ret_data') else ""}
    }}
    _loSlider.addEventListener('input', function() {{ _thLo = parseFloat(this.value); if (_thLo > _thHi) {{ _thHi = _thLo; _hiSlider.value = _thHi; }} _th_update(_thLo, _thHi); }});
    _hiSlider.addEventListener('input', function() {{ _thHi = parseFloat(this.value); if (_thHi < _thLo) {{ _thLo = _thHi; _loSlider.value = _thLo; }} _th_update(_thLo, _thHi); }});"""

        return common_js + shade_js + _JS_MASKED_AREA_PLUGIN + masked_setup + functions_js + equity_js + event_js

    # ── Smoothing sliders builder ────────────────────────────────────────

    def _build_smoothing_js(self, configs, is_dark):
        _sm_by_label = _dd(list)
        for sc in configs:
            _sm_by_label[sc["label"]].append(sc)
        if not _sm_by_label:
            return "", ""

        html_parts, js_parts = [], []
        _lbl_c = "rgba(255,255,255,0.88)" if is_dark else "rgba(0,0,0,0.78)"

        for gi, (lbl, group) in enumerate(_sm_by_label.items()):
            sc = group[0]
            acc = sc["color"]
            sid, lid = f"db-sm-slider-{gi}", f"db-sm-label-{gi}"

            if sc["mode"] == "variants":
                keys = sc["variants_keys"]
                init_idx = sc["variants_init"]
                html_parts.append(
                    f'<div style="display:flex;align-items:center;justify-content:center;gap:10px;'
                    f'background:transparent;padding:4px 16px;white-space:nowrap;height:36px">'
                    f'<span id="{lid}" style="color:{_lbl_c};font:11px/1 \'SF Mono\',\'Consolas\',monospace;'
                    f'min-width:72px">{lbl} {keys[init_idx]}</span>'
                    f'<input id="{sid}" type="range" min="0" max="{len(keys)-1}" step="1" value="{init_idx}" '
                    f'style="width:220px;cursor:pointer;accent-color:{acc}">'
                    f'<span style="min-width:60px"></span></div>')
                js_lines = [f"    // Smoothing #{gi} (variants)"]
                for j, g in enumerate(group):
                    did = f"_dbSmVD{gi}_{j}"
                    js_lines.append(f"    const {did} = {json.dumps(g['variants_data'], separators=(',', ':'))};")
                    js_lines.append(f"    {g['_svar']}.setData({did}[{g['variants_init']}]);")
                js_lines.append(f"    const _dbSmK{gi} = {json.dumps(keys, separators=(',', ':'))};")
                js_lines.append(f"    document.getElementById('{sid}').addEventListener('input', function() {{")
                js_lines.append(f"        const idx = parseInt(this.value);")
                js_lines.append(f"        document.getElementById('{lid}').textContent = '{lbl} ' + _dbSmK{gi}[idx];")
                for j, g in enumerate(group):
                    js_lines.append(f"        {g['_svar']}.setData(_dbSmVD{gi}_{j}[idx]);")
                js_lines += ["    });", ""]
                js_parts.append("\n".join(js_lines))
            else:
                win_init = sc["window_init"]
                html_parts.append(
                    f'<div style="display:flex;align-items:center;gap:10px;background:transparent;'
                    f'padding:4px 16px;white-space:nowrap;height:36px">'
                    f'<span id="{lid}" style="color:{_lbl_c};font:11px/1 \'SF Mono\',\'Consolas\',monospace;'
                    f'min-width:100px">{lbl} {win_init}</span>'
                    f'<input id="{sid}" type="range" min="{sc["window_min"]}" max="{sc["window_max"]}" '
                    f'step="{sc["window_step"]}" value="{win_init}" '
                    f'style="width:220px;cursor:pointer;accent-color:{acc}"></div>')
                js_lines = []
                mode = sc["mode"]
                for j, g in enumerate(group):
                    rid = f"_dbSmR{gi}_{j}"
                    if mode == "ema":
                        cfn = (f"function _dbSmC{gi}_{j}(rd,win){{const k=2/(win+1);let e=null;const o=[];"
                               "for(const d of rd){e=e===null?d.value:d.value*k+e*(1-k);o.push({time:d.time,value:e});}return o;}")
                    else:
                        cfn = (f"function _dbSmC{gi}_{j}(rd,win){{const o=[];let s=0,b=[];"
                               "for(const d of rd){b.push(d.value);s+=d.value;"
                               "if(b.length>win){s-=b.shift();}o.push({time:d.time,value:s/b.length});}return o;}")
                    js_lines += [
                        f"    const {rid} = {json.dumps(g['raw_data'], separators=(',', ':'))};",
                        f"    {cfn}",
                        f"    {g['_svar']}.setData(_dbSmC{gi}_{j}({rid},{win_init}));"]
                js_lines += [
                    f"    document.getElementById('{sid}').addEventListener('input', function() {{",
                    f"        const win = parseInt(this.value);",
                    f"        document.getElementById('{lid}').textContent = '{lbl} ' + win;"]
                for j, g in enumerate(group):
                    js_lines.append(f"        {g['_svar']}.setData(_dbSmC{gi}_{j}(_dbSmR{gi}_{j},win));")
                js_lines += ["    });", ""]
                js_parts.append("\n".join(js_lines))

        return "\n".join(html_parts), "\n".join(js_parts)

    # ── Display Methods ───────────────────────────────────────────────────

    def _repr_html_(self) -> str:
        import base64
        chart_html = self._build_html()
        b64 = base64.b64encode(chart_html.encode("utf-8")).decode("ascii")
        h = self.total_height + 40
        uid = f"fd{id(self)}"
        return (
            f'<div id="{uid}" style="width:100%;height:{h}px;border-radius:12px;overflow:hidden;">'
            f'</div><script>(function(){{'
            f'var a=atob("{b64}"),b=new Uint8Array(a.length);'
            f'for(var i=0;i<a.length;i++)b[i]=a.charCodeAt(i);'
            f'var blob=new Blob([b],{{type:"text/html;charset=utf-8"}});'
            f'var url=URL.createObjectURL(blob);'
            f'var f=document.createElement("iframe");'
            f'f.src=url;f.style.width="100%";f.style.height="{h}px";'
            f'f.style.border="none";f.style.borderRadius="12px";'
            f'document.getElementById("{uid}").appendChild(f);'
            f'}})();</script>')

    def show(self):
        try:
            from IPython.display import display, HTML
            display(HTML(self._repr_html_()))
        except ImportError:
            print("IPython not available. Use .save() or .render() instead.")

    def render(self): return self._build_html()

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(self._build_html())

    def to_dash(self, id=None, style=None):
        from dash import html
        default_style = {"width": "100%", "height": f"{self.total_height + 40}px", "border": "none", "borderRadius": "4px"}
        if style: default_style.update(style)
        return html.Iframe(id=id or "forge-dashboard", srcDoc=self._build_html(), style=default_style)

    def to_streamlit(self, height=None):
        import streamlit.components.v1 as components
        components.html(self._build_html(), height=height or self.total_height + 40)
