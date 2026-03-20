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
import html as _html
from typing import Optional, List

import numpy as np
import pandas as pd

from .themes import THEMES
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
        self._theme = THEMES.get(self._theme_name, THEMES["dark"])
        self._width = width
        self._height = height
        self._cols = cols
        self._title = title
        self._logo = logo
        self._panels: List[dict] = []
        self._color_idx = 0

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

    def scatter(
        self,
        x,
        y,
        name: Optional[str] = None,
        color: Optional[str] = None,
        size: int = 3,
    ) -> "StatChart":
        """Add a scatter-plot panel.

        Parameters
        ----------
        x, y : array-like, Series
            Coordinate arrays (same length).
        name : str, optional
            Panel title.
        color : str, optional
            Dot colour (CSS).
        size : int
            Dot radius in pixels (default 3).
        """
        xarr = np.asarray(x, dtype=float)
        yarr = np.asarray(y, dtype=float)
        mask = ~(np.isnan(xarr) | np.isnan(yarr))
        xarr, yarr = xarr[mask], yarr[mask]

        self._panels.append({
            "type": "scatter",
            "name": name or f"Scatter {len(self._panels) + 1}",
            "color": color or self._next_color(),
            "x": [round(float(v), 6) for v in xarr],
            "y": [round(float(v), 6) for v in yarr],
            "size": size,
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
        return self._theme_name in ("dark", "midnight", "distfit")

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

        text_color = "rgba(255,255,255,0.75)" if dark else "rgba(0,0,0,0.65)"
        sub_color  = "rgba(255,255,255,0.45)" if dark else "rgba(0,0,0,0.40)"
        grid_color = "rgba(255,255,255,0.06)" if dark else "rgba(0,0,0,0.07)"
        axis_color = "rgba(255,255,255,0.18)" if dark else "rgba(0,0,0,0.18)"
        title_color = "rgba(255,255,255,0.88)" if dark else "rgba(0,0,0,0.78)"
        ref_color  = "rgba(255,255,255,0.55)" if dark else "rgba(0,0,0,0.55)"

        custom_bg_css = self._theme.get("background_css", "")
        bg_css = custom_bg_css if custom_bg_css else f"background:{bg};"
        bg_svg = self._theme.get("background_svg", "")
        font = self._font_family()
        logo_invert = "filter:invert(1);" if not dark else ""

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
                f'width="24" height="24" alt="Signum" '
                f'style="position:fixed;left:10px;bottom:6px;'
                f'opacity:0.5;pointer-events:none;{logo_invert}">'
            )

        # Height accounting: title ≈ 40px
        grid_h = f"calc(100vh - {40 if self._title else 0}px)"

        return f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{{bg_css}font-family:{font};font-style:{font_style};overflow:hidden;position:relative;border-radius:12px}}
#grid{{display:grid;grid-template-columns:repeat({cols},1fr);
  gap:10px;padding:4px 16px 10px 16px;width:100%;height:{grid_h}}}
.cell{{position:relative;width:100%;height:100%;min-height:0}}
canvas{{display:block}}
</style></head><body>
{bg_svg}{title_html}
<div id="grid"></div>
{logo_html}
<script>(function(){{
"use strict";
const P={panels_json};
const grid=document.getElementById("grid");
const TC="{text_color}",SC="{sub_color}",GC="{grid_color}",
      AC="{axis_color}",RC="{ref_color}";
const FONT="{font}";
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

/* ── tooltip style ──────────────────────────────────────── */
const TT_BG="{("rgba(30,30,30,0.92)" if dark else "rgba(255,255,255,0.94)")}";
const TT_FG="{("rgba(255,255,255,0.9)" if dark else "rgba(0,0,0,0.8)")}";
const TT_BD="{("rgba(255,255,255,0.15)" if dark else "rgba(0,0,0,0.12)")}";
const CH_C="{("rgba(255,255,255,0.30)" if dark else "rgba(0,0,0,0.25)")}";

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
    +"padding:6px 10px;border-radius:4px;font:11px "+FONT+";"
    +"color:"+TT_FG+";background:"+TT_BG+";border:1px solid "+TT_BD+";"
    +"box-shadow:0 2px 8px rgba(0,0,0,0.3);z-index:10;white-space:nowrap;"
    +"line-height:1.5";
  cell.appendChild(tip);

  /* ── shared layout values (set during draw, used by hover) ── */
  let _pad,_pW,_pH,_xMin,_xMax,_yMin,_yMax,_maxC,_W,_H;
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
      const xP=(_xMax-_xMin)*0.05||1,yP=(_yMax-_yMin)*0.05||1;
      _xMin-=xP;_xMax+=xP;_yMin-=yP;_yMax+=yP;

      _pad={{top:30,right:14,bottom:34,left:52}};
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

      /* dots */
      ctx.fillStyle=p.color;
      for(let i=0;i<X.length;i++){{
        ctx.beginPath();ctx.arc(sx(X[i]),sy(Y[i]),p.size,0,Math.PI*2);ctx.fill();
      }}

      /* title */
      ctx.fillStyle=TC;ctx.font="bold 12px "+FONT;
      ctx.textAlign="center";ctx.textBaseline="top";
      ctx.fillText(p.name,_W/2,6);
    }}
  }}

  draw();
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
      if(best<0||Math.sqrt(bd)>30){{tip.style.display="none";return;}}
      const px=sx(X[best]),py=sy(Y[best]);
      /* ring around nearest dot */
      oc.save();oc.strokeStyle=p.color;oc.lineWidth=2;
      oc.beginPath();oc.arc(px,py,p.size+4,0,Math.PI*2);oc.stroke();
      oc.restore();
      /* snap crosshair to point */
      oc.save();oc.strokeStyle=CH_C;oc.lineWidth=1;oc.setLineDash([4,3]);
      oc.clearRect(0,0,_W,_H);
      oc.beginPath();oc.moveTo(px,_pad.top);oc.lineTo(px,_pad.top+_pH);oc.stroke();
      oc.beginPath();oc.moveTo(_pad.left,py);oc.lineTo(_W-_pad.right,py);oc.stroke();
      oc.restore();
      oc.save();oc.strokeStyle=p.color;oc.lineWidth=2;
      oc.beginPath();oc.arc(px,py,p.size+4,0,Math.PI*2);oc.stroke();
      oc.restore();
      tip.innerHTML="<b>x:</b> "+fmt(X[best])+"<br><b>y:</b> "+fmt(Y[best]);
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
}});
}})();
</script></body></html>"""

    # ── Display Methods (same pipeline as Chart) ──────────────────────────

    def _repr_html_(self) -> str:
        import base64
        html = self._build_html()
        b64 = base64.b64encode(html.encode("utf-8")).decode("ascii")
        h = self._height + (40 if self._title else 10)
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
            "height": f"{self._height + (40 if self._title else 10)}px",
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
            height=height or self._height + (40 if self._title else 10),
        )
