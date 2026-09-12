"""Surface3D — EXPERIMENTAL 3-D surface panel (echarts-gl).

Proof-of-concept for signum's planned 3-D rendering (implied-vol surfaces,
continuous-time / quant-model meshes). Renders an interactive, rotatable WebGL
surface from a grid of (x, y, z) values — e.g. an IV surface (ttm × strike × IV)
— using Apache ECharts + echarts-gl, themed from signum's ``THEMES`` so it
matches the 2-D Chart / StatChart look. Same display pipeline (Jupyter / Dash /
Streamlit / standalone HTML).

    from signum.engine.surface3d import Surface3D
    Surface3D(theme="midnight", title="IV surface").surface(
        ttm, moneyness, iv,
        x_label="TTM (yrs)", y_label="Moneyness", z_label="Implied vol",
    ).show()

PoC NOTE: echarts + echarts-gl are vendored under ``vendor/`` and inlined, so the
output HTML is fully self-contained and works offline (no CDN) — just open it.
If this graduates, fold it into ``StatChart`` as ``StatChart.surface(x, y, z)``
reusing its grid/iframe pipeline.
"""

import base64
import html as _html
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from .themes import resolve_theme, DARK_THEMES
from .chart import _LOGO_B64

_VENDOR = Path(__file__).resolve().parent.parent / "vendor"  # src/signum/vendor
_ECHARTS_JS: Optional[str] = None


def _echarts_js() -> str:
    """Inlined echarts + echarts-gl (vendored — offline, self-contained).

    echarts core must load before echarts-gl (the GL series register onto it).
    Cached after first read.
    """
    global _ECHARTS_JS
    if _ECHARTS_JS is None:
        core = (_VENDOR / "echarts.min.js").read_text(encoding="utf-8")
        gl = (_VENDOR / "echarts-gl.min.js").read_text(encoding="utf-8")
        _ECHARTS_JS = f"<script>{core}</script>\n<script>{gl}</script>"
    return _ECHARTS_JS

# Perceptually-ordered colour ramps for the z (height) gradient. Diverging
# 'rdylbu' suits a centred quantity (e.g. spread vs fair); the rest are
# sequential and read well on both dark and light backgrounds.
_COLORSCALES = {
    "viridis": ["#440154", "#414487", "#2a788e", "#22a884", "#7ad151", "#fde725"],
    "magma":   ["#000004", "#3b0f70", "#8c2981", "#de4968", "#fe9f6d", "#fcfdbf"],
    "plasma":  ["#0d0887", "#6a00a8", "#b12a90", "#e16462", "#fca636", "#f0f921"],
    "turbo":   ["#30123b", "#4146f7", "#1ddfa3", "#a4fc3b", "#fb8022", "#7a0403"],
    "rdylbu":  ["#a50026", "#f46d43", "#fee090", "#e0f3f8", "#74add1", "#313695"],
}


class Surface3D:
    """Experimental rotatable 3-D surface (echarts-gl), themed via signum THEMES."""

    def __init__(
        self,
        theme: str = "dark",
        width: Optional[int] = None,
        height: int = 520,
        title: Optional[str] = None,
        colorscale: str = "viridis",
        auto_rotate: bool = False,
        logo: bool = True,
    ):
        self._theme_name = theme.lower()
        self._theme = resolve_theme(theme)
        self._width = width
        self._height = height
        self._title = title
        self._colorscale = colorscale
        self._auto_rotate = bool(auto_rotate)
        self._logo = logo
        self._panels: List[dict] = []

    # ── Panel ──────────────────────────────────────────────────────────────

    def surface(
        self,
        x=None,
        y=None,
        z=None,
        name: Optional[str] = None,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        z_label: Optional[str] = None,
        colorscale=None,
        color: Optional[str] = None,
        opacity: Optional[float] = None,
        wireframe: bool = True,
        shading: str = "color",
    ) -> "Surface3D":
        """Define the surface.

        Parameters
        ----------
        x : 1-D array, length ``nx``   (e.g. time-to-maturity grid)
        y : 1-D array, length ``ny``   (e.g. strike / moneyness grid)
        z : 2-D array ``(ny, nx)`` OR a ``DataFrame`` (index → y, columns → x)
            The surface heights (e.g. implied vol). ``NaN`` cells punch holes.
        colorscale : str or list, optional
            One of viridis / magma / plasma / turbo / rdylbu, or an explicit
            list of CSS colours. Defaults to the instance colorscale.
        color : str, optional
            Flat CSS colour for this surface. Call ``.surface(...)`` more than
            once to **overlay** multiple surfaces on one chart; give each a
            distinct ``color`` and an ``opacity`` so they read apart.
        opacity : float, optional
            Surface opacity in [0, 1] (defaults to 0.7 in overlay mode).
        wireframe : bool
            Overlay a faint mesh grid on the surface.
        shading : {"color", "lambert", "realistic"}
            echarts-gl shading model.
        """
        if isinstance(z, pd.DataFrame):
            if x is None:
                x = list(z.columns)
            if y is None:
                y = list(z.index)
            zmat = np.asarray(z.values, dtype=float)
        else:
            zmat = np.asarray(z, dtype=float)

        xa = np.asarray(x, dtype=float).ravel()
        ya = np.asarray(y, dtype=float).ravel()
        if zmat.ndim != 2:
            raise ValueError("z must be 2-D (ny, nx) or a pandas DataFrame")
        # Normalise to (ny, nx): if z was given transposed as (nx, ny), flip it.
        if zmat.shape == (len(xa), len(ya)) and len(xa) != len(ya):
            zmat = zmat.T
        ny, nx = zmat.shape
        if (ny, nx) != (len(ya), len(xa)):
            raise ValueError(
                f"z shape {zmat.shape} doesn't match (len(y), len(x)) = "
                f"({len(ya)}, {len(xa)})"
            )

        # echarts-gl non-parametric surface wants a flat [x, y, z] grid, emitted
        # x-outer / y-inner to match echarts-gl's own surface-data example.
        data: List[list] = []
        for i in range(nx):
            for j in range(ny):
                v = zmat[j, i]
                data.append([
                    round(float(xa[i]), 6),
                    round(float(ya[j]), 6),
                    None if v != v else round(float(v), 6),
                ])

        finite = zmat[np.isfinite(zmat)]
        zmin = float(finite.min()) if finite.size else 0.0
        zmax = float(finite.max()) if finite.size else 1.0

        cs = colorscale if colorscale is not None else self._colorscale
        colors = cs if isinstance(cs, (list, tuple)) else _COLORSCALES.get(cs, _COLORSCALES["viridis"])

        self._panels.append({
            "name": name or f"Surface {len(self._panels) + 1}",
            "data": data,
            "data_shape": [nx, ny],           # explicit grid dims -> no auto-detect striping
            "zmin": round(zmin, 6),
            "zmax": round(zmax, 6),
            "x_label": x_label,
            "y_label": y_label,
            "z_label": z_label,
            "colors": list(colors),
            "color": color,                    # flat colour (overlay mode)
            "opacity": opacity,
            "wireframe": bool(wireframe),
            "shading": shading,
        })
        return self

    # ── Theme helpers (subset of StatChart's) ──────────────────────────────

    def _is_dark(self) -> bool:
        bg = (self._theme.get("chart", {}).get("layout", {})
              .get("background", {}).get("color", "#1e1e1e"))
        if bg.startswith("#") and len(bg) >= 7:
            r, g, b = (int(bg[i:i + 2], 16) for i in (1, 3, 5))
            return (r * 0.299 + g * 0.587 + b * 0.114) < 140
        return self._theme_name in DARK_THEMES

    @staticmethod
    def _rgba(color: Optional[str], alpha: float) -> Optional[str]:
        if not color:
            return None
        if color.startswith("#"):
            h = color.lstrip("#")
            if len(h) == 3:
                h = "".join(c * 2 for c in h)
            r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
            return f"rgba({r},{g},{b},{alpha})"
        return color

    # ── Build ──────────────────────────────────────────────────────────────

    def _build_html(self) -> str:
        if not self._panels:
            return "<html><body>No surface defined — call .surface(x, y, z)</body></html>"
        panels = self._panels
        p0 = panels[0]
        _layout = self._theme.get("chart", {}).get("layout", {})
        bg = _layout.get("background", {}).get("color", "#1e1e1e")
        text = _layout.get("textColor", "#d1d4dc")
        font = _layout.get("fontFamily", "sans-serif")
        grid_c = (self._theme.get("chart", {}).get("grid", {})
                  .get("horzLines", {}).get("color") or self._rgba(text, 0.12))
        axis_c = self._rgba(text, 0.55) or text

        def axis3d(default_name, label):
            return {
                "type": "value",
                "name": label or default_name,
                "nameTextStyle": {"color": text, "fontFamily": font},
                "axisLine": {"lineStyle": {"color": axis_c}},
                "axisLabel": {"color": axis_c, "fontFamily": font},
                "splitLine": {"lineStyle": {"color": grid_c}},
            }

        # Overlay mode when >1 surface, or any surface asked for a flat colour:
        # colour each surface by a flat itemStyle (so they read apart) and drop the
        # z-gradient visualMap. A single gradient surface keeps the visualMap.
        overlay = len(panels) > 1 or any(pp.get("color") for pp in panels)
        series = []
        for pp in panels:
            s = {
                "type": "surface",
                "name": pp["name"],
                "dataShape": pp["data_shape"],       # explicit grid -> no striping
                "wireframe": {"show": pp["wireframe"]},
                "shading": pp["shading"],
                "data": pp["data"],
            }
            if overlay:
                col = pp.get("color") or pp["colors"][len(pp["colors"]) // 2]
                op = pp.get("opacity")
                s["itemStyle"] = {"color": col, "opacity": op if op is not None else 0.7}
            elif pp.get("opacity") is not None:
                s["itemStyle"] = {"opacity": pp["opacity"]}
            series.append(s)

        opt = {
            "backgroundColor": bg,
            "tooltip": {"formatter": "__TTFMT__"},
            "xAxis3D": axis3d("x", p0["x_label"]),
            "yAxis3D": axis3d("y", p0["y_label"]),
            "zAxis3D": axis3d("z", p0["z_label"]),
            "grid3D": {
                "axisLine": {"lineStyle": {"color": axis_c}},
                "splitLine": {"lineStyle": {"color": grid_c}},
                "viewControl": {"autoRotate": self._auto_rotate},
            },
            "series": series,
        }
        if overlay:
            opt["legend"] = {
                "data": [pp["name"] for pp in panels],
                "textStyle": {"color": text, "fontFamily": font}, "top": 8, "left": 8,
            }
        else:
            opt["visualMap"] = {
                "show": True, "dimension": 2,
                "min": p0["zmin"], "max": p0["zmax"],
                "inRange": {"color": p0["colors"]},
                "textStyle": {"color": text, "fontFamily": font},
                "right": 12, "top": "center", "calculable": True,
            }
        opt_json = json.dumps(opt, separators=(",", ":"))
        # Labelled hover formatter — echarts formatter is a JS function, so it
        # can't live in the JSON; splice it in by token.
        xl, yl, zl = (p0["x_label"] or "x", p0["y_label"] or "y", p0["z_label"] or "z")
        fmt_js = ("function(pp){var v=pp.value;if(!v)return '';return "
                  + json.dumps(xl) + "+': '+(+v[0]).toFixed(3)+'<br>'+"
                  + json.dumps(yl) + "+': '+(+v[1]).toFixed(3)+'<br>'+"
                  + json.dumps(zl) + "+': '+(+v[2]).toFixed(4);}")
        opt_json = opt_json.replace('"__TTFMT__"', fmt_js)

        title = self._title or p0["name"]
        title_html = ""
        if title:
            tc = self._rgba(text, 0.95) or text
            title_html = (
                f'<div style="position:absolute;top:10px;left:16px;z-index:5;'
                f'color:{tc};font:600 15px {font};letter-spacing:.2px">'
                f'{_html.escape(title)}</div>'
            )

        logo_html = ""
        if self._logo:
            _inv = "" if self._is_dark() else "filter:invert(1);"
            logo_html = (
                f'<img src="data:image/svg+xml;base64,{_LOGO_B64}" width="30" height="30" '
                f'alt="Signum" style="position:fixed;right:12px;bottom:8px;opacity:0.75;'
                f'pointer-events:none;z-index:6;{_inv}">'
            )

        return f"""<!DOCTYPE html><html><head><meta charset="UTF-8">
{_echarts_js()}
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
html,body{{width:100%;height:100%;overflow:hidden;background:{bg};font-family:{font};border-radius:12px}}
#c{{width:100%;height:100%}}
</style></head><body>{title_html}{logo_html}<div id="c"></div>
<script>
(function(){{
  var el=document.getElementById('c');
  if(!window.echarts||!echarts.init){{
    el.innerHTML='<p style="color:{text};padding:24px;font:13px sans-serif">echarts failed to initialise.</p>';
    return;
  }}
  var chart=echarts.init(el,null,{{renderer:'canvas'}});
  chart.setOption({opt_json});
  window.addEventListener('resize',function(){{chart.resize();}});
}})();
</script></body></html>"""

    # ── Display (mirrors Chart / StatChart) ────────────────────────────────

    def _repr_html_(self) -> str:
        # Inline `srcdoc` iframe. Renders in VSCode, JupyterLab and classic notebook.
        # The previous `src=blob:` approach is blocked by VSCode's notebook-output CSP
        # (frame-src forbids blob:), so the chart showed up blank there. echarts-gl is
        # vendored/inlined in the doc, so srcdoc is fully self-contained (no CDN).
        h = self._height + (40 if self._title else 10)
        srcdoc = _html.escape(self._build_html(), quote=True)
        return (
            f'<iframe srcdoc="{srcdoc}" '
            f'style="width:100%;height:{h}px;border:none;border-radius:12px;overflow:hidden;">'
            f'</iframe>'
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
