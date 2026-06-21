# Surface3D — 3-D surface PoC (echarts-gl)

Experimental, isolated on branch `feat/3d-surface`. **Not wired into main / `StatChart`** yet.
A rotatable WebGL surface for vol surfaces, term-structure / continuous-time quant
models, etc., themed from signum's `THEMES` so it matches Chart / StatChart.

## Check it

**Inline in Jupyter (primary)** — open `surface_demo.ipynb` (run from this folder)
and run all cells: the surfaces render **inline** via `_repr_html_` / `.show()`,
same display API as Chart / StatChart. (Each inline chart embeds the vendored
echarts ~2 MB, so don't commit executed outputs — run to view.)

**Standalone HTML (offline)** —

```
python surface_demo.py
```

writes two **self-contained** HTML files (echarts + echarts-gl inlined from
`src/signum/vendor/`, work offline) — open either in a browser:

- `surface_iv.html` — synthetic equity implied-vol surface (smile + skew + term structure), `midnight` theme
- `surface_bs.html` — Black-Scholes call-price surface vs spot × time, `glass` theme

Drag to rotate · scroll to zoom · hover for labelled x / y / z values.

## API

```python
from signum.engine.surface3d import Surface3D

Surface3D(theme="midnight", height=560, title="IV surface",
          colorscale="viridis", auto_rotate=False).surface(
    ttm, moneyness, iv,                       # x (nx), y (ny), z (ny×nx) or DataFrame
    x_label="TTM (yrs)", y_label="Moneyness", z_label="Implied vol",
    wireframe=True, shading="color",
).show()        # Jupyter — also .render() / .save(path)
```

- `z` accepts a 2-D array `(ny, nx)` or a DataFrame (index→y, cols→x); NaN punches holes.
- `colorscale`: `viridis` / `magma` / `plasma` / `turbo` / `rdylbu`, or an explicit list.
- Display API mirrors Chart / StatChart: `show()` / `render()` / `save()`.

## Status / next steps

- [x] Vendored echarts + echarts-gl (offline, self-contained HTML)
- [x] Theme-driven colours (bg / text / grid / font from `THEMES`)
- [x] Labelled hover tooltip, wireframe, colorscales, auto-rotate
- [ ] Fold into `StatChart` as `StatChart.surface(x, y, z)` (reuse grid/iframe pipeline)
- [ ] Mixed 2-D + 3-D panels in one StatChart grid
- [ ] Dated frames + slider (morph a surface through time), like `curve()`
