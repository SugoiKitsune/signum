"""Theme gallery — renders one sample chart per theme into theme_gallery.html,
a visual reference for reviewing/comparing signum's themes. Re-run after editing
themes to refresh the HTML. Not shipped in the package (lives under support/).

    python support/theme_gallery.py     # -> support/theme_gallery.html
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
from signum.engine.themes import THEME_NAMES  # noqa: E402
from signum import Chart  # noqa: E402

rng = np.random.default_rng(7)
n = 120
close = 100 + np.cumsum(rng.normal(0.15, 1.6, n))
openp = close - rng.normal(0, 0.8, n)
high = np.maximum(openp, close) + rng.uniform(0.1, 1.2, n)
low = np.minimum(openp, close) - rng.uniform(0.1, 1.2, n)
vol = rng.uniform(5e5, 2e6, n)
dates = pd.date_range("2025-01-01", periods=n, freq="B")
df = pd.DataFrame({"time": dates, "Open": openp, "High": high, "Low": low,
                   "Close": close, "Volume": vol})
sma = pd.DataFrame({"time": dates, "value": pd.Series(close).rolling(20).mean()}).dropna()

NOTE = {
    "dark": "signum original (teal/red + blue)",
    "light": "ForgeFolio (green/crimson + gray)",
    "ft": "matches ForgeFolio",
    "midnight": "signum only",
    "rome": "matches ForgeFolio",
    "glass": "academic navy frost (formerly distfit; shared with ForgeFolio glass)",
}

cards = []
for name in THEME_NAMES:
    chart = Chart(theme=name, height=300, logo=False)
    chart.candlestick(df).volume(df).line(sma, name="SMA 20", width=2)
    cards.append(
        f'<div class="card"><div class="cap">theme="{name}"'
        f'<span class="nm">{NOTE.get(name, "")}</span></div>{chart._repr_html_()}</div>'
    )

page = f"""<!DOCTYPE html><html><head><meta charset="UTF-8"><title>signum themes</title>
<style>
body{{margin:0;padding:24px;background:#666;font-family:'Segoe UI',sans-serif}}
h1{{color:#fff;font-size:18px;margin:0 0 16px}}
.grid{{display:grid;grid-template-columns:repeat(2,1fr);gap:20px}}
.card{{background:#222;border-radius:12px;padding:8px;box-shadow:0 4px 16px rgba(0,0,0,.4)}}
.cap{{color:#eee;font:600 13px monospace;padding:6px 8px;display:flex;justify-content:space-between;align-items:center}}
.nm{{color:#9aa;font:11px sans-serif;font-weight:400}}
</style></head><body>
<h1>signum themes — open in a browser ({len(THEME_NAMES)} themes)</h1>
<div class="grid">{''.join(cards)}</div>
</body></html>"""

out = Path(__file__).resolve().parent / "theme_gallery.html"
out.write_text(page, encoding="utf-8")
print(f"wrote {out}  ({len(page)//1024} KB, {len(THEME_NAMES)} themes)")
