"""Signum logo variants (base64-encoded SVGs).

To switch the active logo, change the import in chart.py:
    from .logos import LOGO_APEX as _LOGO_B64     # current: minimal arc + ECG
    from .logos import LOGO_DIAMOND as _LOGO_B64  # classic: diamond frame + pulse

Both are white-on-transparent SVGs (64×64).  The logo is auto-inverted on
light backgrounds via the _logo_invert CSS filter in _build_html().
"""

# ── LOGO_APEX (current) ───────────────────────────────────────────────────────
# 25th-century Apple-style: thin incomplete progress ring (gap intentional —
# watch-complication aesthetic, like an Activity ring mid-fill) + ghost inner
# ring for depth + single clean minimal ECG waveform.
LOGO_APEX = (
    "PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA2NCA2NCIg"
    "d2lkdGg9IjY0IiBoZWlnaHQ9IjY0Ij48Y2lyY2xlIGN4PSIzMiIgY3k9IjMyIiByPSIyNyIgZmlsbD0i"
    "bm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIxIiBzdHJva2UtbGluZWNhcD0icm91bmQi"
    "IHN0cm9rZS1kYXNoYXJyYXk9IjE1MyAxNyIgc3Ryb2tlLWRhc2hvZmZzZXQ9Ii01MCIvPjxjaXJjbGUg"
    "Y3g9IjMyIiBjeT0iMzIiIHI9IjE5IiBmaWxsPSJub25lIiBzdHJva2U9IndoaXRlIiBzdHJva2Utd2lk"
    "dGg9IjAuNSIgb3BhY2l0eT0iMC4yOCIvPjxwb2x5bGluZSBwb2ludHM9IjEzLDMyIDIxLDMyIDI3LDE3"
    "IDMyLDQ3IDM3LDE3IDQzLDMyIDUxLDMyIiBmaWxsPSJub25lIiBzdHJva2U9IndoaXRlIiBzdHJva2Utd2lk"
    "dGg9IjIuMiIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PC9zdmc+"
)

# ── LOGO_DIAMOND (classic / v2) ───────────────────────────────────────────────
# Original Signum v2 logo: outer diamond frame with corner tick marks,
# complex multi-point signal waveform, flanking dots.
LOGO_DIAMOND = (
    "PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA2"
    "NCA2NCIgd2lkdGg9IjY0IiBoZWlnaHQ9IjY0Ij4NCiAgPCEtLSBTaWdudW0gdjI6IEVuaGFu"
    "Y2VkIHNpZ25hbCBwdWxzZSB0aHJvdWdoIGRpYW1vbmQgd2l0aCBpbm5lciBnZW9tZXRyeSAt"
    "LT4NCiAgPCEtLSBPdXRlciBkaWFtb25kIGZyYW1lIC0tPg0KICA8cG9seWdvbiBwb2ludHM9"
    "IjMyLDIgNjIsMzIgMzIsNjIgMiwzMiIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ry"
    "b2tlLXdpZHRoPSIyIi8+DQogIDwhLS0gQ29ybmVyIHRpY2sgbWFya3Mgb24gb3V0ZXIgZGlh"
    "bW9uZCAtLT4NCiAgPGxpbmUgeDE9IjMyIiB5MT0iMiIgeDI9IjMyIiB5Mj0iOCIgc3Ryb2tl"
    "PSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIxLjUiLz4NCiAgPGxpbmUgeDE9IjYyIiB5MT0iMzIi"
    "IHgyPSI1NiIgeTI9IjMyIiBzdHJva2U9IndoaXRlIiBzdHJva2Utd2lkdGg9IjEuNSIvPg0K"
    "ICA8bGluZSB4MT0iMzIiIHkxPSI2MiIgeDI9IjMyIiB5Mj0iNTYiIHN0cm9rZT0id2hpdGUi"
    "IHN0cm9rZS13aWR0aD0iMS41Ii8+DQogIDxsaW5lIHgxPSIyIiB5MT0iMzIiIHgyPSI4IiB5"
    "Mj0iMzIiIHN0cm9rZT0id2hpdGUiIHN0cm9rZS13aWR0aD0iMS41Ii8+DQogIDwhLS0gU2ln"
    "bmFsIHB1bHNlIHdhdmUgKG1vcmUgY29tcGxleCB3YXZlZm9ybSkgLS0+DQogIDxwb2x5bGlu"
    "ZSBwb2ludHM9IjYsMzIgMTYsMzIgMTksMzIgMjIsMjIgMjUsNDAgMjgsMTYgMzIsNDggMzUs"
    "MjAgMzgsMzggNDEsMjggNDQsMzIgNDgsMzIgNTgsMzIiDQogICAgICAgICAgICBmaWxsPSJu"
    "b25lIiBzdHJva2U9IndoaXRlIiBzdHJva2Utd2lkdGg9IjIuMiIgc3Ryb2tlLWxpbmVqb2lu"
    "PSJyb3VuZCIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+DQogIDwhLS0gU21hbGwgZmxhbmtp"
    "bmcgZG90cyAtLT4NCiAgPGNpcmNsZSBjeD0iMTAiIGN5PSIzMiIgcj0iMS4yIiBmaWxsPSJ3"
    "aGl0ZSIvPg0KICA8Y2lyY2xlIGN4PSI1NCIgY3k9IjMyIiByPSIxLjIiIGZpbGw9IndoaXRl"
    "Ii8+DQo8L3N2Zz4NCg=="
)
