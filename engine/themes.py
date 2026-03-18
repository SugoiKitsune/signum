"""Theme definitions matching Signum's design system.

Five themes: dark, light, ft (Financial Times), midnight, rome (Roman Empire).
"""

THEMES = {
    "dark": {
        "chart": {
            "layout": {
                "background": {"type": "solid", "color": "#1e1e1e"},
                "textColor": "#d1d4dc",
                "fontSize": 12,
                "fontFamily": "'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif",
            },
            "grid": {
                "vertLines": {"color": "rgba(255, 255, 255, 0.04)"},
                "horzLines": {"color": "rgba(255, 255, 255, 0.04)"},
            },
            "crosshair": {
                "mode": 0,
                "vertLine": {"width": 1, "color": "rgba(255, 255, 255, 0.3)", "style": 3},
                "horzLine": {"width": 1, "color": "rgba(255, 255, 255, 0.3)", "style": 3},
            },
            "rightPriceScale": {"borderColor": "rgba(255, 255, 255, 0.08)"},
            "timeScale": {"borderColor": "rgba(255, 255, 255, 0.08)"},
        },
        "candlestick": {
            "upColor": "#26a69a",
            "downColor": "#ef5350",
            "borderUpColor": "#26a69a",
            "borderDownColor": "#ef5350",
            "wickUpColor": "#26a69a",
            "wickDownColor": "#ef5350",
        },
        "line": {"color": "#2962FF", "lineWidth": 2},
        "line_colors": [
            "#2962FF", "#FF6D00", "#E91E63", "#00BCD4",
            "#4CAF50", "#9C27B0", "#FF9800", "#607D8B",
        ],
        "area": {
            "topColor": "rgba(41, 98, 255, 0.56)",
            "bottomColor": "rgba(41, 98, 255, 0.04)",
            "lineColor": "#2962FF",
            "lineWidth": 2,
        },
        "baseline": {
            "topLineColor": "#26a69a",
            "topFillColor1": "rgba(38, 166, 154, 0.28)",
            "topFillColor2": "rgba(38, 166, 154, 0.05)",
            "bottomLineColor": "#ef5350",
            "bottomFillColor1": "rgba(239, 83, 80, 0.05)",
            "bottomFillColor2": "rgba(239, 83, 80, 0.28)",
        },
        "histogram": {"color": "#26a69a"},
        "volume": {
            "upColor": "rgba(38, 166, 154, 0.5)",
            "downColor": "rgba(239, 83, 80, 0.5)",
        },
    },

    "light": {
        "chart": {
            "layout": {
                "background": {"type": "solid", "color": "#ffffff"},
                "textColor": "#191919",
                "fontSize": 12,
                "fontFamily": "'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif",
            },
            "grid": {
                "vertLines": {"color": "rgba(0, 0, 0, 0.06)"},
                "horzLines": {"color": "rgba(0, 0, 0, 0.06)"},
            },
            "crosshair": {
                "mode": 0,
                "vertLine": {"width": 1, "color": "rgba(0, 0, 0, 0.3)", "style": 3},
                "horzLine": {"width": 1, "color": "rgba(0, 0, 0, 0.3)", "style": 3},
            },
            "rightPriceScale": {"borderColor": "rgba(0, 0, 0, 0.1)"},
            "timeScale": {"borderColor": "rgba(0, 0, 0, 0.1)"},
        },
        "candlestick": {
            "upColor": "#26a69a",
            "downColor": "#ef5350",
            "borderUpColor": "#26a69a",
            "borderDownColor": "#ef5350",
            "wickUpColor": "#26a69a",
            "wickDownColor": "#ef5350",
        },
        "line": {"color": "#2962FF", "lineWidth": 2},
        "line_colors": [
            "#2962FF", "#FF6D00", "#E91E63", "#00897B",
            "#43A047", "#7B1FA2", "#EF6C00", "#546E7A",
        ],
        "area": {
            "topColor": "rgba(41, 98, 255, 0.56)",
            "bottomColor": "rgba(41, 98, 255, 0.04)",
            "lineColor": "#2962FF",
            "lineWidth": 2,
        },
        "baseline": {
            "topLineColor": "#26a69a",
            "topFillColor1": "rgba(38, 166, 154, 0.28)",
            "topFillColor2": "rgba(38, 166, 154, 0.05)",
            "bottomLineColor": "#ef5350",
            "bottomFillColor1": "rgba(239, 83, 80, 0.05)",
            "bottomFillColor2": "rgba(239, 83, 80, 0.28)",
        },
        "histogram": {"color": "#26a69a"},
        "volume": {
            "upColor": "rgba(38, 166, 154, 0.5)",
            "downColor": "rgba(239, 83, 80, 0.5)",
        },
    },

    "ft": {
        "chart": {
            "layout": {
                "background": {"type": "solid", "color": "#fff1e5"},
                "textColor": "#33302e",
                "fontSize": 12,
                "fontFamily": "Georgia, 'Times New Roman', serif",
            },
            "grid": {
                "vertLines": {"color": "rgba(51, 48, 46, 0.06)"},
                "horzLines": {"color": "rgba(51, 48, 46, 0.06)"},
            },
            "crosshair": {
                "mode": 0,
                "vertLine": {"width": 1, "color": "rgba(153, 15, 61, 0.4)", "style": 3},
                "horzLine": {"width": 1, "color": "rgba(153, 15, 61, 0.4)", "style": 3},
            },
            "rightPriceScale": {"borderColor": "rgba(51, 48, 46, 0.12)"},
            "timeScale": {"borderColor": "rgba(51, 48, 46, 0.12)"},
        },
        "candlestick": {
            "upColor": "#00847b",
            "downColor": "#cc0000",
            "borderUpColor": "#00847b",
            "borderDownColor": "#cc0000",
            "wickUpColor": "#00847b",
            "wickDownColor": "#cc0000",
        },
        "line": {"color": "#990f3d", "lineWidth": 2},
        "line_colors": [
            "#990f3d", "#0d7680", "#593380", "#ff7faa",
            "#00847b", "#96cc28", "#ff8833", "#0f5499",
        ],
        "area": {
            "topColor": "rgba(153, 15, 61, 0.4)",
            "bottomColor": "rgba(153, 15, 61, 0.04)",
            "lineColor": "#990f3d",
            "lineWidth": 2,
        },
        "baseline": {
            "topLineColor": "#00847b",
            "topFillColor1": "rgba(0, 132, 123, 0.28)",
            "topFillColor2": "rgba(0, 132, 123, 0.05)",
            "bottomLineColor": "#cc0000",
            "bottomFillColor1": "rgba(204, 0, 0, 0.05)",
            "bottomFillColor2": "rgba(204, 0, 0, 0.28)",
        },
        "histogram": {"color": "#990f3d"},
        "volume": {
            "upColor": "rgba(0, 132, 123, 0.4)",
            "downColor": "rgba(204, 0, 0, 0.4)",
        },
    },

    "midnight": {
        "chart": {
            "layout": {
                "background": {"type": "solid", "color": "#131722"},
                "textColor": "#d1d4dc",
                "fontSize": 12,
                "fontFamily": "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
            },
            "grid": {
                "vertLines": {"color": "#1f2937"},
                "horzLines": {"color": "#1f2937"},
            },
            "crosshair": {
                "mode": 0,
                "vertLine": {"width": 1, "color": "#758696", "style": 3},
                "horzLine": {"width": 1, "color": "#758696", "style": 3},
            },
            "rightPriceScale": {"borderColor": "#2a2e39"},
            "timeScale": {"borderColor": "#2a2e39"},
        },
        "candlestick": {
            "upColor": "#26a69a",
            "downColor": "#ef5350",
            "borderUpColor": "#26a69a",
            "borderDownColor": "#ef5350",
            "wickUpColor": "#26a69a",
            "wickDownColor": "#ef5350",
        },
        "line": {"color": "#2962FF", "lineWidth": 2},
        "line_colors": [
            "#2962FF", "#FF6D00", "#E91E63", "#00BCD4",
            "#4CAF50", "#9C27B0", "#FF9800", "#607D8B",
        ],
        "area": {
            "topColor": "rgba(41, 98, 255, 0.56)",
            "bottomColor": "rgba(41, 98, 255, 0.04)",
            "lineColor": "#2962FF",
            "lineWidth": 2,
        },
        "baseline": {
            "topLineColor": "#26a69a",
            "topFillColor1": "rgba(38, 166, 154, 0.28)",
            "topFillColor2": "rgba(38, 166, 154, 0.05)",
            "bottomLineColor": "#ef5350",
            "bottomFillColor1": "rgba(239, 83, 80, 0.05)",
            "bottomFillColor2": "rgba(239, 83, 80, 0.28)",
        },
        "histogram": {"color": "#26a69a"},
        "volume": {
            "upColor": "rgba(38, 166, 154, 0.5)",
            "downColor": "rgba(239, 83, 80, 0.5)",
        },
    },

    # ── Rome (Roman Empire) ───────────────────────────────────────────────
    # Cool white marble background, Mediterranean teal (up), shield crimson (down),
    # gold accents, Tyrian purple lines, Palatino serif font.
    "rome": {
        "chart": {
            "layout": {
                "background": {"type": "solid", "color": "#fafafa"},
                "textColor": "#3d2b1f",
                "fontSize": 12,
                "fontFamily": "'Palatino Linotype', Palatino, 'Book Antiqua', Georgia, serif",
            },
            "grid": {
                "vertLines": {"color": "rgba(0, 0, 0, 0.05)"},
                "horzLines": {"color": "rgba(0, 0, 0, 0.05)"},
            },
            "crosshair": {
                "mode": 0,
                "vertLine": {"width": 1, "color": "rgba(184, 134, 11, 0.5)", "style": 3},
                "horzLine": {"width": 1, "color": "rgba(184, 134, 11, 0.5)", "style": 3},
            },
            "rightPriceScale": {"borderColor": "rgba(61, 43, 31, 0.12)"},
            "timeScale": {"borderColor": "rgba(61, 43, 31, 0.12)"},
        },
        "candlestick": {
            "upColor": "#2a8c82",
            "downColor": "#8b1a2b",
            "borderUpColor": "#2a8c82",
            "borderDownColor": "#8b1a2b",
            "wickUpColor": "#2a8c82",
            "wickDownColor": "#8b1a2b",
        },
        "line": {"color": "#6b3fa0", "lineWidth": 2},
        "line_colors": [
            "#6b3fa0", "#b8860b", "#8b1a2b", "#2a8c82",
            "#cd853f", "#4a6741", "#c17817", "#5b7eaa",
        ],
        "area": {
            "topColor": "rgba(107, 63, 160, 0.35)",
            "bottomColor": "rgba(107, 63, 160, 0.04)",
            "lineColor": "#6b3fa0",
            "lineWidth": 2,
        },
        "baseline": {
            "topLineColor": "#2a8c82",
            "topFillColor1": "rgba(42, 140, 130, 0.25)",
            "topFillColor2": "rgba(42, 140, 130, 0.04)",
            "bottomLineColor": "#8b1a2b",
            "bottomFillColor1": "rgba(139, 26, 43, 0.04)",
            "bottomFillColor2": "rgba(139, 26, 43, 0.25)",
        },
        "histogram": {"color": "#b8860b"},
        "volume": {
            "upColor": "rgba(42, 140, 130, 0.40)",
            "downColor": "rgba(139, 26, 43, 0.40)",
        },
    },
}
