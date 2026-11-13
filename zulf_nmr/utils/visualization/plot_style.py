"""Colours of the program in two palettes (dark and light) and the matching plot style.

The colours are shared with the window theme (zulf_nmr.gui.theme), so a plot looks like
a part of the panel it is drawn on. use() switches the palette; apply() sets the plot
style for every plot made afterwards. The plotters read the colours from this module
when they draw, so a plot drawn after use() has the new colours.
"""

from __future__ import annotations

from functools import lru_cache

import matplotlib as mpl
from cycler import cycler
from matplotlib import font_manager

PALETTES: dict[str, dict] = {
    "dark": {
        "BACKGROUND": "#0b1220",
        "SURFACE": "#0e1729",
        "PANEL": "#111c30",
        "RAISED": "#1a2740",
        "RAISED_HOVER": "#22324f",
        "FIELD": "#0a1222",
        "BORDER": "#22314b",
        "TEXT": "#e2e8f0",
        "NAV_TEXT": "#c3cede",
        "MUTED": "#8a9bb5",
        "DIM": "#5b6b84",
        "ACCENT": "#22c3e6",
        "ACCENT_HOVER": "#4fd5f0",
        "ACCENT_TEXT": "#03161d",
        "ACCENT_SOFT": "#123b4a",
        "DANGER": "#e5484d",
        "DANGER_HOVER": "#f06a6e",
        "DANGER_SOFT": "#3a1820",
        "OK": "#2fbf71",
        "WARNING": "#f5a524",
        "INFO": "#60a5fa",
        "SELECTION": "#164e63",
        "ERROR_FIELD": "#3a1620",
        "CURSOR": "#f87171",
        "SPAN": "#94a3b8",
        "PLOT_BACKGROUND": "#0a111e",
        "AVERAGE_BACKGROUND": "#15151f",
        "AXIS_EDGE": "#2c3d5a",
        "GRID": "#3a4d6b",
        "GUIDE": "#334360",
        "SERIES": ("#22c3e6", "#f5b400", "#34d399", "#f472b6", "#fb923c", "#a78bfa", "#e2e8f0", "#94a3b8"),
        "CATEGORY_COLOURS": {"Shim": "#3b82f6", "Motor": "#eab308", "Digital": "#ec4899", "AWG": "#22c3e6",
                             "Acquisition": "#f97316", "Sequence": "#22c55e", "Other": "#94a3b8"},
    },
    "light": {
        "BACKGROUND": "#f2f5f9",
        "SURFACE": "#fbfcfe",
        "PANEL": "#ffffff",
        "RAISED": "#eef2f7",
        "RAISED_HOVER": "#e1e7ef",
        "FIELD": "#f8fafc",
        "BORDER": "#dbe2ea",
        "TEXT": "#0f172a",
        "NAV_TEXT": "#334155",
        "MUTED": "#64748b",
        "DIM": "#94a3b8",
        "ACCENT": "#0b8fb8",
        "ACCENT_HOVER": "#0aa3d1",
        "ACCENT_TEXT": "#ffffff",
        "ACCENT_SOFT": "#d6eef6",
        "DANGER": "#dc2626",
        "DANGER_HOVER": "#ef4444",
        "DANGER_SOFT": "#fbe3e3",
        "OK": "#16a34a",
        "WARNING": "#c2610c",
        "INFO": "#2563eb",
        "SELECTION": "#d3edf6",
        "ERROR_FIELD": "#fdeceb",
        "CURSOR": "#dc2626",
        "SPAN": "#64748b",
        "PLOT_BACKGROUND": "#ffffff",
        "AVERAGE_BACKGROUND": "#fff8f0",
        "AXIS_EDGE": "#c5ced9",
        "GRID": "#c9d2dd",
        "GUIDE": "#d5dce5",
        "SERIES": ("#0e7490", "#d97706", "#059669", "#db2777", "#ea580c", "#7c3aed", "#334155", "#64748b"),
        "CATEGORY_COLOURS": {"Shim": "#1d70b8", "Motor": "#d97706", "Digital": "#db2777", "AWG": "#0891b2",
                             "Acquisition": "#ea580c", "Sequence": "#16a34a", "Other": "#64748b"},
    },
}

MONO_FAMILIES = ("JetBrains Mono", "Cascadia Mono", "Consolas", "DejaVu Sans Mono")

current = "dark"
BACKGROUND = SURFACE = PANEL = RAISED = RAISED_HOVER = FIELD = BORDER = TEXT = NAV_TEXT = ""
MUTED = DIM = ACCENT = ACCENT_HOVER = ACCENT_TEXT = ACCENT_SOFT = DANGER = DANGER_HOVER = ""
DANGER_SOFT = OK = WARNING = INFO = SELECTION = ERROR_FIELD = CURSOR = SPAN = ""
PLOT_BACKGROUND = AVERAGE_BACKGROUND = AXIS_EDGE = GRID = GUIDE = ""
SERIES: tuple[str, ...] = ()
CATEGORY_COLOURS: dict[str, str] = {}


def use(name: str) -> None:
    """Switch to palette name ("dark" or "light"): the module's colours and the plot style."""
    global current
    if name not in PALETTES:
        name = "dark"
    current = name
    globals().update(PALETTES[name])
    apply()


@lru_cache(maxsize=None)
def mono_family() -> str:
    """The first monospace font of MONO_FAMILIES that matplotlib has in bold too."""
    available = {f.name for f in font_manager.fontManager.ttflist
                 if f.weight in ("bold", 700) or (isinstance(f.weight, int) and f.weight >= 600)}
    return next((name for name in MONO_FAMILIES if name in available), "monospace")


def rc() -> dict:
    """The rcParams of the current palette."""
    return {
        "figure.facecolor": PANEL,
        "figure.edgecolor": PANEL,
        "savefig.facecolor": PANEL,
        "savefig.edgecolor": PANEL,
        "axes.facecolor": PLOT_BACKGROUND,
        "axes.edgecolor": AXIS_EDGE,
        "axes.labelcolor": TEXT,
        "axes.titlecolor": TEXT,
        "axes.prop_cycle": cycler(color=SERIES),
        "text.color": TEXT,
        "xtick.color": AXIS_EDGE,
        "ytick.color": AXIS_EDGE,
        "xtick.labelcolor": MUTED,
        "ytick.labelcolor": MUTED,
        "grid.color": GRID,
        "legend.facecolor": PANEL,
        "legend.edgecolor": BORDER,
        "legend.labelcolor": TEXT,
        "patch.edgecolor": AXIS_EDGE,
        "font.family": mono_family(),
        "font.size": 9,
    }


def apply() -> None:
    """Use the current palette for every plot made from now on."""
    mpl.rcParams.update(rc())


def series(index: int) -> str:
    """Colour of data series index (0 = first channel)."""
    return SERIES[index % len(SERIES)]


globals().update(PALETTES[current])
