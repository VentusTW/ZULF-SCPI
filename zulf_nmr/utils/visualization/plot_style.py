"""Dark style of the plots, matching the program's window theme.

The colours are shared with the window theme (zulf_nmr.gui.theme), so a plot looks like
a part of the panel it is drawn on. apply() sets them for every plot made afterwards.
"""

from __future__ import annotations

from functools import lru_cache

import matplotlib as mpl
from cycler import cycler
from matplotlib import font_manager

BACKGROUND = "#0b1220"
SURFACE = "#0e1729"
PANEL = "#111c30"
RAISED = "#1a2740"
FIELD = "#0a1222"
BORDER = "#22314b"
TEXT = "#e2e8f0"
MUTED = "#8a9bb5"
DIM = "#5b6b84"
ACCENT = "#22c3e6"
DANGER = "#e5484d"
OK = "#2fbf71"
WARNING = "#f5a524"

PLOT_BACKGROUND = "#0a111e"
AXIS_EDGE = "#2c3d5a"
GRID = "#3a4d6b"
GUIDE = "#334360"
SERIES = ("#22c3e6", "#f5b400", "#34d399", "#f472b6", "#fb923c", "#a78bfa", "#e2e8f0", "#94a3b8")

MONO_FAMILIES = ("JetBrains Mono", "Cascadia Mono", "Consolas", "DejaVu Sans Mono")


@lru_cache(maxsize=None)
def mono_family() -> str:
    """The first monospace font of MONO_FAMILIES that matplotlib has in bold too."""
    available = {f.name for f in font_manager.fontManager.ttflist
                 if f.weight in ("bold", 700) or (isinstance(f.weight, int) and f.weight >= 600)}
    return next((name for name in MONO_FAMILIES if name in available), "monospace")


def rc() -> dict:
    """The rcParams of the dark style."""
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
    """Use the dark style for every plot made from now on."""
    mpl.rcParams.update(rc())
