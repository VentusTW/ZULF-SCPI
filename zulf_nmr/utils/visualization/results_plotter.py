"""Plots the results of a queue run: for every configuration its transients, then its average."""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure

from zulf_nmr.experiment.schedule import QueueResult

ROW_HEIGHT_IN = 1.9
TRANSIENT_COLOUR = "#0072B2"
AVERAGE_COLOUR = "#D55E00"
_MAX_POINTS = 20000  # longer traces are decimated for drawing only


def result_rows(result: QueueResult) -> int:
    return sum(len(g.transients) + 1 for g in result.groups if g.transients)


def plot_results(result: QueueResult, fig: Figure | None = None) -> Figure:
    """One row per trace, top to bottom: group 1 transients, group 1 average, group 2..."""
    rows = result_rows(result)
    if fig is None:
        fig = Figure(figsize=(11, ROW_HEIGHT_IN * max(rows, 1) + 0.6), layout="constrained")
    if rows == 0:
        fig.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    axes = np.atleast_1d(fig.subplots(rows, 1))
    row = 0
    for group in result.groups:
        if not group.transients:
            continue
        traces = [(f"Transient {t.number:03d} (queue set {t.set_index}) — {t.file.name}", t.data, TRANSIENT_COLOUR)
                  for t in group.transients]
        numbers = ", ".join(f"{t.number:03d}" for t in group.transients)
        traces.append((f"Average of transient(s) {numbers} — {group.average_file.name}", group.average,
                       AVERAGE_COLOUR))
        for title, data, colour in traces:
            ax = axes[row]
            step = max(1, len(data) // _MAX_POINTS)
            t = np.arange(0, len(data), step) / group.sample_rate_hz
            ax.plot(t, data[::step], color=colour, linewidth=0.6)
            ax.set_title(title, loc="left", fontsize=9, fontweight="bold" if colour == AVERAGE_COLOUR else None)
            ax.set_ylabel("Signal", fontsize=8)
            ax.tick_params(labelsize=8)
            ax.grid(linestyle="--", alpha=0.4)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            row += 1
        axes[row - 1].set_xlabel("Time (s)", fontsize=8)
    fig.suptitle(f"Results — {result.folder.name}", fontweight="bold")
    return fig
