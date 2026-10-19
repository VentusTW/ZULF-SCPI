"""Plots the event timeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from zulf_nmr.core.scpi import mnemonic_matches
from zulf_nmr.experiment.parser import Event, ExperimentConfig

# Subsystem -> (root mnemonic, colour). Okabe-Ito palette, colour-blind safe.
CATEGORIES = {
    "Shim": ("SHIM", "#0072B2"),
    "Motor": ("MOTor", "#E69F00"),
    "Digital": ("DIGital", "#CC79A7"),
    "AWG": ("AWG", "#56B4E9"),
    "Acquisition": ("ACQuire", "#D55E00"),
    "Sequence": ("SEQuence", "#009E73"),
}
OTHER = ("Other", "#7F7F7F")


def event_category(action: str) -> tuple[str, str]:
    """Return (category name, colour) from the root of the command header."""
    root = action.strip().lstrip(":").split(":", 1)[0].split(" ", 1)[0]
    for name, (mnemonic, colour) in CATEGORIES.items():
        if mnemonic_matches(root, mnemonic):
            return name, colour
    return OTHER


def load_events(source: ExperimentConfig | str | Path) -> tuple[str, list[Event]]:
    """Return (experiment name, events) from a parsed config or a JSON file.

    A JSON file is read without validation, so drafts can be plotted too.
    """
    if isinstance(source, ExperimentConfig):
        return source.experiment_name, list(source.events)
    path = Path(source)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    events = [Event(int(e["time_ms"]), str(e["action"])) for e in data.get("events", [])]
    return data.get("experiment_name", path.stem), sorted(events, key=lambda e: e.time_ms)


ACQUISITION_COLOUR = "#D55E00"


@dataclass(frozen=True)
class AcquisitionWindow:
    """Acquisition shown on the timeline: start and end in ms of the sequence, and its channels."""

    start_ms: float
    end_ms: float
    channels: tuple[int, ...]


def plot_events(
    events: Iterable[Event],
    title: str | None = None,
    ax: Axes | None = None,
    durations_ms: Sequence[float | None] | None = None,
    acquisition: AcquisitionWindow | None = None,
) -> Figure | SubFigure:
    """Draw one row per event (first event on top): a dot at its time, a bar as
    long as its duration (durations_ms, one per event, None = unknown), the
    command on the right, coloured by subsystem. acquisition adds a bottom row
    with the recording window and its channels."""
    events = list(events)
    durations = list(durations_ms) if durations_ms is not None else [None] * len(events)
    n_rows = len(events) + (1 if acquisition else 0)
    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 0.45 * n_rows + 2))
    else:
        fig = ax.figure

    ends = [e.time_ms + (d or 0) for e, d in zip(events, durations)]
    if acquisition:
        ends.append(acquisition.end_ms)
    t_max = max(ends, default=0) or 1
    rows = range(n_rows, 0, -1)
    used: dict[str, str] = {}

    for y, event, duration in zip(rows, events, durations):
        name, colour = event_category(event.action)
        used.setdefault(name, colour)
        ax.hlines(y, 0, event.time_ms, color="#D0D0D0", linewidth=0.8, zorder=1)
        if duration:
            ax.add_patch(Rectangle((event.time_ms, y - 0.28), duration, 0.56, facecolor=colour,
                                   alpha=0.35, edgecolor=colour, linewidth=0.8, zorder=2))
        ax.plot(event.time_ms, y, "o", color=colour, markersize=7, zorder=3)
        label = f"{event.time_ms:g} ms" + (f"  (+{duration:g} ms)" if duration else "")
        ax.annotate(
            label, (event.time_ms + (duration or 0), y), xytext=(6, 0),
            textcoords="offset points", va="center", fontsize=8, color="#404040",
        )
        ax.annotate(
            event.action, (1.01, y), xycoords=("axes fraction", "data"),
            va="center", fontsize=8, family="monospace", color=colour,
        )

    if acquisition:
        y = 1
        width = acquisition.end_ms - acquisition.start_ms
        ax.add_patch(Rectangle((acquisition.start_ms, y - 0.28), width, 0.56, facecolor=ACQUISITION_COLOUR,
                               alpha=0.25, edgecolor=ACQUISITION_COLOUR, linewidth=1.0, zorder=2))
        channels = ", ".join(map(str, acquisition.channels))
        ax.annotate(
            f"{acquisition.start_ms:g} – {acquisition.end_ms:g} ms", (acquisition.start_ms, y),
            xytext=(4, 0), textcoords="offset points", va="center", fontsize=8, color="#404040",
        )
        ax.annotate(
            f"Acquisition, CH {channels}", (1.01, y), xycoords=("axes fraction", "data"),
            va="center", fontsize=8, family="monospace", color=ACQUISITION_COLOUR, fontweight="bold",
        )
        used.setdefault("Acquisition", ACQUISITION_COLOUR)

    ax.set_xlim(-0.03 * t_max, 1.12 * t_max)
    ax.set_ylim(0.3, n_rows + 0.7)
    ax.set_yticks([])
    ax.set_xlabel("Time (ms)", fontweight="bold")
    ax.set_ylabel("Event", fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    handles = [Line2D([], [], marker="o", linestyle="", color=c, label=n) for n, c in used.items()]
    ax.legend(
        handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0),
        ncol=max(len(handles), 1), frameon=False, fontsize=9,
    )
    ax.set_title(f"Sequence timeline — {title}" if title else "Sequence timeline",
                 fontweight="bold", pad=28)
    if isinstance(fig, Figure) and fig.get_layout_engine() is None:
        fig.tight_layout()  # sub-figures and constrained layouts arrange themselves
    return fig


def plot_experiment(source: ExperimentConfig | str | Path, ax: Axes | None = None) -> Figure:
    """Plot the timeline of a parsed config or of an experiment JSON file."""
    name, events = load_events(source)
    return plot_events(events, title=name, ax=ax)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Plot the event timeline of an experiment JSON file.")
    parser.add_argument("config", type=Path)
    parser.add_argument("-o", "--output", type=Path, help="save to file instead of showing a window")
    args = parser.parse_args(argv)
    fig = plot_experiment(args.config)
    if args.output:
        fig.savefig(args.output, dpi=150, bbox_inches="tight")
    else:
        plt.show()


if __name__ == "__main__":
    main()
