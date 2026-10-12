"""Plots the event timeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure
from matplotlib.lines import Line2D

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


def plot_events(
    events: Iterable[Event],
    title: str | None = None,
    ax: Axes | None = None,
) -> Figure | SubFigure:
    """Draw one row per event (first event on top): a dot at its time, the
    command on the right, coloured by subsystem."""
    events = list(events)
    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 0.45 * len(events) + 2))
    else:
        fig = ax.figure

    t_max = max((e.time_ms for e in events), default=0) or 1
    rows = range(len(events), 0, -1)
    used: dict[str, str] = {}

    for y, event in zip(rows, events):
        name, colour = event_category(event.action)
        used.setdefault(name, colour)
        ax.hlines(y, 0, event.time_ms, color="#D0D0D0", linewidth=0.8, zorder=1)
        ax.plot(event.time_ms, y, "o", color=colour, markersize=7, zorder=3)
        ax.annotate(
            f"{event.time_ms:g} ms", (event.time_ms, y), xytext=(6, 0),
            textcoords="offset points", va="center", fontsize=8, color="#404040",
        )
        ax.annotate(
            event.action, (1.01, y), xycoords=("axes fraction", "data"),
            va="center", fontsize=8, family="monospace", color=colour,
        )

    ax.set_xlim(-0.03 * t_max, 1.12 * t_max)
    ax.set_ylim(0.3, len(events) + 0.7)
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
