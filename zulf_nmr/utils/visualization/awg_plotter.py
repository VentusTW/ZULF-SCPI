"""Plots AWG signals from an .npz file."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure, SubFigure

from zulf_nmr.hardware.awg import RANGE_FULL_SCALE_A, AwgRange, AwgWaveform, load_awg_npz

CHANNEL_COLOURS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7")  # Okabe-Ito


def plot_awg(waveform: AwgWaveform, fig: Figure | SubFigure | None = None) -> Figure | SubFigure:
    """Draw one row per AWG channel: current in mA versus time in ms.

    The y range of every row is the full scale of its AWG range.
    """
    if fig is None:
        fig = plt.figure(figsize=(11, 7))
    axes = fig.subplots(len(waveform.ranges), 1, sharex=True)
    t_ms = waveform.time_s() * 1000

    for ch, (ax, range_, colour) in enumerate(zip(axes, waveform.ranges, CHANNEL_COLOURS), start=1):
        full_scale_mA = RANGE_FULL_SCALE_A[range_] * 1000
        ax.step(t_ms, waveform.currents_A[:, ch - 1] * 1000, where="post", color=colour, linewidth=1.2)
        ax.axhline(0, color="#D0D0D0", linewidth=0.8, zorder=0)
        ax.set_ylim(-1.1 * full_scale_mA, 1.1 * full_scale_mA)
        range_name = "LOW" if range_ == AwgRange.LOW else "HIGH"
        ax.set_ylabel(f"CH{ch} (mA)\n{range_name} ±{full_scale_mA:g}", fontsize=9)
        ax.grid(linestyle="--", alpha=0.4)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    axes[-1].set_xlabel("Time (ms)", fontweight="bold")
    name = waveform.source_path.name if waveform.source_path else "AWG sequence"
    fig.suptitle(
        f"AWG sequence — {name}  ({waveform.n_samples} samples, "
        f"{waveform.sample_rate_hz:g} Hz, {waveform.duration_s * 1000:g} ms)",
        fontweight="bold",
    )
    if isinstance(fig, Figure) and fig.get_layout_engine() is None:
        fig.tight_layout()  # sub-figures and constrained layouts arrange themselves
    return fig


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Plot the channels of an AWG sequence .npz file.")
    parser.add_argument("file", type=Path)
    parser.add_argument("-o", "--output", type=Path, help="save to file instead of showing a window")
    args = parser.parse_args(argv)
    fig = plot_awg(load_awg_npz(args.file))
    if args.output:
        fig.savefig(args.output, dpi=150, bbox_inches="tight")
    else:
        plt.show()


if __name__ == "__main__":
    main()
