"""Plots AWG signals from an .npz file."""

from __future__ import annotations

import argparse
from pathlib import Path

from matplotlib.figure import Figure, SubFigure

from zulf_nmr.hardware.awg import RANGE_FULL_SCALE_A, AwgRange, AwgWaveform, load_awg_npz
from zulf_nmr.utils.visualization import plot_style

_Y_MARGIN = 0.1


def _y_range(values_mA, full_scale_mA: float) -> tuple[float, float]:
    """Y axis of one channel: fitted to its own values (0 always shown, a margin around);
    an empty channel (all zero) gets the standard scale, its range's full scale."""
    low, high = min(float(values_mA.min()), 0.0), max(float(values_mA.max()), 0.0)
    if high - low == 0:
        return -1.1 * full_scale_mA, 1.1 * full_scale_mA
    margin = _Y_MARGIN * (high - low)
    return low - margin, high + margin


def plot_awg(
    waveform: AwgWaveform,
    fig: Figure | SubFigure | None = None,
    acquisition: tuple[float, float, tuple[int, ...]] | None = None,
    show_title: bool = True,
) -> Figure | SubFigure:
    """Draw one row per AWG channel: current in mA versus time in ms after :AWG:RUN.

    The y range of every row fits its values. acquisition = (start_ms, end_ms, ADC
    channels) is drawn as a bar in a strip above the channels and as dashed lines at
    its start and end; the time axis shows the whole waveform and the start of the
    acquisition (zoom out to see a longer acquisition to its end). show_title=False
    leaves out the title (for a panel that shows it itself).
    """
    if fig is None:
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(11, 7))
    n = len(waveform.ranges)
    if acquisition is None:
        axes = fig.subplots(n, 1, sharex=True)
    else:
        strip, *axes = fig.subplots(n + 1, 1, sharex=True, height_ratios=[0.3] + [1] * n)
    t_ms = waveform.time_s() * 1000
    awg_end = waveform.duration_s * 1000
    view_end = awg_end
    if acquisition is not None:
        view_end = max(awg_end, min(acquisition[1], acquisition[0] + 0.25 * awg_end))

    acquisition_colour = plot_style.CATEGORY_COLOURS["Acquisition"]
    for ch, (ax, range_) in enumerate(zip(axes, waveform.ranges), start=1):
        colour = plot_style.series(ch - 1)
        full_scale_mA = RANGE_FULL_SCALE_A[range_] * 1000
        values_mA = waveform.currents_A[:, ch - 1] * 1000
        ax.plot(t_ms, values_mA, color=colour, linewidth=1.4, solid_joinstyle="round", antialiased=True)
        ax.axhline(0, color=plot_style.GUIDE, linewidth=0.8, zorder=0)
        ax.set_ylim(*_y_range(values_mA, full_scale_mA))
        range_name = "LOW" if range_ == AwgRange.LOW else "HIGH"
        ax.set_ylabel(f"CH{ch} (mA)\n{range_name} ±{full_scale_mA:g}", fontsize=9)
        ax.grid(linestyle="--", alpha=0.4)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    if acquisition is not None:
        start, end, channels = acquisition
        strip.broken_barh([(start, end - start)], (0.2, 0.6), facecolors=acquisition_colour, alpha=0.35,
                          edgecolors=acquisition_colour, linewidth=1.0)
        strip.set_ylim(0, 1)
        strip.set_yticks([])
        strip.tick_params(axis="x", bottom=False)
        for side in ("top", "right", "left"):
            strip.spines[side].set_visible(False)
        strip.set_title(f"Acquisition CH {', '.join(map(str, channels))}: {start:g} – {end:g} ms",
                        loc="left", fontsize=8, color=acquisition_colour, fontweight="bold", pad=2)
        for ax in axes:
            ax.axvline(start, color=acquisition_colour, linewidth=1.0, linestyle="--", zorder=1)
            ax.axvline(end, color=acquisition_colour, linewidth=1.0, linestyle="--", zorder=1)
    axes[-1].set_xlim(-0.02 * view_end, view_end * 1.02)
    axes[-1].set_xlabel("Time after :AWG:RUN (ms)", fontweight="bold")
    name = waveform.source_path.name if waveform.source_path else "AWG sequence"
    if show_title:
        fig.suptitle(
            f"AWG sequence — {name}  ({waveform.n_samples} samples, "
            f"{waveform.sample_rate_hz:g} Hz, {waveform.duration_s * 1000:g} ms)",
            fontweight="bold",
        )
    if isinstance(fig, Figure) and fig.get_layout_engine() is None:
        fig.tight_layout()
    return fig


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Plot the channels of an AWG sequence .npz file.")
    parser.add_argument("file", type=Path)
    parser.add_argument("-o", "--output", type=Path, help="save to file instead of showing a window")
    args = parser.parse_args(argv)
    plot_style.apply()
    fig = plot_awg(load_awg_npz(args.file))
    if args.output:
        fig.savefig(args.output, dpi=150, bbox_inches="tight")
    else:
        import matplotlib.pyplot as plt

        plt.show()


if __name__ == "__main__":
    main()
