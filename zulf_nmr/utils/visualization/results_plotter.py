"""Drawing of recorded signals: all channels of a trace in one time plot, and their spectrum.

The plots are made once (setup_time_axes, setup_spectrum_axes) and then only get new
data (Curves.set_data): rebuilding axes, ticks and legend for every chunk of a live
transient costs more than the data itself.
"""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes

from zulf_nmr.utils.visualization import plot_style

CHANNEL_COLOURS = dict(enumerate(plot_style.SERIES, start=1))
AVERAGE_BACKGROUND = "#15151f"
_MAX_POINTS = 200000
_SMOOTH_FACTOR = 8
_DENSE = 4


def smooth_for_display(data: np.ndarray, sample_rate_hz: float) -> tuple[np.ndarray, np.ndarray]:
    """(time, values) with points added between the samples by band-limited (FFT)
    interpolation, so a signal close to the Nyquist frequency is drawn as a smooth
    curve instead of a zigzag. Only for drawing - files and spectra use the samples."""
    n = len(data)
    factor = max(1, min(_SMOOTH_FACTOR, _MAX_POINTS // max(n, 1)))
    if factor == 1 or n < 4:
        step = max(1, n // _MAX_POINTS)
        return np.arange(0, n, step) / sample_rate_hz, data[::step]
    m = n * factor
    spectrum_ = np.fft.rfft(data, axis=0)
    values = np.fft.irfft(spectrum_, m, axis=0) * factor
    return np.arange(m) / (sample_rate_hz * factor), values


def channel_colour(channel: int) -> str:
    return CHANNEL_COLOURS.get(channel, plot_style.MUTED)


def spectrum(data: np.ndarray, sample_rate_hz: float) -> tuple[np.ndarray, np.ndarray]:
    """Amplitude spectrum of every column: (frequencies in Hz, |FFT| of shape (F, columns)).

    The mean of each column is removed first so the DC term does not dominate;
    the scale is the amplitude of a sine (2/N x |FFT|).
    """
    if data.ndim == 1:
        data = data[:, None]
    if len(data) < 2:
        return np.zeros(1), np.zeros((1, data.shape[1]))
    centred = data - data.mean(axis=0)
    freqs = np.fft.rfftfreq(len(data), 1 / sample_rate_hz)
    return freqs, np.abs(np.fft.rfft(centred, axis=0)) * 2 / len(data)


def sample_range(n: int, sample_rate_hz: float, t_range: tuple[float, float] | None,
                 start_s: float) -> tuple[int, int]:
    """Indexes i0:i1 of the samples in t_range (seconds from the sequence start, None = all)."""
    if t_range is None:
        return 0, n
    i0 = min(n, max(0, int(round((t_range[0] - start_s) * sample_rate_hz))))
    i1 = min(n, max(0, int(round((t_range[1] - start_s) * sample_rate_hz))))
    return i0, i1


class Curves:
    """One curve per channel on an axes, drawn with only as much detail as the view needs.

    set_data() takes all the points; refresh() shows the part in view: as lines when there
    are few points per pixel, otherwise as the band between the lowest and the highest
    value of every pixel (a filled polygon - it looks the same as the dense line, but
    draws several times faster).
    """

    def __init__(self, ax: Axes, channels: tuple[int, ...], linewidth: float) -> None:
        self.ax = ax
        ax.set_autoscale_on(False)
        self.lines = []
        self.bands = []
        for channel in channels:
            colour = channel_colour(channel)
            (line,) = ax.plot([], [], color=colour, linewidth=linewidth, label=f"CH{channel}")
            band = ax.fill_between([], [], [], color=colour, edgecolor=colour, linewidth=linewidth)
            band.set_visible(False)
            self.lines.append(line)
            self.bands.append(band)
        self.x = np.zeros(0)
        self.y = np.zeros((0, len(channels)))
        self._detail = None
        self._detail_always = False
        self._fine: tuple[np.ndarray, np.ndarray] | None = None

    def set_data(self, x: np.ndarray, y: np.ndarray, detail=None, detail_always: bool = False) -> None:
        """All points: x ascending, y of shape (len(x), channels). detail() may give the
        same curve with more points (e.g. smoothed samples); it is computed when a view
        shows fewer points than pixels - or at once with detail_always, so even the band of
        a whole trace is exact (its peaks between the samples) - and kept until the next
        set_data."""
        self.x, self.y = x, y.reshape(-1, 1) if y.ndim == 1 else y
        self._detail, self._detail_always, self._fine = detail, detail_always, None

    def y_limits(self, margin: float = 0.05) -> tuple[float, float] | None:
        """Range of all values with a margin around it; None without data."""
        if not len(self.y):
            return None
        low, high = float(self.y.min()), float(self.y.max())
        pad = (high - low) * margin or abs(high) * margin or 1.0
        return low - pad, high + pad

    def refresh(self, xlim: tuple[float, float], width_px: float) -> None:
        """Give the artists the points in xlim for a plot width_px pixels wide."""
        bins = max(int(width_px), 1)
        x, y = self._in_view(self.x, self.y, xlim)
        if self._detail is not None and (self._detail_always or len(x) <= bins):
            if self._fine is None:
                self._fine = self._detail()
            x, y = self._in_view(*self._fine, xlim)
        dense = len(x) > _DENSE * bins
        if dense:
            starts = np.arange(0, len(x), len(x) // bins)
            xb = np.append(x[starts], x[-1])
            low, high = np.minimum.reduceat(y, starts), np.maximum.reduceat(y, starts)
            low, high = np.vstack([low, low[-1:]]), np.vstack([high, high[-1:]])
        for column, (line, band) in enumerate(zip(self.lines, self.bands)):
            if dense:
                band.set_data(xb, low[:, column], high[:, column])
            else:
                line.set_data(x, y[:, column])
            band.set_visible(dense)
            line.set_visible(not dense)

    @staticmethod
    def _in_view(x: np.ndarray, y: np.ndarray, xlim: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
        """The points in xlim and one more on each side, so lines reach the plot's edges."""
        i0 = max(int(np.searchsorted(x, xlim[0])) - 1, 0)
        i1 = min(int(np.searchsorted(x, xlim[1])) + 1, len(x))
        return x[i0:i1], y[i0:i1]


def _style(ax: Axes, average: bool) -> None:
    ax.tick_params(labelsize=8)
    ax.grid(linestyle="--", alpha=0.4)
    if average:
        ax.set_facecolor(AVERAGE_BACKGROUND)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def setup_time_axes(ax: Axes, channels: tuple[int, ...], average: bool, start_s: float) -> Curves:
    """The time plot of a trace without data: labels, legend, the acquisition start line.

    The time axis counts from the start of the sequence; the first sample is at start_s
    (the acquisition start), so the plot is empty before it.
    """
    curves = Curves(ax, channels, linewidth=1.0)
    if start_s > 0:
        ax.axvline(start_s, color=plot_style.MUTED, linewidth=0.8, linestyle="--")
    ax.set_xlabel("Time from the start of the sequence (s)", fontsize=8)
    ax.set_ylabel("Signal", fontsize=8)
    _style(ax, average)
    ax.legend(handles=curves.lines, loc="upper right", fontsize=7, ncol=len(channels),
              frameon=True, framealpha=0.8)
    return curves


def setup_spectrum_axes(ax: Axes, channels: tuple[int, ...], average: bool) -> Curves:
    """The spectrum plot of a trace without data."""
    curves = Curves(ax, channels, linewidth=0.8)
    ax.set_xlabel("Frequency (Hz)", fontsize=8)
    ax.set_ylabel("Amplitude", fontsize=8)
    _style(ax, average)
    return curves


def spectrum_title(i0: int, i1: int, sample_rate_hz: float, start_s: float) -> str:
    span = (f"{start_s + i0 / sample_rate_hz:g} – {start_s + i1 / sample_rate_hz:g} s"
            if i1 > i0 else "no samples in the range")
    return f"Spectrum, {span} ({max(i1 - i0, 0)} samples)"
