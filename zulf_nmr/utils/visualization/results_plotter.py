"""Drawing of recorded signals: all channels of a trace in one time plot, and their spectrum."""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes

# Okabe-Ito, by ADC channel number.
CHANNEL_COLOURS = {1: "#0072B2", 2: "#E69F00", 3: "#009E73", 4: "#CC79A7"}
AVERAGE_BACKGROUND = "#FFF6EE"
_MAX_POINTS = 200000  # longer traces are decimated for drawing only (not for the spectrum)
_SMOOTH_FACTOR = 8  # drawn points per sample (fewer for long traces, see smooth_for_display)


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
    return CHANNEL_COLOURS.get(channel, "#7F7F7F")


def spectrum(data: np.ndarray, sample_rate_hz: float) -> tuple[np.ndarray, np.ndarray]:
    """Amplitude spectrum of every column: (frequencies in Hz, |FFT| of shape (F, columns)).

    The mean of each column is removed first so the DC term does not dominate;
    the scale is the amplitude of a sine (2/N x |FFT|).
    """
    data = data.reshape(len(data), -1)
    if len(data) < 2:
        return np.zeros(1), np.zeros((1, data.shape[1]))
    centred = data - data.mean(axis=0)
    freqs = np.fft.rfftfreq(len(data), 1 / sample_rate_hz)
    return freqs, np.abs(np.fft.rfft(centred, axis=0)) * 2 / len(data)


def draw_time(ax: Axes, data: np.ndarray, sample_rate_hz: float, channels: tuple[int, ...],
              title: str, average: bool = False, xlim: tuple[float, float] | None = None) -> None:
    """All channels of one trace on one plot, one colour per channel, with a legend.

    The curves are smoothed for drawing (smooth_for_display); xlim is the shown
    time range in seconds.
    """
    ax.clear()
    data = data.reshape(len(data), -1)
    t, values = smooth_for_display(data, sample_rate_hz)
    for col, channel in enumerate(channels):
        ax.plot(t, values[:, col], color=channel_colour(channel), linewidth=1.0, label=f"CH{channel}")
    ax.set_title(title, loc="left", fontsize=8, fontweight="bold" if average else None)
    ax.set_xlabel("Time from the start of the acquisition (s)", fontsize=8)
    ax.set_ylabel("Signal", fontsize=8)
    if xlim is not None:
        ax.set_xlim(*xlim)
    _style(ax, average)
    ax.legend(loc="upper right", fontsize=7, ncol=len(channels), frameon=True, framealpha=0.8)


def draw_spectrum(ax: Axes, data: np.ndarray, sample_rate_hz: float, channels: tuple[int, ...],
                  t_range: tuple[float, float] | None = None, average: bool = False) -> None:
    """Spectrum of the samples in t_range (seconds from the start of the acquisition, None = all)."""
    ax.clear()
    data = data.reshape(len(data), -1)
    i0, i1 = 0, len(data)
    if t_range is not None:
        i0 = max(0, int(round(t_range[0] * sample_rate_hz)))
        i1 = min(len(data), int(round(t_range[1] * sample_rate_hz)))
    freqs, amplitude = spectrum(data[i0:i1], sample_rate_hz)
    for col, channel in enumerate(channels):
        ax.plot(freqs, amplitude[:, col], color=channel_colour(channel), linewidth=0.8, label=f"CH{channel}")
    span = f"{i0 / sample_rate_hz:g} – {i1 / sample_rate_hz:g} s" if i1 > i0 else "empty range"
    ax.set_title(f"Spectrum, {span} ({max(i1 - i0, 0)} samples)", loc="left", fontsize=9)
    ax.set_xlabel("Frequency (Hz)", fontsize=8)
    ax.set_ylabel("Amplitude", fontsize=8)
    ax.set_xlim(0, sample_rate_hz / 2)
    _style(ax, average)


def _style(ax: Axes, average: bool) -> None:
    ax.tick_params(labelsize=8)
    ax.grid(linestyle="--", alpha=0.4)
    if average:
        ax.set_facecolor(AVERAGE_BACKGROUND)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
