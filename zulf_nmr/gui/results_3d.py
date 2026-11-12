"""Experimental 3D view of the single transients of one set, like MATLAB's mesh and surf:
the transients are joined into one 3D grid - time (or frequency) on the x axis, the
transient number on the y axis and the value as the height, coloured by the height.
Drag the plot to turn it."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable

import numpy as np
from matplotlib import colormaps
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from zulf_nmr.gui import theme
from zulf_nmr.gui.tooltip import Tooltip
from zulf_nmr.utils.visualization import plot_style
from zulf_nmr.utils.visualization.results_plotter import spectrum

MAX_CELLS = 150_000
MAX_LINE_POINTS = 600_000
MODES = ("Signal", "Spectrum")
STYLES = ("Mesh", "Surface", "Lines")
COLOUR_MAPS = ("jet", "turbo", "viridis", "plasma")


def binned(x: np.ndarray, y: np.ndarray, points: int, peak: bool) -> tuple[np.ndarray, np.ndarray]:
    """At most `points` points: the mean of every bin, or its highest value (peak=True, so
    the peaks of a spectrum are kept)."""
    n = len(x)
    if n <= points:
        return x, y
    starts = np.linspace(0, n, points, endpoint=False).astype(int)
    counts = np.diff(np.append(starts, n))
    if peak:
        values = np.maximum.reduceat(y, starts)
    else:
        values = np.add.reduceat(y, starts) / counts
    centres = np.add.reduceat(x, starts) / counts
    return centres, values


def envelope(x: np.ndarray, y: np.ndarray, points: int) -> tuple[np.ndarray, np.ndarray]:
    """At most about `points` points of a line keeping its shape: the lowest and the
    highest value of every bin."""
    n = len(x)
    if n <= points:
        return x, y
    starts = np.linspace(0, n, points // 2, endpoint=False).astype(int)
    lows, highs = np.minimum.reduceat(y, starts), np.maximum.reduceat(y, starts)
    return np.repeat(x[starts], 2), np.column_stack([lows, highs]).ravel()


class ThreeDView(ttk.Frame):
    """A tab with the 3D plot of one set's transients; "Draw" reads them again (also those
    that arrived since).

    "Transients" (from - to, by number) and "Points" follow the data (all transients of
    the set, all samples or frequencies in the range) until the user types a number;
    "Auto" makes them follow again. One transient is drawn as an ordinary 2D plot. A grid too big to turn smoothly (more than MAX_CELLS cells) is drawn with
    fewer points, as the line above the plot says, unless "Full resolution" is ticked."""

    def __init__(self, parent: tk.Misc, transients: Callable[[], list[tuple[int, np.ndarray]]],
                 sample_rate_hz: float, channels: tuple[int, ...], start_s: float, title: str) -> None:
        super().__init__(parent, padding=(0, 10, 0, 0))
        self._transients = transients
        self._rate = sample_rate_hz
        self._channel_numbers = channels
        self._start_s = start_s
        self._channel = tk.StringVar(value=f"CH{channels[0]}")
        self._mode = tk.StringVar(value=MODES[0])
        self._style = tk.StringVar(value=STYLES[0])
        self._colours = tk.StringVar(value=COLOUR_MAPS[0])
        self._from = tk.StringVar()
        self._to = tk.StringVar()
        self._first = tk.StringVar()
        self._last = tk.StringVar()
        self._points = tk.StringVar()
        self._full = tk.BooleanVar(value=False)
        self._auto = True
        self._filling = False

        bar = theme.card(self, padding=10)
        bar.pack(side="top", fill="x")
        top = ttk.Frame(bar.body, style="Card.TFrame")
        top.pack(side="top", fill="x")
        theme.Badge(top, "experimental", theme.WARNING).pack(side="left")
        ttk.Label(top, text=title, style="Card.TLabel", font=theme.mono(10, "bold")).pack(side="left", padx=10)
        ttk.Button(top, text="Draw", style="Accent.TButton", command=self.draw).pack(side="right")
        controls = ttk.Frame(bar.body, style="Card.TFrame")
        controls.pack(side="top", fill="x", pady=(8, 0))
        for label, var, values, width in (("Channel", self._channel, [f"CH{c}" for c in channels], 6),
                                          ("Show", self._mode, MODES, 9), ("Plot", self._style, STYLES, 8),
                                          ("Colours", self._colours, COLOUR_MAPS, 8)):
            theme.caption(controls, label).pack(side="left")
            ttk.Combobox(controls, textvariable=var, values=list(values), state="readonly", width=width,
                         font=theme.mono(9)).pack(side="left", padx=(4, 14))
        self._range_label = theme.caption(controls, "")
        self._range_label.pack(side="left")
        for var in (self._from, self._to):
            ttk.Entry(controls, textvariable=var, width=8, font=theme.mono(9)).pack(side="left", padx=(4, 2))
        theme.caption(controls, "Transients").pack(side="left", padx=(12, 0))
        ttk.Entry(controls, textvariable=self._first, width=5, font=theme.mono(9)).pack(side="left", padx=(4, 2))
        ttk.Label(controls, text="–", style="Card.TLabel").pack(side="left")
        ttk.Entry(controls, textvariable=self._last, width=5, font=theme.mono(9)).pack(side="left", padx=(2, 2))
        self._of_transients = ttk.Label(controls, text="", style="Card.TLabel", foreground=theme.MUTED,
                                        font=theme.mono(8))
        self._of_transients.pack(side="left", padx=(0, 12))
        theme.caption(controls, "Points").pack(side="left")
        ttk.Entry(controls, textvariable=self._points, width=7, font=theme.mono(9)).pack(side="left", padx=(4, 2))
        self._of_points = ttk.Label(controls, text="", style="Card.TLabel", foreground=theme.MUTED,
                                    font=theme.mono(8))
        self._of_points.pack(side="left", padx=(0, 8))
        auto = ttk.Button(controls, text="Auto", style="Small.TButton", command=self._set_auto)
        auto.pack(side="left", padx=(0, 12))
        Tooltip(auto, "All transients of the set and all points in the range")
        full = ttk.Checkbutton(controls, text="Full resolution", variable=self._full, style="Card.TCheckbutton",
                               command=self.draw)
        full.pack(side="left")
        Tooltip(full, "Draw every point even when the plot then turns slowly")
        for var in (self._first, self._last, self._points):
            var.trace_add("write", lambda *_: self._typed())
        for var in (self._channel, self._style, self._colours):
            var.trace_add("write", lambda *_: self.draw())
        self._mode.trace_add("write", lambda *_: self._mode_changed())

        plot = theme.card(self, padding=8)
        plot.pack(side="top", fill="both", expand=True, pady=(10, 0))
        self._info = ttk.Label(plot.body, text="", style="Card.TLabel", foreground=theme.MUTED, font=theme.mono(9))
        self._info.pack(side="top", anchor="w")
        self._figure = Figure(figsize=(10, 7))
        self._canvas = FigureCanvasTkAgg(self._figure, master=plot.body)
        toolbar = NavigationToolbar2Tk(self._canvas, plot.body, pack_toolbar=False)
        toolbar.update()
        theme.style_toolbar(toolbar)
        toolbar.pack(side="bottom", fill="x")
        theme.figure_widget(self._canvas).pack(side="top", fill="both", expand=True)
        self._mode_changed()

    def restyle(self) -> None:
        self.draw()

    def _typed(self) -> None:
        if not self._filling:
            self._auto = False

    def _set_auto(self) -> None:
        self._auto = True
        self.draw()

    def _set_fields(self, first: int, last: int, points: int) -> None:
        self._filling = True
        self._first.set(str(first))
        self._last.set(str(last))
        self._points.set(str(points))
        self._filling = False

    def _available_points(self, data: np.ndarray) -> int:
        """Samples (or frequencies) of one transient in the chosen range."""
        signal = self._mode.get() == "Signal"
        n = len(data)
        xs = self._start_s + np.arange(n) / self._rate if signal else np.fft.rfftfreq(n, 1 / self._rate)
        low, high = self._number(self._from.get(), None), self._number(self._to.get(), None)
        keep = np.ones(len(xs), dtype=bool)
        if low is not None:
            keep &= xs >= low
        if high is not None:
            keep &= xs <= high
        return int(keep.sum())

    def _mode_changed(self) -> None:
        self._range_label.config(text="Time (s)" if self._mode.get() == "Signal" else "Frequency (Hz)")
        self._from.set("")
        self._to.set("")
        self.draw()

    @staticmethod
    def _number(text: str, default: float | None) -> float | None:
        try:
            return float(text.strip().replace(",", "."))
        except ValueError:
            return default

    def _grid(self, chosen: list[tuple[int, np.ndarray]], column: int, points: int, lines: bool | None = None
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(x, transient numbers, values of shape (transients, x)) of the chosen transients;
        with lines (default: the Lines plot) a list of (x, values) per transient instead."""
        if lines is None:
            lines = self._style.get() == "Lines"
        signal = self._mode.get() == "Signal"
        low, high = self._number(self._from.get(), None), self._number(self._to.get(), None)
        rows, x = [], None
        for _number, data in chosen:
            values = data.reshape(-1, 1) if data.ndim == 1 else data
            if signal:
                xs = self._start_s + np.arange(len(values)) / self._rate
                ys = values[:, column]
            else:
                xs, amplitudes = spectrum(values, self._rate)
                ys = amplitudes[:, column]
            keep = np.ones(len(xs), dtype=bool)
            if low is not None:
                keep &= xs >= low
            if high is not None:
                keep &= xs <= high
            xs, ys = xs[keep], ys[keep]
            if lines:
                rows.append(envelope(xs, ys, points))
                continue
            xs, ys = binned(xs, ys, points, peak=not signal)
            if x is None:
                x = xs
            n = min(len(x), len(ys))
            x = x[:n]
            rows = [r[:n] for r in rows]
            rows.append(ys[:n])
        numbers = np.array([n for n, _ in chosen], dtype=float)
        if lines:
            return np.array([]), numbers, rows
        return (x if x is not None else np.array([])), numbers, np.array(rows)

    def draw(self) -> None:
        """Read the transients and draw those in the chosen range: a 3D plot of two or more,
        an ordinary 2D plot of one."""
        transients = self._transients()
        figure = self._figure
        figure.clear()
        figure.set_facecolor(plot_style.PANEL)
        column = self._channel_numbers.index(int(self._channel.get()[2:])) if self._channel.get() else 0
        style = self._style.get()
        numbers_all = [n for n, _ in transients]
        available_points = self._available_points(transients[0][1]) if transients else 0
        if self._auto and transients:
            self._set_fields(numbers_all[0], numbers_all[-1], available_points)
        self._of_transients.config(text=f"of {numbers_all[0]}–{numbers_all[-1]}" if transients else "of none")
        self._of_points.config(text=f"of {available_points}")
        if not transients:
            return self._message("No complete transients in this set yet.")
        first = int(self._number(self._first.get(), numbers_all[0]) or numbers_all[0])
        last = int(self._number(self._last.get(), numbers_all[-1]) or numbers_all[-1])
        first, last = min(first, last), max(first, last)
        in_range = [(n, d) for n, d in transients if first <= n <= last]
        if not in_range:
            return self._message(f"No transients between {first} and {last}.")
        wanted_points = min(max(int(self._number(self._points.get(), available_points) or 0), 4),
                            max(available_points, 4))
        if len(in_range) == 1:
            return self._draw_2d(in_range[0], column, wanted_points)

        most, points = len(in_range), wanted_points
        budget = MAX_LINE_POINTS if style == "Lines" else MAX_CELLS
        reduced = not self._full.get() and most * points > budget
        if reduced:
            points = max(budget // most, min(points, 50))
            if most * points > budget:
                most = max(budget // points, 2)
        step = max(1, int(np.ceil(len(in_range) / most)))
        chosen = in_range[::step]
        if len(chosen) < 2:
            chosen = [in_range[0], in_range[-1]]
        x, numbers, values = self._grid(chosen, column, points)
        ax = figure.add_subplot(projection="3d")
        ax.set_facecolor(plot_style.PANEL)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_pane_color(plot_style.PLOT_BACKGROUND)
            axis._axinfo["grid"]["color"] = plot_style.GRID
            axis._axinfo["grid"]["linewidth"] = 0.4
            axis._axinfo["grid"]["linestyle"] = ":"
        cmap = colormaps[self._colours.get() or "jet"]
        signal = self._mode.get() == "Signal"
        if style == "Lines":
            segments = [np.column_stack([xs, np.full(len(xs), n), ys]) for (xs, ys), n in zip(values, numbers)
                        if len(xs)]
            if not segments:
                return self._message("Nothing in the chosen range.")
            all_z = np.concatenate([s[:, 2] for s in segments])
            norm = Normalize(all_z.min(), all_z.max())
            colours = cmap(norm([s[:, 2].max() for s in segments]))
            ax.add_collection3d(Line3DCollection(segments, colors=colours, linewidths=0.8))
            all_x = np.concatenate([s[:, 0] for s in segments])
            ax.set_xlim(all_x.min(), all_x.max())
            z_low, z_high = all_z.min(), all_z.max()
        else:
            if values.size == 0 or len(x) < 2:
                return self._message("Nothing in the chosen range.")
            grid_x, grid_y = np.meshgrid(x, numbers)
            norm = Normalize(values.min(), values.max())
            if style == "Mesh":
                surface = ax.plot_surface(grid_x, grid_y, values, rstride=1, cstride=1, linewidth=0.7,
                                          antialiased=True, shade=False)
                faces = (values[:-1, :-1] + values[1:, :-1] + values[:-1, 1:] + values[1:, 1:]) / 4
                surface.set_facecolor(plot_style.PLOT_BACKGROUND)
                surface.set_edgecolor(cmap(norm(faces.ravel())))
            else:
                ax.plot_surface(grid_x, grid_y, values, rstride=1, cstride=1, cmap=cmap, norm=norm,
                                linewidth=0.2, edgecolor=plot_style.GRID, antialiased=True, shade=True)
            ax.set_xlim(x.min(), x.max())
            z_low, z_high = values.min(), values.max()
        pad = (z_high - z_low) * 0.05 or 1.0
        ax.set_zlim(z_low - pad, z_high + pad)
        ax.set_ylim(numbers.min(), numbers.max())
        ax.set_xlabel("Time from the start of the sequence (s)" if signal else "Frequency (Hz)", labelpad=10)
        ax.set_ylabel("Transient", labelpad=8)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_zlabel("Signal" if signal else "Amplitude", labelpad=8)
        ax.tick_params(labelsize=7)
        ax.view_init(elev=30, azim=-37.5)
        figure.colorbar(ax.collections[0] if style == "Surface" else _mappable(cmap, norm), ax=ax, shrink=0.55,
                        pad=0.08).ax.tick_params(labelsize=7)
        figure.subplots_adjust(left=0, right=0.95, bottom=0, top=1)
        shown = (f"{first}–{last}" if step == 1 else f"every {step}. of {first}–{last}")
        drawn_points = len(x) if style != "Lines" else max(len(xs) for xs, _ys in values)
        note = (f" - fewer than the {len(in_range)} x {wanted_points} asked for, so that the plot turns "
                f"smoothly (tick Full resolution for all)" if reduced else "")
        self._info.config(text=f"{self._channel.get()} · {len(numbers)} transient(s) ({shown}) x {drawn_points} "
                               f"points{note} · drag the plot to turn it, right-drag to zoom",
                          foreground=theme.WARNING if reduced else theme.MUTED)
        self._canvas.draw_idle()

    def _draw_2d(self, transient: tuple[int, np.ndarray], column: int, points: int) -> None:
        """One transient: an ordinary plot of its value against time (or frequency)."""
        number, _data = transient
        signal = self._mode.get() == "Signal"
        _x, _numbers, rows = self._grid([transient], column, points, lines=True)
        xs, ys = rows[0]
        if not len(xs):
            return self._message("Nothing in the chosen range.")
        ax = self._figure.add_subplot()
        ax.plot(xs, ys, color=plot_style.series(column), linewidth=1.0)
        ax.set_xlim(xs.min(), xs.max())
        ax.set_xlabel("Time from the start of the sequence (s)" if signal else "Frequency (Hz)")
        ax.set_ylabel("Signal" if signal else "Amplitude")
        ax.set_title(f"Transient {number} - one transient is drawn as an ordinary plot", loc="left", fontsize=9,
                     color=plot_style.MUTED)
        ax.grid(linestyle="--", alpha=0.4)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        self._figure.subplots_adjust(left=0.07, right=0.98, bottom=0.1, top=0.92)
        self._info.config(text=f"{self._channel.get()} · transient {number} x {len(xs)} points (2D: one transient)",
                          foreground=theme.MUTED)
        self._canvas.draw_idle()

    def _message(self, text: str) -> None:
        self._info.config(text=text, foreground=theme.MUTED)
        self._canvas.draw_idle()


def _mappable(cmap, norm) -> ScalarMappable:
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    return mappable
