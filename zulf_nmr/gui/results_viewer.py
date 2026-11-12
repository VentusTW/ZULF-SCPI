"""Results panel: tabs of runs, each with every transient and the running average of each
set, with spectra - live while the queue runs, or read back from a saved run."""

from __future__ import annotations

import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import numpy as np
from matplotlib.backend_tools import Cursors
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector

from zulf_nmr.experiment.schedule import QueueResult, ResultGroup, Transient
from zulf_nmr.gui import theme
from zulf_nmr.gui.results_3d import ThreeDView
from zulf_nmr.gui.tooltip import Tooltip
from zulf_nmr.utils.visualization.results_plotter import (
    sample_range,
    setup_spectrum_axes,
    setup_time_axes,
    smooth_for_display,
    spectrum,
    spectrum_title,
)

ROW_SIZE_IN = (11.0, 3.6)
LIVE_AVERAGE_INTERVAL_S = 1.0
EDGE_GRAB_PX = 10
ZOOM_STEP = 1.25
PAN_STEP = 0.2
_MARGIN_PX = {"left": 64, "right": 14, "top": 40, "bottom": 40, "gap": 74}
_TIME_SHARE = 0.6
_ROW_BORDER_PX = 1


class RowToolbar(NavigationToolbar2Tk):
    """Matplotlib's toolbar (Home, Back, Forward, Pan, Zoom, Save) plus measuring cursors.

    Home shows the whole trace again and lets a growing trace widen its time axis.
    """

    toolitems = [item for item in NavigationToolbar2Tk.toolitems if item[0] != "Subplots"]

    def __init__(self, canvas, window, row: TraceRow) -> None:
        self._row = row
        super().__init__(canvas, window, pack_toolbar=False)
        self.config(height=36)
        self.cursors_button = self._Button("Cursors", None, True, row.toggle_cursors)
        Tooltip(self.cursors_button, "Measuring cursors A and B: click or drag on a plot")
        theme.style_toolbar(self, theme.PANEL)
        self.cursors_button.config(font=theme.mono(9), padx=6)

    def home(self, *args) -> None:
        self._row.reset_view()

    def pan(self, *args) -> None:
        self._row.set_cursors(False)
        super().pan(*args)

    def zoom(self, *args) -> None:
        self._row.set_cursors(False)
        super().zoom(*args)


class TraceRow:
    """One trace: all channels in time (left) and the spectrum of the selected range (right).

    The time axis runs from the sequence start (0) to the newest sample and widens while
    a transient is received. The toolbar above the plots zooms (to a rectangle), pans,
    goes back, saves and turns on two measuring cursors - on both plots; Home shows
    everything again. Ctrl + wheel zooms either plot, Shift + wheel moves it along its
    axis; a zoomed spectrum stays zoomed while its range or data change. Drag on the time plot to choose the range of the spectrum;
    drag the selection to move it; a click (left, without dragging, or right) removes it -
    the spectrum is then of the whole trace. The header names the trace; its +/- button
    hides or shows the plots.
    """

    def __init__(self, master: tk.Misc, sample_rate_hz: float, channels: tuple[int, ...], average: bool,
                 total_samples: int | None = None, on_scroll=None, start_s: float = 0.0,
                 expanded: bool = True) -> None:
        self.rate = sample_rate_hz
        self.channels = channels
        self.average = average
        self.start_s = start_s
        self.total_s = start_s + (total_samples or 0) / sample_rate_hz
        self.data = np.empty((0, len(channels)))
        self.title = ""
        self.live = False
        self.t_range: tuple[float, float] | None = None
        self._user_limits: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {}
        self._auto_ylim: tuple[float, float] | None = None
        self._drawing = False
        self._spectrum = (np.zeros(1), np.zeros((1, len(channels))))
        self.on_scroll = on_scroll
        self._size: tuple[int, int] | None = None

        self.frame = tk.Frame(master, background=theme.PANEL, highlightthickness=1,
                              highlightbackground=theme.ACCENT_SOFT if average else theme.BORDER,
                              highlightcolor=theme.ACCENT_SOFT if average else theme.BORDER)
        self._header = ttk.Frame(self.frame, style="Card.TFrame", padding=(6, 4))
        self._header.pack(side="top", fill="x")
        self._toggle = ttk.Button(self._header, text="+", width=3, style="Small.TButton", command=self.toggle)
        self._toggle.pack(side="left")
        if average:
            theme.Badge(self._header, "average", theme.ACCENT).pack(side="left", padx=(8, 0))
        self._name = ttk.Label(self._header, text="", style="Card.TLabel", cursor="hand2",
                               font=theme.mono(9, "bold" if average else "normal"),
                               foreground=theme.TEXT if average else theme.MUTED)
        self._name.pack(side="left", padx=6)
        self._name.bind("<Button-1>", lambda _e: self.toggle())
        self._readout = ttk.Label(self.frame, text="", style="Card.TLabel", font=theme.mono(9),
                                  foreground=theme.CURSOR)
        for w in (self._header, self._toggle, self._name, self._readout):
            w.bind("<MouseWheel>", lambda e: self.on_scroll and self.on_scroll(e.delta / 120))

        self.cursors_on = False
        self._cursor_x: dict[str, list[float | None]] = {"time": [None, None], "spec": [None, None]}
        self._cursor_plot = "time"

        self.figure: Figure | None = None
        self.expanded = False
        self.expand(expanded)

    def _build_plots(self) -> None:
        self.figure = Figure(figsize=ROW_SIZE_IN)
        self.ax_time = self.figure.add_axes((0, 0, 1, 1))
        self.ax_spec = self.figure.add_axes((0, 0, 1, 1))
        self._time_curves = setup_time_axes(self.ax_time, self.channels, self.average, self.start_s)
        self._spec_curves = setup_spectrum_axes(self.ax_spec, self.channels, self.average)
        self._watch_limits(self.ax_time, "time")
        self._watch_limits(self.ax_spec, "spec")
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.frame)
        self.widget = theme.figure_widget(self.canvas)
        if self._size is not None:
            self.widget.config(width=self._size[0], height=self._size[1])
        self.toolbar = RowToolbar(self.canvas, self._header, self)
        self.toolbar.bind("<MouseWheel>", lambda e: self.on_scroll and self.on_scroll(e.delta / 120))
        self._layout()

        self.selector: SpanSelector | None = None
        self._new_selector()
        self.canvas.mpl_connect("resize_event", lambda _e: self._layout())
        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        self._cursor = Cursors.POINTER
        self._drag_cursor: Cursors | None = None
        self._dragged_cursor: tuple[str, int] | None = None
        self._background = None
        if self.cursors_on:
            self.canvas.widgetlock(self)
            self.toolbar.cursors_button.select()
            self._draw_cursors("time")

    def _drop_plots(self) -> None:
        """Free the figure of a closed row; the data, the zoom and the cursors stay."""
        if self.figure is None:
            return
        self.selector.disconnect_events()
        self.figure.clear()
        self.toolbar.destroy()
        self.widget.destroy()
        self.figure = self.canvas = self.widget = self.toolbar = self.selector = None

    def restyle(self) -> None:
        """Make the plots again in the colours of the current theme; the zoom and the
        selected range stay, the measuring cursors are turned off."""
        if self.figure is None:
            return
        if self.cursors_on:
            self.set_cursors(False)
        expanded = self.expanded
        self._readout.pack_forget()
        self._drop_plots()
        self.expanded = False
        self._toggle.config(text="+")
        if expanded:
            self.expand(True)

    @property
    def header(self) -> ttk.Frame:
        return self._header

    @property
    def name_label(self) -> ttk.Label:
        return self._name

    def set_size(self, width: int, height: int) -> None:
        """Size of the plots in pixels (the figure follows its widget)."""
        self._size = (width, height)
        if self.figure is not None:
            self.widget.config(width=width, height=height)

    def _layout(self) -> None:
        """Place both plots with fixed margins (in pixels) for the current figure size."""
        width, height = self.figure.bbox.width, self.figure.bbox.height
        m = _MARGIN_PX
        plots = max(width - m["left"] - m["right"] - m["gap"], 50)
        time_w, spec_w = plots * _TIME_SHARE, plots * (1 - _TIME_SHARE)
        bottom, plot_h = m["bottom"] / height, max(height - m["top"] - m["bottom"], 20) / height
        self.ax_time.set_position((m["left"] / width, bottom, time_w / width, plot_h))
        self.ax_spec.set_position(((m["left"] + time_w + m["gap"]) / width, bottom, spec_w / width, plot_h))
        self._refresh_curves()

    def show(self, data: np.ndarray, title: str, live: bool = False) -> None:
        """Draw new data (a new trace, or an updated average); the selected range, the
        zoom and the cursors are kept. live: the trace is still growing."""
        self.data, self.title, self.live = data.reshape(-1, 1) if data.ndim == 1 else data, title, live
        self._name.config(text=title.replace("\n", "  \u2014  "))
        if self.figure is None:
            return
        self._background = None
        self.ax_time.set_title(title, loc="left", fontsize=8, fontweight="bold" if self.average else None,
                               color=theme.TEXT if self.average else theme.MUTED)
        data = self.data

        def smoothed() -> tuple[np.ndarray, np.ndarray]:
            t, values = smooth_for_display(data, self.rate)
            return t + self.start_s, values

        self._time_curves.set_data(self.start_s + np.arange(len(data)) / self.rate, data, smoothed,
                                   detail_always=not live)
        self._set_time_limits()
        self._update_spectrum()
        self._update_readout()
        self.canvas.draw_idle()

    def _time_xlim(self) -> tuple[float, float]:
        """From the sequence start (0) to the newest sample: the axis widens as the samples
        of a transient arrive."""
        end = self.start_s + len(self.data) / self.rate
        if end <= self.start_s:
            end = max(self.total_s, self.start_s + 1 / self.rate)
        return 0.0, end

    def _set_limits(self, ax, plot: str, xlim, ylim) -> None:
        """Limits set by the program: the user's own if there are any, else xlim and ylim."""
        self._drawing = True
        try:
            if plot in self._user_limits:
                xlim, ylim = self._user_limits[plot]
            ax.set_xlim(*xlim)
            if ylim is not None:
                ax.set_ylim(*ylim)
        finally:
            self._drawing = False
        self._refresh_curves()

    def _set_time_limits(self) -> None:
        ylim = self._time_curves.y_limits()
        if ylim is not None and self._auto_ylim is not None:
            ylim = min(ylim[0], self._auto_ylim[0]), max(ylim[1], self._auto_ylim[1])
        if ylim is not None:
            self._auto_ylim = ylim
        self._set_limits(self.ax_time, "time", self._time_xlim(), ylim)

    def _update_spectrum(self) -> None:
        """Spectrum of the selected range (of all samples without a selection)."""
        i0, i1 = sample_range(len(self.data), self.rate, self.t_range, self.start_s)
        self._spectrum = spectrum(self.data[i0:i1], self.rate)
        self._spec_curves.set_data(*self._spectrum)
        self.ax_spec.set_title(spectrum_title(i0, i1, self.rate, self.start_s), loc="left", fontsize=8,
                               color=theme.MUTED)
        ylim = self._spec_curves.y_limits()
        self._set_limits(self.ax_spec, "spec", (0, self.rate / 2), ylim and (min(0.0, ylim[0]), ylim[1]))
        self._draw_cursors("spec")

    def _refresh_curves(self) -> None:
        """Give the curves the points of the ranges now in view (after new data, a zoom,
        a pan or a resize): drawing then costs the same at any zoom."""
        self._time_curves.refresh(self.ax_time.get_xlim(), self.ax_time.bbox.width)
        self._spec_curves.refresh(self.ax_spec.get_xlim(), self.ax_spec.bbox.width)

    def _watch_limits(self, ax, plot: str) -> None:
        """Remember limits the user sets (toolbar, wheel) and thin the curves for them."""
        def changed(_ax) -> None:
            if not self._drawing:
                self._user_limits[plot] = (ax.get_xlim(), ax.get_ylim())
                self._refresh_curves()
        ax.callbacks.connect("xlim_changed", changed)
        ax.callbacks.connect("ylim_changed", changed)

    def reset_view(self) -> None:
        """Home: both plots show all their data again and follow it."""
        self._user_limits.clear()
        self._auto_ylim = None
        self.show(self.data, self.title, self.live)

    def _new_selector(self) -> None:
        """A fresh SpanSelector showing t_range, with no drag in progress."""
        if self.selector is not None:
            self.selector.disconnect_events()
            self.selector.set_visible(False)
        self.selector = SpanSelector(
            self.ax_time, self._on_select, "horizontal", useblit=True, interactive=True,
            drag_from_anywhere=True, button=1, grab_range=EDGE_GRAB_PX,
            props=dict(facecolor=theme.SPAN, alpha=0.25),
        )
        if self.t_range is not None:
            self.selector.extents = self.t_range

    def expand(self, expanded: bool = True) -> None:
        """Show (or, with expanded=False, hide) the plots and the toolbar; the header stays."""
        if expanded == self.expanded:
            return
        self.expanded = expanded
        if expanded:
            built = self.figure is None
            if built:
                self._build_plots()
            self.toolbar.pack(side="left", fill="x", expand=True, padx=(12, 0))
            self.widget.pack(side="top", anchor="w")
            if self.cursors_on:
                self._readout.pack(side="top", anchor="w", padx=6, before=self.widget)
            if built and self.title:
                self.show(self.data, self.title, self.live)
        else:
            self._readout.pack_forget()
            if self.average:
                self.toolbar.pack_forget()
                self.widget.pack_forget()
            else:
                self._drop_plots()
        self._toggle.config(text="−" if expanded else "+")

    def toggle(self) -> None:
        self.expand(not self.expanded)

    def _on_select(self, t0: float, t1: float) -> None:
        if t1 - t0 <= 0:
            self.clear_selection()
            return
        self.t_range = (max(t0, self.start_s), t1)
        self._update_spectrum()
        self._update_readout()
        self.canvas.draw_idle()

    def clear_selection(self) -> None:
        """Remove the selected range; the spectrum is of the whole trace again."""
        self.t_range = None
        if self.selector is not None:
            self.selector.clear()
        self._update_spectrum()
        self._update_readout()
        self.canvas.draw_idle()

    def _tool_active(self) -> bool:
        """The toolbar's Pan or Zoom has the mouse."""
        return bool(self.toolbar.mode)

    def _on_press(self, event) -> None:
        if self._tool_active():
            return
        if self.cursors_on:
            self._press_cursor(event)
        elif event.button == 3 and event.inaxes is self.ax_time:
            self.clear_selection()
        elif event.button == 1 and event.dblclick and self._hover_cursor(event) == Cursors.RESIZE_HORIZONTAL:
            left, right = (self.ax_time.transData.transform((x, 0))[0] for x in self.selector.extents)
            edge = 0 if abs(event.x - left) <= abs(event.x - right) else 1
            self.widget.after_idle(self._edit_edge, edge)
        elif event.button == 1:
            self._drag_cursor = self._hover_cursor(event)
            self.widget.after_idle(self._set_cursor, self._drag_cursor, True)

    def _on_release(self, event) -> None:
        if self._tool_active():
            return
        self._end_cursor_drag()
        self._drag_cursor = None
        self.widget.after_idle(self._set_cursor, self._hover_cursor(event), True)

    def _on_motion(self, event) -> None:
        """Mouse cursor over the time plot: double arrow at an edge of the selection (resize it),
        hand inside it (move it), arrow elsewhere. Kept while the selection is dragged.
        With the measuring cursors on: a cross, a double arrow at a cursor; dragging moves it."""
        if self._tool_active():
            return
        if self._dragged_cursor is not None:
            plot, index = self._dragged_cursor
            if event.inaxes is self._axes(plot) and event.xdata is not None:
                self._move_cursor(plot, index, event.xdata)
            return
        if event.button is not None:
            if self._drag_cursor is not None:
                self._set_cursor(self._drag_cursor)
            return
        self._set_cursor(self._hover_cursor(event))

    def _edit_edge(self, edge: int) -> None:
        """Small window to type the position of an edge (0 = left, 1 = right) of the range."""
        if self.selector is None or self.t_range is None:
            return
        EdgeDialog(self.widget, "Start" if edge == 0 else "End", self.selector.extents, edge,
                   (self.start_s, max(self.total_s, self.start_s + len(self.data) / self.rate)),
                   self._apply_range, self._end_edit)

    def _end_edit(self) -> None:
        """After OK or Cancel: the dialog took the mouse release of the double click, so the
        selector still believes an edge is being dragged - start a fresh one, not dragging."""
        self._drag_cursor = None
        self._new_selector()
        self._set_cursor(Cursors.POINTER, True)
        self.canvas.draw_idle()

    def _apply_range(self, t0: float, t1: float) -> None:
        """Select the range t0..t1 (seconds from the sequence start) for the spectrum."""
        if self.selector is not None:
            self.selector.extents = (t0, t1)
        self._on_select(t0, t1)

    def _hover_cursor(self, event) -> Cursors:
        if self.cursors_on:
            if event.inaxes in (self.ax_time, self.ax_spec):
                plot = "time" if event.inaxes is self.ax_time else "spec"
                near = self._nearest_cursor(plot, event.x)
                return Cursors.RESIZE_HORIZONTAL if near is not None else Cursors.SELECT_REGION
            return Cursors.POINTER
        if event.inaxes is self.ax_time and self.t_range is not None and self.selector is not None:
            left, right = (self.ax_time.transData.transform((x, 0))[0] for x in self.selector.extents)
            if abs(event.x - left) <= EDGE_GRAB_PX or abs(event.x - right) <= EDGE_GRAB_PX:
                return Cursors.RESIZE_HORIZONTAL
            if left < event.x < right:
                return Cursors.HAND
        return Cursors.POINTER

    def _set_cursor(self, cursor: Cursors, force: bool = False) -> None:
        if force or cursor != self._cursor:
            self._cursor = cursor
            self.canvas.set_cursor(cursor)

    def toggle_cursors(self) -> None:
        self.set_cursors(not self.cursors_on)

    def set_cursors(self, on: bool) -> None:
        """Turn the measuring cursors on or off. While they are on, a click on a plot puts
        cursor A, the next one B; a click near a cursor, or a drag, moves it."""
        button = self.toolbar.cursors_button
        if on != self.cursors_on:
            if on:
                if self.toolbar.mode:
                    (self.toolbar.pan if self.toolbar.mode.name == "PAN" else self.toolbar.zoom)()
                self.canvas.widgetlock(self)
                if self.expanded:
                    self._readout.pack(side="top", anchor="w", padx=6, before=self.widget)
            else:
                if self.canvas.widgetlock.isowner(self):
                    self.canvas.widgetlock.release(self)
                self._readout.pack_forget()
                self._cursor_x = {"time": [None, None], "spec": [None, None]}
                self._draw_cursors("time")
                self._draw_cursors("spec")
                self.canvas.draw_idle()
            self.cursors_on = on
            self._update_readout()
        if on:
            button.select()
        else:
            button.deselect()

    def _axes(self, plot: str):
        return self.ax_time if plot == "time" else self.ax_spec

    def _nearest_cursor(self, plot: str, x_px: float) -> int | None:
        """Index of the cursor within EDGE_GRAB_PX of the mouse, if any."""
        ax = self._axes(plot)
        best = None
        for index, x in enumerate(self._cursor_x[plot]):
            if x is not None:
                distance = abs(ax.transData.transform((x, 0))[0] - x_px)
                if distance <= EDGE_GRAB_PX and (best is None or distance < best[0]):
                    best = (distance, index)
        return None if best is None else best[1]

    def _press_cursor(self, event) -> None:
        if event.button != 1 or event.inaxes not in (self.ax_time, self.ax_spec) or event.xdata is None:
            return
        plot = "time" if event.inaxes is self.ax_time else "spec"
        index = self._nearest_cursor(plot, event.x)
        if index is None:
            a, b = self._cursor_x[plot]
            index = 0 if a is None else 1 if b is None else (0 if abs(a - event.xdata) <= abs(b - event.xdata) else 1)
        self._dragged_cursor = (plot, index)
        self._cursor_x[plot][index] = event.xdata
        self._cursor_plot = plot
        self._draw_cursors(plot, animated=True)
        self.canvas.draw()
        self._background = self.canvas.copy_from_bbox(self.figure.bbox)
        self._blit_cursors(plot)
        self._update_readout()

    def _end_cursor_drag(self) -> None:
        if self._dragged_cursor is not None:
            self._draw_cursors(self._dragged_cursor[0])
            self.canvas.draw_idle()
        self._dragged_cursor = None
        self._background = None

    def _move_cursor(self, plot: str, index: int, x: float) -> None:
        self._cursor_x[plot][index] = x
        self._cursor_plot = plot
        if self._background is not None:
            self._draw_cursors(plot, animated=True)
            self._blit_cursors(plot)
        else:
            self._draw_cursors(plot)
            self.canvas.draw_idle()
        self._update_readout()

    def _blit_cursors(self, plot: str) -> None:
        ax = self._axes(plot)
        self.canvas.restore_region(self._background)
        for artist in (*ax.lines, *ax.texts):
            if artist.get_gid() == "cursor":
                ax.draw_artist(artist)
        self.canvas.blit(self.figure.bbox)

    def _draw_cursors(self, plot: str, animated: bool = False) -> None:
        ax = self._axes(plot)
        for artist in [a for a in (*ax.lines, *ax.texts) if a.get_gid() == "cursor"]:
            artist.remove()
        for name, x in zip("AB", self._cursor_x[plot]):
            if x is None:
                continue
            line = ax.axvline(x, color=theme.CURSOR, linewidth=1.0, linestyle="-" if name == "A" else "--",
                              animated=animated)
            label = ax.annotate(name, (x, 1.0), xycoords=("data", "axes fraction"), xytext=(3, -3),
                                textcoords="offset points", va="top", fontsize=8, color=theme.CURSOR,
                                fontweight="bold", annotation_clip=True, animated=animated)
            line.set_gid("cursor")
            label.set_gid("cursor")

    def _values_at(self, plot: str, x: float) -> list[float]:
        """Value of every channel at x: the signal at time x, or the amplitude at frequency x."""
        if plot == "time":
            if len(self.data) == 0:
                return [float("nan")] * len(self.channels)
            xs = self.start_s + np.arange(len(self.data)) / self.rate
            columns = self.data
        else:
            xs, columns = self._spectrum
        return [float(np.interp(x, xs, columns[:, c], left=np.nan, right=np.nan))
                for c in range(len(self.channels))]

    def _update_readout(self) -> None:
        if not self.cursors_on:
            return
        plot = self._cursor_plot
        positions = self._cursor_x[plot]
        if all(x is None for x in positions):
            self._readout.config(text="Cursors: click on a plot to put cursor A, then B; drag a cursor to move it.")
            return
        unit = "s" if plot == "time" else "Hz"
        parts = []
        for name, x in zip("AB", positions):
            if x is not None:
                values = "  ".join(f"CH{ch} {v:.4g}" for ch, v in zip(self.channels, self._values_at(plot, x)))
                parts.append(f"{name}: {x:.6g} {unit}  [{values}]")
        if None not in positions:
            delta = positions[1] - positions[0]
            if plot == "time":
                inverse = f" (1/Δt = {1 / abs(delta):.4g} Hz)" if delta else ""
                parts.append(f"Δt = {delta:.6g} s{inverse}")
            else:
                parts.append(f"Δf = {delta:.6g} Hz")
        title = "Time" if plot == "time" else "Spectrum"
        self._readout.config(text=f"{title} cursors — " + "    ".join(parts))

    def _on_scroll(self, event) -> None:
        state = getattr(event.guiEvent, "state", 0)
        ctrl, shift = bool(state & 0x4), bool(state & 0x1)
        ax = event.inaxes
        if not (ctrl or shift) or ax not in (self.ax_time, self.ax_spec):
            if self.on_scroll:
                self.on_scroll(event.step)
            return
        start, end = ax.get_xlim()
        width = end - start
        if ax is self.ax_time:
            limit, narrowest = max(self.total_s, self.start_s + len(self.data) / self.rate), 5 / self.rate
        else:
            freqs = self._spectrum[0]
            limit = self.rate / 2
            narrowest = 5 * (freqs[1] - freqs[0]) if len(freqs) > 1 else limit / 1000
        if ctrl:
            new_width = width / ZOOM_STEP if event.step > 0 else width * ZOOM_STEP
            new_width = min(max(new_width, narrowest), limit)
            x = event.xdata if event.xdata is not None else start + width / 2
            start = x - (x - start) * new_width / width
            width = new_width
        else:
            start -= event.step * PAN_STEP * width
        width = min(width, limit)
        start = max(0.0, min(start, limit - width))
        ax.set_xlim(start, start + width)
        self.canvas.draw_idle()


class EdgeDialog(tk.Toplevel):
    """Entry for the exact position of one edge of the spectrum range, opened at the mouse."""

    def __init__(self, master: tk.Misc, name: str, extents: tuple[float, float], edge: int,
                 limits: tuple[float, float], on_apply, on_close=None) -> None:
        super().__init__(master, background=theme.BACKGROUND)
        self._on_close = on_close
        self.bind("<Destroy>", lambda e: e.widget is self and self._on_close and self._on_close())
        self.title(f"{name} of the range")
        self.resizable(False, False)
        self.transient(master.winfo_toplevel())
        self._extents, self._edge, self._limits, self._on_apply = extents, edge, limits, on_apply

        frame = ttk.Frame(self, padding=10)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text=f"{name} of the range (s from the sequence start):").grid(
            row=0, column=0, columnspan=3, sticky="w")
        self._value = tk.StringVar(value=f"{extents[edge]:.6g}")
        entry = ttk.Entry(frame, textvariable=self._value, width=14)
        entry.grid(row=1, column=0, sticky="w", pady=6)
        ttk.Button(frame, text="OK", style="Accent.TButton", command=self._ok).grid(row=1, column=2, sticky="e")
        ttk.Button(frame, text="Cancel", command=self.destroy).grid(row=1, column=1, sticky="e", padx=6)
        self._error = ttk.Label(frame, text=f"Allowed: {limits[0]:.6g} - {limits[1]:.6g} s",
                                foreground=theme.MUTED)
        self._error.grid(row=2, column=0, columnspan=3, sticky="w")

        entry.bind("<Return>", lambda _e: self._ok())
        self.bind("<Escape>", lambda _e: self.destroy())
        x, y = self.winfo_pointerxy()
        self.geometry(f"+{x + 12}+{y + 12}")
        theme.title_bar(self)
        entry.focus_set()
        entry.select_range(0, "end")
        self.grab_set()

    def _ok(self) -> None:
        text = self._value.get().strip().replace(",", ".")
        try:
            value = float(text)
        except ValueError:
            return self._fail(f"{text!r} is not a number")
        low, high = self._limits
        if not low <= value <= high:
            return self._fail(f"Must be within {low:.6g} - {high:.6g} s")
        t0, t1 = self._extents
        t0, t1 = (value, t1) if self._edge == 0 else (t0, value)
        if t1 <= t0:
            other = "end" if self._edge == 0 else "start"
            return self._fail(f"Must be {'before' if self._edge == 0 else 'after'} the {other} ({self._extents[1 - self._edge]:.6g} s)")
        self.destroy()
        self._on_apply(t0, t1)

    def _fail(self, message: str) -> None:
        self._error.config(text=message, foreground=theme.DANGER)
        self.bell()


class TransientBrowser:
    """The single transients of one set in one row: "−" and "+" step through them, a
    number can be typed (Enter), "Latest" follows the newest one while the queue runs.

    The data of every transient are kept; only the one shown is drawn, in the row's
    figure that is made once.
    """

    def __init__(self, master: tk.Misc, group: ResultGroup, total_samples: int, on_scroll,
                 expanded: bool = True, on_3d=None) -> None:
        self.set_index = group.set_index
        self.group = group
        self.expected = group.expected_transients
        self.row = TraceRow(master, group.sample_rate_hz, group.channels, average=False,
                            total_samples=total_samples, on_scroll=on_scroll, start_s=group.acquisition_start_s,
                            expanded=expanded)
        self._data: dict[int, tuple[np.ndarray, str]] = {}
        self._live: tuple[int, np.ndarray, int] | None = None
        self._shown: int | None = None
        self.follow = True

        picker = ttk.Frame(self.row.header, style="Card.TFrame")
        picker.pack(side="left", padx=(4, 8), after=self.row.name_label)
        theme.caption(picker, "Transient").pack(side="left", padx=(0, 6))
        ttk.Button(picker, text="−", style="Small.TButton", width=3, command=lambda: self.step(-1)).pack(side="left")
        self._number = tk.StringVar()
        self._entry = ttk.Entry(picker, textvariable=self._number, width=6, font=theme.mono(10), justify="center")
        self._entry.pack(side="left", padx=4)
        self._entry.bind("<Return>", lambda _e: self._typed())
        self._entry.bind("<KP_Enter>", lambda _e: self._typed())
        ttk.Button(picker, text="+", style="Small.TButton", width=3, command=lambda: self.step(1)).pack(side="left")
        self._of = ttk.Label(picker, text="", style="Card.TLabel", foreground=theme.MUTED, font=theme.mono(9))
        self._of.pack(side="left", padx=(6, 6))
        self._latest = ttk.Button(picker, text="Latest", style="Small.TButton", command=self.follow_latest)
        self._latest.pack(side="left")
        Tooltip(self._latest, "Show the newest transient, and keep following it while the queue runs")
        if on_3d is not None:
            three_d = ttk.Button(picker, text="3D", style="Small.TButton", command=lambda: on_3d(self))
            three_d.pack(side="left", padx=(6, 0))
            Tooltip(three_d, "Experimental: all transients of this set in one 3D plot (transient number on z)")
        Tooltip(self._entry, "Type a transient number and press Enter")
        for widget in (picker, *picker.winfo_children()):
            widget.bind("<MouseWheel>", lambda e: on_scroll and on_scroll(e.delta / 120), add="+")
        self._update_picker()

    def complete(self) -> list[tuple[int, np.ndarray]]:
        """(number, data) of every complete transient, in order."""
        return [(n, self._data[n][0]) for n in sorted(self._data)]

    def numbers(self) -> list[int]:
        """The transients that can be shown: the saved ones and the one being acquired."""
        numbers = set(self._data)
        if self._live is not None:
            numbers.add(self._live[0])
        return sorted(numbers)

    def add(self, number: int, data: np.ndarray, file_name: str) -> None:
        """A complete transient."""
        self._data[number] = (data, file_name)
        if self._live is not None and self._live[0] == number:
            self._live = None
        if self.follow or self._shown == number or self._shown is None:
            self._show(number)
        else:
            self._update_picker()

    def live(self, number: int, data: np.ndarray, total_samples: int) -> None:
        """A transient still being acquired (data = samples received so far)."""
        self._live = (number, data, total_samples)
        if self.follow or self._shown == number:
            self._show(number)
        else:
            self._update_picker()

    def step(self, direction: int) -> None:
        numbers = self.numbers()
        if not numbers:
            return
        current = self._shown if self._shown is not None else numbers[-1]
        later = [n for n in numbers if (n > current if direction > 0 else n < current)]
        if later:
            self.follow = False
            self._show(later[0] if direction > 0 else later[-1])

    def follow_latest(self) -> None:
        self.follow = True
        numbers = self.numbers()
        if numbers:
            self._show(numbers[-1])

    def show_number(self, number: int) -> bool:
        """Show transient number; False if there is no such transient (yet)."""
        if number not in self.numbers():
            return False
        self.follow = False
        self._show(number)
        return True

    def _typed(self) -> None:
        text = self._number.get().strip()
        if not (text.isdigit() and self.show_number(int(text))):
            self._entry.config(style="Error.TEntry")
            self._entry.after(900, self._end_error)
            self._entry.bell()

    def _end_error(self) -> None:
        """After a number that is not there: the field shows the transient on view again."""
        if self._entry.winfo_exists():
            self._entry.config(style="TEntry")
            if self._shown is not None:
                self._number.set(str(self._shown))

    def _show(self, number: int) -> None:
        self._shown = number
        head = f"Set {self.set_index}: transient {number:03d} of {self.expected}"
        if self._live is not None and self._live[0] == number:
            _, data, total = self._live
            percent = 100 * len(data) // total if total else 0
            self.row.show(data, f"{head}\nacquiring... {len(data)}/{total} samples ({percent}%)", live=True)
        elif number in self._data:
            data, file_name = self._data[number]
            self.row.show(data, f"{head}\n{file_name}")
        self._update_picker()

    def _update_picker(self) -> None:
        if self._shown is not None and self.root_focus() is not self._entry:
            self._number.set(str(self._shown))
        saved = len(self._data)
        self._of.config(text=f"of {self.expected}" + (f"  ({saved} saved)" if saved < self.expected else ""))
        self._latest.state(["pressed"] if self.follow else ["!pressed"])

    def root_focus(self):
        try:
            return self._entry.focus_get()
        except (KeyError, tk.TclError):
            return None


class ResultsView(ttk.Frame):
    """One run in a tab. Two rows per set, top to bottom: its single transients (one
    row, browsed by number) and its average; then set 2... Rows are added while the
    queue runs; the average row of a set is redrawn after every transient."""

    def __init__(self, parent: tk.Misc, folder: Path, on_3d=None) -> None:
        super().__init__(parent, padding=(0, 10, 0, 0))
        self.folder = folder
        self._on_3d = on_3d
        dpi = self.winfo_fpixels("1i")
        self._row_px = [int(ROW_SIZE_IN[0] * dpi), int(ROW_SIZE_IN[1] * dpi)]

        bottom = theme.card(self, padding=8)
        bottom.pack(side="bottom", fill="x", pady=(10, 0))
        theme.caption(bottom.body, "Folder").pack(side="left")
        ttk.Label(bottom.body, text=str(folder), style="Card.TLabel", foreground=theme.ACCENT,
                  font=theme.mono(9)).pack(side="left", padx=(6, 16))
        self._progress_text = ttk.Label(bottom.body, text="", style="Card.TLabel", foreground=theme.MUTED,
                                        font=theme.mono(9))
        self._progress_text.pack(side="right")

        holder = ttk.Frame(self)
        holder.pack(side="top", fill="both", expand=True)
        scroll = ttk.Scrollbar(holder, orient="vertical")
        scroll.pack(side="right", fill="y")
        self._view = tk.Canvas(holder, highlightthickness=0, yscrollcommand=scroll.set,
                               background=theme.BACKGROUND)
        self._view.pack(side="left", fill="both", expand=True)
        scroll.config(command=self._view.yview)
        self._inner = ttk.Frame(self._view)
        self._inner_id = self._view.create_window(0, 0, window=self._inner, anchor="nw")
        self._inner.bind("<Configure>", lambda _e: self._view.config(scrollregion=self._view.bbox("all")))
        self._view.bind("<Configure>", self._on_resize)
        self._view.bind("<MouseWheel>", self._on_wheel)
        self._inner.bind("<MouseWheel>", self._on_wheel)

        self._averages: dict[int, TraceRow] = {}
        self._browsers: dict[int, TransientBrowser] = {}
        self._complete: dict[int, tuple[np.ndarray, int]] = {}
        self._average_drawn: dict[int, float] = {}

    def restyle(self) -> None:
        for row in self._all_rows():
            row.restyle()

    def _all_rows(self) -> list[TraceRow]:
        return [*(b.row for b in self._browsers.values()), *self._averages.values()]

    def browser(self, set_index: int) -> TransientBrowser | None:
        return self._browsers.get(set_index)

    def show_live(self, group: ResultGroup, number: int, data: np.ndarray, total_samples: int) -> None:
        """Draw a transient that is still being acquired (data = samples received so far)
        and the set's average with it, so the average is shown from the very start."""
        follow = self._at_bottom()
        self._browser(group, total_samples).live(number, data, total_samples)
        percent = 100 * len(data) // total_samples if total_samples else 0

        complete, count = self._complete.get(group.set_index, (None, 0))
        now = time.monotonic()
        if now - self._average_drawn.get(group.set_index, -LIVE_AVERAGE_INTERVAL_S) < LIVE_AVERAGE_INTERVAL_S:
            self._follow(follow)
            return
        self._average_drawn[group.set_index] = now
        if complete is None:
            live_average = data
        else:
            k = min(len(data), len(complete))
            live_average = complete.copy()
            live_average[:k] = (complete[:k] * count + data[:k]) / (count + 1)
        self._average_row(group, total_samples).show(
            live_average, f"Set {group.set_index}: average - {count} complete transient(s) + transient "
                          f"{number:03d} in progress ({percent}%)")
        self._follow(follow)

    def add_transient(self, group: ResultGroup, transient: Transient, average_data: np.ndarray, count: int) -> None:
        """Add a complete transient to its set's transients row, then redraw the average
        (average_data = the mean of the set's first count transients)."""
        follow = self._at_bottom()
        self._browser(group, len(transient.data)).add(transient.number, transient.data, transient.file.name)
        self._complete[group.set_index] = (average_data, count)
        average = self._average_row(group, len(average_data))
        of = "" if count == group.expected_transients else f" (so far {count} of {group.expected_transients})"
        average.show(average_data, f"Set {group.set_index}: average of {count} transient(s){of}\n"
                                   f"{group.average_file.name}")
        self._follow(follow)

    def show_saved(self, result: QueueResult) -> None:
        """Show the results of an earlier run, read back from its folder: every set's
        transients (one open row, showing the last one) and its average."""
        n_transients = 0
        for group in result.groups:
            if group.transients:
                browser = self._browser(group, len(group.transients[0].data), expanded=False)
                for transient in group.transients:
                    browser.add(transient.number, transient.data, transient.file.name)
                browser.show_number(group.transients[-1].number)
                browser.row.expand(True)
            count = len(group.transients)
            n_transients += count
            missing = (f" (only {count} of {group.expected_transients} saved)"
                       if count < group.expected_transients else "")
            self._average_row(group, len(group.average)).show(
                group.average, f"Set {group.set_index}: average of {count} transient(s){missing}\n"
                               f"{group.average_file.name}")
        self.update_idletasks()
        self._view.yview_moveto(0.0)
        self.set_status(f"Loaded {len(result.groups)} set(s), {n_transients} transient(s)")

    def set_status(self, text: str) -> None:
        self._progress_text.config(text=text)

    def _browser(self, group: ResultGroup, total_samples: int, expanded: bool = True) -> TransientBrowser:
        """The transients row of a set, created (above the set's average, if there is one)
        when needed. A new open one closes those of the earlier sets."""
        browser = self._browsers.get(group.set_index)
        if browser is None:
            if expanded:
                for other in self._browsers.values():
                    other.row.expand(False)
            browser = self._browsers[group.set_index] = TransientBrowser(
                self._inner, group, total_samples, self._scroll, expanded=expanded,
                on_3d=self._on_3d and (lambda b: self._on_3d(self, b)))
            average = self._averages.get(group.set_index)
            self._pack(browser.row, before=average.frame if average else None)
        return browser

    def _average_row(self, group: ResultGroup, total_samples: int) -> TraceRow:
        """The average row of a set, created below its transients when needed."""
        average = self._averages.get(group.set_index)
        if average is None:
            average = self._averages[group.set_index] = TraceRow(
                self._inner, group.sample_rate_hz, group.channels, average=True,
                total_samples=total_samples, on_scroll=self._scroll, start_s=group.acquisition_start_s)
            self._pack(average)
        return average

    def _at_bottom(self) -> bool:
        return self._view.yview()[1] >= 0.999

    def _follow(self, was_at_bottom: bool) -> None:
        """Keep showing the newest rows if the view was at the bottom."""
        self.update_idletasks()
        if was_at_bottom:
            self._view.yview_moveto(1.0)

    def _on_resize(self, event: tk.Event) -> None:
        """Stretch every row to the width of the view (the figures follow their widgets)."""
        width = event.width - 2 * _ROW_BORDER_PX
        if width < 200 or width == self._row_px[0]:
            return
        self._row_px[0] = width
        self._view.itemconfigure(self._inner_id, width=event.width)
        for row in self._all_rows():
            row.set_size(*self._row_px)

    def _pack(self, row: TraceRow, before: tk.Widget | None = None) -> None:
        row.set_size(*self._row_px)
        if before is None:
            row.frame.pack(side="top", fill="x", anchor="w", pady=(0, 8))
        else:
            row.frame.pack(side="top", fill="x", anchor="w", pady=(0, 8), before=before)

    def _on_wheel(self, event: tk.Event) -> None:
        self._scroll(event.delta / 120)

    def _scroll(self, notches: float) -> None:
        self._view.yview_scroll(int(-notches * 3), "units")


class ResultsPanel(ttk.Frame):
    """Tabs of results: the live run(s) and runs loaded from their folders."""

    def __init__(self, parent: tk.Misc, on_load) -> None:
        super().__init__(parent, padding=(18, 16, 18, 16))
        head = theme.card(self, padding=12)
        head.pack(side="top", fill="x")
        theme.heading(head.body, "Results").pack(side="left")
        self._count = theme.Badge(head.body, "no runs open", theme.MUTED)
        self._count.pack(side="left", padx=10)
        ttk.Button(head.body, text="Load results…", command=on_load).pack(side="right")
        ttk.Label(head.body, text="Ctrl + wheel: zoom   Shift + wheel: pan   drag: spectrum range",
                  style="Card.TLabel", foreground=theme.MUTED, font=theme.mono(8)).pack(side="right", padx=16)

        self._tabs = theme.closable_notebook(self, self.close)
        self._empty = ttk.Frame(self)
        ttk.Label(self._empty, text="No results open.\n\nRun the queue to see every transient live, "
                                    "or load the folder of an earlier run.",
                  foreground=theme.MUTED, font=theme.sans(11), justify="center").pack(pady=(80, 16))
        ttk.Button(self._empty, text="Load results…", style="Accent.TButton", command=on_load).pack()
        self._views: list[ResultsView] = []
        self._plots: list[ThreeDView] = []
        self._update()

    @property
    def views(self) -> list[ResultsView]:
        return list(self._views)

    def new_view(self, folder: Path, title: str) -> ResultsView:
        """A new tab, shown at once."""
        view = ResultsView(self._tabs, folder, on_3d=self.open_3d)
        self._views.append(view)
        self._tabs.add(view, text=title)
        self._tabs.select(view)
        self._update()
        return view

    def restyle(self) -> None:
        for view in (*self._views, *self._plots):
            view.restyle()

    def open_3d(self, view: ResultsView, browser: TransientBrowser) -> ThreeDView:
        """A new tab with the 3D plot of one set's transients."""
        group = browser.group
        title = f"3D · set {group.set_index} · {view.folder.name}"
        plot = ThreeDView(self._tabs, browser.complete, group.sample_rate_hz, group.channels,
                          group.acquisition_start_s, title)
        self._plots.append(plot)
        self._tabs.add(plot, text=title)
        self._tabs.select(plot)
        self._update()
        return plot

    def rename(self, view: ResultsView, title: str) -> None:
        if view in self._views:
            self._tabs.tab(view, text=title)

    def close(self, tab: tk.Misc) -> None:
        """Close a tab (a run or a 3D plot)."""
        for tabs in (self._views, self._plots):
            if tab in tabs:
                tabs.remove(tab)
                self._tabs.forget(tab)
                tab.destroy()
                self._update()
                return

    def _update(self) -> None:
        n = len(self._views)
        self._count.set(f"{n} run(s) open" if n else "no runs open", theme.ACCENT if n else theme.MUTED)
        if n or self._plots:
            self._empty.pack_forget()
            self._tabs.pack(side="top", fill="both", expand=True, pady=(12, 0))
        else:
            self._tabs.pack_forget()
            self._empty.pack(side="top", fill="both", expand=True)
