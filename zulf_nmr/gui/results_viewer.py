"""Live results window: every transient and the running average of each set, with spectra."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import ttk

import numpy as np
from matplotlib.backend_tools import Cursors
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector

from zulf_nmr.experiment.schedule import QueueResult, ResultGroup, Transient
from zulf_nmr.gui.tooltip import Tooltip
from zulf_nmr.utils.visualization.results_plotter import (
    draw_spectrum,
    draw_time,
    smooth_for_display,
    visible_points,
)

ROW_SIZE_IN = (11.0, 3.6)  # starting size; rows are stretched to the width of the window
SELECTION_COLOUR = "#999999"
CURSOR_COLOUR = "#C0392B"
EDGE_GRAB_PX = 10  # how close to an edge of the selection (or a cursor) the mouse must be to grab it
ZOOM_STEP = 1.25  # per wheel notch
PAN_STEP = 0.2  # fraction of the shown range per wheel notch
# Fixed margins around the plots in pixels: the plots keep their place and size while the
# data, the tick labels and the titles change (an automatic layout would move them).
_MARGIN_PX = {"left": 64, "right": 14, "top": 40, "bottom": 40, "gap": 74}
_TIME_SHARE = 0.6  # of the width of both plots


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
        background = ttk.Style().lookup("TFrame", "background")  # the window's, not tk's white
        for widget in (self, *self.winfo_children()):
            try:
                widget.config(background=background)
            except tk.TclError:
                pass

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
                 total_samples: int | None = None, on_scroll=None, start_s: float = 0.0) -> None:
        self.rate = sample_rate_hz
        self.channels = channels
        self.average = average
        self.start_s = start_s  # acquisition start, seconds after the sequence start
        self.total_s = start_s + (total_samples or 0) / sample_rate_hz  # end of the time axis
        self.data = np.empty((0, len(channels)))
        self.title = ""
        self.live = False
        self.t_range: tuple[float, float] | None = None
        # Limits set by the user (toolbar, wheel) per plot: (xlim, ylim); a plot without
        # them follows its data.
        self._user_limits: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {}
        self._auto_ylim: tuple[float, float] | None = None  # only grows, so a growing trace does not jump
        self._drawing = False  # limits are being set by the program, not by the user
        self._spectrum = (np.zeros(1), np.zeros((1, len(channels))))
        self._smooth = (np.zeros(0), np.zeros((0, len(channels))))  # smoothed trace, all of it
        self._channel_lines: list = []  # the drawn curves, one per channel
        self.on_scroll = on_scroll  # wheel without a modifier scrolls the list of rows

        # frame = header (toggle button + name, toolbar) above the plots; the frame is what is packed.
        self.frame = ttk.Frame(master)
        header = ttk.Frame(self.frame)
        header.pack(side="top", fill="x", pady=(4, 0))
        self._toggle = ttk.Button(header, text="−", width=3, command=self.toggle)
        self._toggle.pack(side="left")
        self._name = ttk.Label(header, text="", font=("Segoe UI", 9, "bold" if average else "normal"),
                               cursor="hand2")
        self._name.pack(side="left", padx=6)
        self._name.bind("<Button-1>", lambda _e: self.toggle())
        self.expanded = True

        self.figure = Figure(figsize=ROW_SIZE_IN)
        self.ax_time = self.figure.add_axes((0, 0, 1, 1))
        self.ax_spec = self.figure.add_axes((0, 0, 1, 1))
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.frame)
        self.widget = self.canvas.get_tk_widget()
        self.toolbar = RowToolbar(self.canvas, header, self)
        self.toolbar.pack(side="left", fill="x", expand=True, padx=(12, 0))
        self._readout = ttk.Label(self.frame, text="", font=("Consolas", 9), foreground=CURSOR_COLOUR)
        self.widget.pack(side="top", anchor="w")
        for w in (header, self._toggle, self._name, self.toolbar, self._readout):
            w.bind("<MouseWheel>", lambda e: self.on_scroll and self.on_scroll(e.delta / 120))
        self._layout()

        self.selector: SpanSelector | None = None
        self.canvas.mpl_connect("resize_event", lambda _e: self._layout())
        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        self._cursor = Cursors.POINTER
        self._drag_cursor: Cursors | None = None  # cursor kept while the selection is dragged

        # Measuring cursors A and B on each plot (s on the time plot, Hz on the spectrum).
        self.cursors_on = False
        self._cursor_x: dict[str, list[float | None]] = {"time": [None, None], "spec": [None, None]}
        self._cursor_plot = "time"  # the plot whose cursors the readout describes
        self._dragged_cursor: tuple[str, int] | None = None
        self._background = None  # picture without the cursors while one is dragged

    def _layout(self) -> None:
        """Place both plots with fixed margins (in pixels) for the current figure size."""
        width, height = self.figure.bbox.width, self.figure.bbox.height
        m = _MARGIN_PX
        plots = max(width - m["left"] - m["right"] - m["gap"], 50)
        time_w, spec_w = plots * _TIME_SHARE, plots * (1 - _TIME_SHARE)
        bottom, plot_h = m["bottom"] / height, max(height - m["top"] - m["bottom"], 20) / height
        self.ax_time.set_position((m["left"] / width, bottom, time_w / width, plot_h))
        self.ax_spec.set_position(((m["left"] + time_w + m["gap"]) / width, bottom, spec_w / width, plot_h))
        self._thin_lines()  # as many points as the plot is now wide

    # ------------------------------------------------------------------ drawing

    def show(self, data: np.ndarray, title: str, live: bool = False) -> None:
        """Draw new data (a new trace, or an updated average); the selected range, the
        zoom and the cursors are kept. live: the trace is still growing."""
        self.data, self.title, self.live = data, title, live
        self._background = None  # new data: a cursor being dragged is drawn the slow way
        self._name.config(text=title.replace("\n", "  —  "))
        # Smoothed once per data; only the part in view, thinned to the plot's width, is drawn.
        t, values = smooth_for_display(data.reshape(len(data), -1), self.rate)
        self._smooth = (t + self.start_s, values)
        points = visible_points(*self._smooth, self._time_xlim(), self.ax_time.bbox.width)
        self._drawing = True
        try:
            draw_time(self.ax_time, data, self.rate, self.channels, title, self.average, None, self.start_s,
                      points)
            self._channel_lines = self.ax_time.lines[:len(self.channels)]
            self._set_time_limits()
            self._draw_cursors("time")
        finally:
            self._drawing = False
        self._watch_limits(self.ax_time, "time")
        self.ax_time.callbacks.connect("xlim_changed", lambda _ax: self._thin_lines())
        self._draw_spectrum()
        # ax.clear() removes the selector's artists, so a new one is made for every redraw.
        self._new_selector()
        self._update_readout()
        self.canvas.draw_idle()

    def _time_xlim(self) -> tuple[float, float]:
        """The user's time range, or from the sequence start (0) to the newest sample:
        the axis widens as the samples of a transient arrive."""
        if "time" in self._user_limits:
            return self._user_limits["time"][0]
        end = self.start_s + len(self.data) / self.rate
        if end <= self.start_s:
            end = max(self.total_s, self.start_s + 1 / self.rate)
        return 0.0, end

    def _thin_lines(self) -> None:
        """Give the channel lines the points of the time range now in view (after a zoom,
        a pan or a resize): drawing then costs the same at any zoom."""
        if not self._channel_lines:
            return
        t, values = visible_points(*self._smooth, self.ax_time.get_xlim(), self.ax_time.bbox.width)
        for column, line in enumerate(self._channel_lines):
            line.set_data(t, values[:, column])

    def _set_time_limits(self) -> None:
        if "time" in self._user_limits:
            xlim, ylim = self._user_limits["time"]
            self.ax_time.set_xlim(*xlim)
            self.ax_time.set_ylim(*ylim)
            return
        self.ax_time.set_xlim(*self._time_xlim())
        low, high = self.ax_time.get_ylim()  # autoscaled to the data
        if self._auto_ylim is not None:
            low, high = min(low, self._auto_ylim[0]), max(high, self._auto_ylim[1])
        self._auto_ylim = (low, high)
        self.ax_time.set_ylim(low, high)

    def _draw_spectrum(self) -> None:
        self._drawing = True
        try:
            self._spectrum = draw_spectrum(self.ax_spec, self.data, self.rate, self.channels, self.t_range,
                                           self.average, self.start_s)
            if "spec" in self._user_limits:
                xlim, ylim = self._user_limits["spec"]
                self.ax_spec.set_xlim(*xlim)
                self.ax_spec.set_ylim(*ylim)
            self._draw_cursors("spec")
        finally:
            self._drawing = False
        self._watch_limits(self.ax_spec, "spec")

    def _watch_limits(self, ax, plot: str) -> None:
        """Remember limits the user sets (toolbar, wheel); ax.clear() drops the callbacks."""
        def changed(_ax) -> None:
            if not self._drawing:
                self._user_limits[plot] = (ax.get_xlim(), ax.get_ylim())
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
            props=dict(facecolor=SELECTION_COLOUR, alpha=0.25),
        )
        if self.t_range is not None:
            self.selector.extents = self.t_range

    def expand(self, expanded: bool = True) -> None:
        """Show (or, with expanded=False, hide) the plots and the toolbar; the header stays."""
        if expanded == self.expanded:
            return
        self.expanded = expanded
        if expanded:
            self.toolbar.pack(side="left", fill="x", expand=True, padx=(12, 0))
            self.widget.pack(side="top", anchor="w")
            if self.cursors_on:
                self._readout.pack(side="top", anchor="w", padx=6, before=self.widget)
        else:
            self.toolbar.pack_forget()
            self.widget.pack_forget()
            self._readout.pack_forget()
        self._toggle.config(text="−" if expanded else "+")

    def toggle(self) -> None:
        self.expand(not self.expanded)

    # ----------------------------------------------------------- spectrum range

    def _on_select(self, t0: float, t1: float) -> None:
        if t1 - t0 <= 0:  # a click without dragging
            self.clear_selection()
            return
        self.t_range = (max(t0, self.start_s), t1)
        self._draw_spectrum()
        self._update_readout()
        self.canvas.draw_idle()

    def clear_selection(self) -> None:
        """Remove the selected range; the spectrum is of the whole trace again."""
        self.t_range = None
        if self.selector is not None:
            self.selector.clear()
        self._draw_spectrum()
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
            # Double click on an edge: type its position. After matplotlib has finished
            # with the click, so the SpanSelector does not take the dialog's events.
            left, right = (self.ax_time.transData.transform((x, 0))[0] for x in self.selector.extents)
            edge = 0 if abs(event.x - left) <= abs(event.x - right) else 1
            self.widget.after_idle(self._edit_edge, edge)
        elif event.button == 1:
            # Keep the hover cursor while dragging; the SpanSelector sets its own on press,
            # so it is set again once matplotlib has handled the event.
            self._drag_cursor = self._hover_cursor(event)
            self.widget.after_idle(self._set_cursor, self._drag_cursor, True)

    def _on_release(self, event) -> None:
        if self._tool_active():
            return
        self._end_cursor_drag()
        self._drag_cursor = None
        # The SpanSelector resets the cursor on release; show the one for where the mouse is.
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

    # -------------------------------------------------------- measuring cursors

    def toggle_cursors(self) -> None:
        self.set_cursors(not self.cursors_on)

    def set_cursors(self, on: bool) -> None:
        """Turn the measuring cursors on or off. While they are on, a click on a plot puts
        cursor A, the next one B; a click near a cursor, or a drag, moves it."""
        button = self.toolbar.cursors_button
        if on != self.cursors_on:
            if on:
                if self.toolbar.mode:  # Pan or Zoom off first: one tool at a time
                    (self.toolbar.pan if self.toolbar.mode.name == "PAN" else self.toolbar.zoom)()
                self.canvas.widgetlock(self)  # the SpanSelector waits meanwhile
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
        # While dragging, only the cursors are redrawn over a saved picture of the rest
        # (blitting): a full redraw of a long trace is too slow to follow the mouse.
        self._draw_cursors(plot, animated=True)
        self.canvas.draw()
        self._background = self.canvas.copy_from_bbox(self.figure.bbox)
        self._blit_cursors(plot)
        self._update_readout()

    def _end_cursor_drag(self) -> None:
        if self._dragged_cursor is not None:
            self._draw_cursors(self._dragged_cursor[0])  # part of the picture again
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
            line = ax.axvline(x, color=CURSOR_COLOUR, linewidth=1.0, linestyle="-" if name == "A" else "--",
                              animated=animated)
            label = ax.annotate(name, (x, 1.0), xycoords=("data", "axes fraction"), xytext=(3, -3),
                                textcoords="offset points", va="top", fontsize=8, color=CURSOR_COLOUR,
                                fontweight="bold", annotation_clip=True, animated=animated)
            line.set_gid("cursor")
            label.set_gid("cursor")

    def _values_at(self, plot: str, x: float) -> list[float]:
        """Value of every channel at x: the signal at time x, or the amplitude at frequency x."""
        if plot == "time":
            if len(self.data) == 0:
                return [float("nan")] * len(self.channels)
            xs = self.start_s + np.arange(len(self.data)) / self.rate
            columns = self.data.reshape(len(self.data), -1)
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

    # -------------------------------------------------------------------- wheel

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
            # 0 (sequence start) .. the end of the recording; at least 5 samples shown.
            limit, narrowest = max(self.total_s, self.start_s + len(self.data) / self.rate), 5 / self.rate
        else:
            # 0 .. the Nyquist frequency; at least 5 frequency steps shown.
            freqs = self._spectrum[0]
            limit = self.rate / 2
            narrowest = 5 * (freqs[1] - freqs[0]) if len(freqs) > 1 else limit / 1000
        if ctrl:  # zoom around the mouse
            new_width = width / ZOOM_STEP if event.step > 0 else width * ZOOM_STEP
            new_width = min(max(new_width, narrowest), limit)
            x = event.xdata if event.xdata is not None else start + width / 2
            start = x - (x - start) * new_width / width
            width = new_width
        else:  # move along the axis
            start -= event.step * PAN_STEP * width
        width = min(width, limit)
        start = max(0.0, min(start, limit - width))
        ax.set_xlim(start, start + width)  # remembered as the user's view
        self.canvas.draw_idle()


class EdgeDialog(tk.Toplevel):
    """Entry for the exact position of one edge of the spectrum range, opened at the mouse."""

    def __init__(self, master: tk.Misc, name: str, extents: tuple[float, float], edge: int,
                 limits: tuple[float, float], on_apply, on_close=None) -> None:
        super().__init__(master)
        self._on_close = on_close  # called once the dialog is gone (OK, Cancel, Esc or closed)
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
        ttk.Button(frame, text="OK", command=self._ok).grid(row=1, column=2, sticky="e")
        ttk.Button(frame, text="Cancel", command=self.destroy).grid(row=1, column=1, sticky="e", padx=6)
        self._error = ttk.Label(frame, text=f"Allowed: {limits[0]:.6g} - {limits[1]:.6g} s",
                                foreground="#666666")
        self._error.grid(row=2, column=0, columnspan=3, sticky="w")

        entry.bind("<Return>", lambda _e: self._ok())
        self.bind("<Escape>", lambda _e: self.destroy())
        x, y = self.winfo_pointerxy()
        self.geometry(f"+{x + 12}+{y + 12}")
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
        self._error.config(text=message, foreground="#d0342c")
        self.bell()


class ResultsViewer(tk.Toplevel):
    """Rows top to bottom: set 1 transients, set 1 average, set 2... Rows are added
    while the queue runs; the average row of a set is redrawn after every transient."""

    def __init__(self, parent: tk.Misc, folder: Path) -> None:
        super().__init__(parent)
        self.title(f"Results — {folder.name}")
        dpi = self.winfo_fpixels("1i")
        self._row_px = [int(ROW_SIZE_IN[0] * dpi), int(ROW_SIZE_IN[1] * dpi)]
        # Nearly the whole screen; the rows take the full width of the window.
        self.geometry(f"{self.winfo_screenwidth() - 80}x{self.winfo_screenheight() - 120}+40+20")

        # Bottom edge: progress of the whole queue, the same as in the main window.
        progress = ttk.Frame(self, padding=(8, 0, 8, 8))
        progress.pack(side="bottom", fill="x")
        self._progress_text = ttk.Label(progress, text="", width=48)
        self._progress_text.pack(side="left")
        self._progress = ttk.Progressbar(progress, mode="determinate")
        self._progress.pack(side="left", fill="x", expand=True, padx=(8, 0))

        bottom = ttk.Frame(self, padding=8)
        bottom.pack(side="bottom", fill="x")
        ttk.Label(bottom, text=str(folder)).pack(side="left")  # what happens is shown by the progress bar
        ttk.Button(bottom, text="Close", command=self.destroy).pack(side="right")

        holder = ttk.Frame(self)
        holder.pack(side="top", fill="both", expand=True)
        scroll = ttk.Scrollbar(holder, orient="vertical")
        scroll.pack(side="right", fill="y")
        self._view = tk.Canvas(holder, highlightthickness=0, yscrollcommand=scroll.set)
        self._view.pack(side="left", fill="both", expand=True)
        scroll.config(command=self._view.yview)
        self._inner = ttk.Frame(self._view)
        self._inner_id = self._view.create_window(0, 0, window=self._inner, anchor="nw")
        self._inner.bind("<Configure>", lambda _e: self._view.config(scrollregion=self._view.bbox("all")))
        self._view.bind("<Configure>", self._on_resize)
        self._view.bind("<MouseWheel>", self._on_wheel)

        self._averages: dict[int, TraceRow] = {}  # by set index
        self._rows: dict[tuple[int, int], TraceRow] = {}  # by (set index, transient number)
        # Mean of the complete transients of every set and how many there are.
        self._complete: dict[int, tuple[np.ndarray, int]] = {}

    # ------------------------------------------------------------ live update

    def show_live(self, group: ResultGroup, number: int, data: np.ndarray, total_samples: int) -> None:
        """Draw a transient that is still being acquired (data = samples received so far)
        and the set's average with it, so the average is shown from the very start."""
        follow = self._at_bottom()
        row = self._row(group, number, total_samples)
        percent = 100 * len(data) // total_samples if total_samples else 0
        row.show(data, f"Set {group.set_index}: transient {number:03d} of {group.expected_transients}"
                       f"\nacquiring... {len(data)}/{total_samples} samples ({percent}%)", live=True)

        complete, count = self._complete.get(group.set_index, (None, 0))
        if complete is None:
            live_average = data  # nothing to average with yet
        else:
            # Received samples: mean including the running transient; the rest: the complete ones.
            k = min(len(data), len(complete))
            live_average = complete.copy()
            live_average[:k] = (complete[:k] * count + data[:k]) / (count + 1)
        self._average_row(group, total_samples).show(
            live_average, f"Set {group.set_index}: average - {count} complete transient(s) + transient "
                          f"{number:03d} in progress ({percent}%)")
        self._follow(follow)

    def add_transient(self, group: ResultGroup, transient: Transient, average_data: np.ndarray, count: int) -> None:
        """Show a complete transient (in its live row, or a new one above its set's average),
        then redraw the average (average_data = the mean of the set's first count transients)."""
        follow = self._at_bottom()
        row = self._row(group, transient.number, len(transient.data))
        row.show(transient.data, f"Set {group.set_index}: transient {transient.number:03d} of "
                                 f"{group.expected_transients}\n{transient.file.name}")
        self._complete[group.set_index] = (average_data, count)
        average = self._average_row(group, len(average_data))
        of = "" if count == group.expected_transients else f" (so far {count} of {group.expected_transients})"
        average.show(average_data, f"Set {group.set_index}: average of {count} transient(s){of}\n"
                                   f"{group.average_file.name}")
        self._follow(follow)

    def show_saved(self, result: QueueResult) -> None:
        """Show the results of an earlier run, read back from its folder: every set's
        transients, then its average."""
        n_transients = 0
        for group in result.groups:
            for transient in group.transients:
                # Only the averages are open at first; a transient opens with its + button.
                row = self._row(group, transient.number, len(transient.data), expanded=False)
                row.show(transient.data, f"Set {group.set_index}: transient {transient.number:03d} of "
                                         f"{group.expected_transients}\n{transient.file.name}")
            count = len(group.transients)
            n_transients += count
            missing = (f" (only {count} of {group.expected_transients} saved)"
                       if count < group.expected_transients else "")
            self._average_row(group, len(group.average)).show(
                group.average, f"Set {group.set_index}: average of {count} transient(s){missing}\n"
                               f"{group.average_file.name}")
        self.update_idletasks()
        self._view.yview_moveto(0.0)
        self.set_progress(1, 1, f"Loaded {len(result.groups)} set(s), {n_transients} transient(s)")

    def set_progress(self, value: float, maximum: float, text: str) -> None:
        self._progress.config(maximum=maximum, value=value)
        self._progress_text.config(text=text)

    # ---------------------------------------------------------------- helpers

    def _row(self, group: ResultGroup, number: int, total_samples: int, expanded: bool = True) -> TraceRow:
        """The row of a transient, created (above the set's average, if there is one) when needed.

        A new open row closes the rows of the earlier transients (only their names stay),
        so while the queue runs just the transient being drawn and the averages are open.
        """
        key = (group.set_index, number)
        row = self._rows.get(key)
        if row is None:
            if expanded:
                for other in self._rows.values():
                    other.expand(False)
            row = self._rows[key] = TraceRow(self._inner, group.sample_rate_hz, group.channels,
                                             average=False, total_samples=total_samples,
                                             on_scroll=self._scroll, start_s=group.acquisition_start_s)
            row.expand(expanded)
            average = self._averages.get(group.set_index)
            self._pack(row, before=average.frame if average else None)
        return row

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
        """Stretch every row to the width of the window (the figures follow their widgets)."""
        if event.width < 200 or event.width == self._row_px[0]:
            return
        self._row_px[0] = event.width
        self._view.itemconfigure(self._inner_id, width=event.width)
        for row in (*self._rows.values(), *self._averages.values()):
            row.widget.config(width=event.width)

    def _pack(self, row: TraceRow, before: tk.Widget | None = None) -> None:
        # Wheel events over the plots arrive through matplotlib (TraceRow._on_scroll).
        row.widget.config(width=self._row_px[0], height=self._row_px[1])
        if before is None:
            row.frame.pack(side="top", fill="x", anchor="w")
        else:
            row.frame.pack(side="top", fill="x", anchor="w", before=before)

    def _on_wheel(self, event: tk.Event) -> None:
        self._scroll(event.delta / 120)

    def _scroll(self, notches: float) -> None:
        self._view.yview_scroll(int(-notches * 3), "units")
