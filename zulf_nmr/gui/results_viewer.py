"""Live results window: every transient and the running average of each set, with spectra."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector

from zulf_nmr.experiment.schedule import QueueResult, ResultGroup, Transient
from zulf_nmr.utils.visualization.results_plotter import draw_spectrum, draw_time

ROW_SIZE_IN = (11.0, 3.6)  # starting size; rows are stretched to the width of the window
SELECTION_COLOUR = "#999999"
VIEW_S = 0.05  # time range shown at first, short enough to see the single samples
ZOOM_STEP = 1.25  # per wheel notch
PAN_STEP = 0.2  # fraction of the shown range per wheel notch


class TraceRow:
    """One trace: all channels in time (left) and the spectrum of the selected range (right).

    The time plot shows VIEW_S at first; Ctrl + wheel zooms it, Shift + wheel moves it.
    Drag on the time plot to choose the range of the spectrum; drag the selection to move it;
    a click (left, without dragging, or right) removes it - the spectrum is then of the whole trace.
    A row of a transient that is still being received follows its newest samples
    until the user zooms or moves it.
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
        self.t_range: tuple[float, float] | None = None
        self.view: tuple[float, float] | None = None  # (start, width) in s; None = not set by the user
        self.on_scroll = on_scroll  # wheel without a modifier scrolls the list of rows

        self.figure = Figure(figsize=ROW_SIZE_IN, layout="constrained")
        self.ax_time, self.ax_spec = self.figure.subplots(1, 2, width_ratios=[3, 2])
        self.canvas = FigureCanvasTkAgg(self.figure, master=master)
        self.widget = self.canvas.get_tk_widget()
        self.selector: SpanSelector | None = None
        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.canvas.mpl_connect("button_press_event", self._on_press)

    def show(self, data: np.ndarray, title: str, live: bool = False) -> None:
        """Draw new data (a new trace, or an updated average); the selected range and the
        zoom are kept. live: the trace is still growing - show its newest samples."""
        self.data, self.title = data, title
        draw_time(self.ax_time, data, self.rate, self.channels, title, self.average, self._xlim(live),
                  self.start_s)
        self._draw_spectrum()
        # ax.clear() removes the selector's artists, so a new one is made for every redraw.
        self.selector = SpanSelector(
            self.ax_time, self._on_select, "horizontal", useblit=False, interactive=True,
            drag_from_anywhere=True, button=1, props=dict(facecolor=SELECTION_COLOUR, alpha=0.25),
        )
        if self.t_range is not None:
            self.selector.extents = self.t_range
        self.canvas.draw_idle()

    def _on_select(self, t0: float, t1: float) -> None:
        if t1 - t0 <= 0:  # a click without dragging
            self.clear_selection()
            return
        self.t_range = (max(t0, self.start_s), t1)
        self._draw_spectrum()
        self.canvas.draw_idle()

    def clear_selection(self) -> None:
        """Remove the selected range; the spectrum is of the whole trace again."""
        self.t_range = None
        if self.selector is not None:
            self.selector.clear()
        self._draw_spectrum()
        self.canvas.draw_idle()

    def _on_press(self, event) -> None:
        if event.button == 3 and event.inaxes is self.ax_time:
            self.clear_selection()

    def _draw_spectrum(self) -> None:
        draw_spectrum(self.ax_spec, self.data, self.rate, self.channels, self.t_range, self.average,
                      self.start_s)

    def _xlim(self, live: bool) -> tuple[float, float]:
        if self.view is not None:
            start, width = self.view
        else:
            # The first VIEW_S of the recording (not the empty time before it), or while
            # the trace is still growing, its newest VIEW_S.
            width = min(VIEW_S, (self.total_s - self.start_s) or VIEW_S)
            newest = self.start_s + len(self.data) / self.rate
            start = max(self.start_s, newest - width) if live else self.start_s
        return start, start + width

    def _on_scroll(self, event) -> None:
        state = getattr(event.guiEvent, "state", 0)
        ctrl, shift = bool(state & 0x4), bool(state & 0x1)
        if not (ctrl or shift) or event.inaxes is not self.ax_time:
            if self.on_scroll:
                self.on_scroll(event.step)
            return
        start, end = self.ax_time.get_xlim()
        width = end - start
        if ctrl:  # zoom around the cursor
            new_width = width / ZOOM_STEP if event.step > 0 else width * ZOOM_STEP
            new_width = min(max(new_width, 5 / self.rate),
                            max(self.total_s, self.start_s + len(self.data) / self.rate, VIEW_S))
            x = event.xdata if event.xdata is not None else start + width / 2
            start = x - (x - start) * new_width / width
            width = new_width
        else:  # move along the time axis
            start -= event.step * PAN_STEP * width
        # Stay within 0 (sequence start) .. the end of the recording.
        end_s = max(self.total_s, self.start_s + len(self.data) / self.rate)
        width = min(width, end_s)
        start = max(0.0, min(start, end_s - width))
        self.view = (start, width)
        self.ax_time.set_xlim(start, start + width)
        self.canvas.draw_idle()


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
                row = self._row(group, transient.number, len(transient.data))
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

    def _row(self, group: ResultGroup, number: int, total_samples: int) -> TraceRow:
        """The row of a transient, created (above the set's average, if there is one) when needed."""
        key = (group.set_index, number)
        row = self._rows.get(key)
        if row is None:
            row = self._rows[key] = TraceRow(self._inner, group.sample_rate_hz, group.channels,
                                             average=False, total_samples=total_samples,
                                             on_scroll=self._scroll, start_s=group.acquisition_start_s)
            average = self._averages.get(group.set_index)
            self._pack(row, before=average.widget if average else None)
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
        # Wheel events over a row arrive through matplotlib (TraceRow._on_scroll).
        row.widget.config(width=self._row_px[0], height=self._row_px[1])
        if before is None:
            row.widget.pack(side="top", anchor="w")
        else:
            row.widget.pack(side="top", anchor="w", before=before)

    def _on_wheel(self, event: tk.Event) -> None:
        self._scroll(event.delta / 120)

    def _scroll(self, notches: float) -> None:
        self._view.yview_scroll(int(-notches * 3), "units")
