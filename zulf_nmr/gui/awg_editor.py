"""AWG editor panel: an AWG sequence (.npz) written as formulas, with a calculator keypad."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

import numpy as np

from zulf_nmr.core.errors import english_error
from zulf_nmr.hardware.awg import N_CHANNELS, RANGE_FULL_SCALE_A, AwgRange, AwgWaveform
from zulf_nmr.hardware.awg_fit import fit_channel
from zulf_nmr.hardware.awg_formula import (
    ChannelFormula,
    Segment,
    SourceData,
    build_waveform,
    open_for_editing,
    save_waveform,
)
from zulf_nmr.gui import theme, user_settings
from zulf_nmr.gui.axis_drag import AxisDrag
from zulf_nmr.gui.formula_hint import GhostText
from zulf_nmr.gui.tooltip import Tooltip
from zulf_nmr.utils.visualization import plot_style
from zulf_nmr.utils.visualization.awg_plotter import plot_awg

PREVIEW_DELAY_MS = 150
_RANGES = {"HIGH (1 A)": AwgRange.HIGH, "LOW (10 mA)": AwgRange.LOW}
_SETTINGS = "awg_editor"
CHANNELS_HEIGHT_PX = 300

_KEYPAD = [
    ("Numbers", [["7", "8", "9", "/"],
                 ["4", "5", "6", "*"],
                 ["1", "2", "3", "-"],
                 ["0", ".", ",", "+"],
                 ["(", ")", "<", "C"]]),
    ("Functions", [["sin", "cos", "tan"],
                   ["exp", "log", "abs"],
                   ["^", "sqrt", "root"],
                   ["min", "max", ""]]),
    ("Shapes", [["step", "pulse"],
                ["ramp", "gauss"],
                ["hann", "chirp"],
                ["data", ""]]),
    ("Time and units", [["t", "tau"],
                        ["s", "ms"],
                        ["us", "Hz"],
                        ["kHz", "pi"],
                        ["e", ""]]),
]
_KEY_LABELS = {"<": "⌫"}
_FUNCTIONS = {"sin", "cos", "tan", "exp", "log", "sqrt", "root", "abs", "min", "max",
              "step", "pulse", "ramp", "gauss", "hann", "chirp", "data"}
_KEY_NAMES = {
    "/": "divide", "*": "multiply", "-": "subtract (or minus)", "+": "add", "^": "power: 2^3 = 8",
    ".": "decimal point", ",": "separates the arguments of a function", "(": "opening bracket",
    ")": "closing bracket",
    "sin": "sin(x) - sine (x in radians)", "cos": "cos(x) - cosine (x in radians)",
    "tan": "tan(x) - tangent (x in radians)", "exp": "exp(x) - exponential function, e to the power x",
    "log": "log(x) - natural logarithm", "sqrt": "sqrt(x) - square root",
    "root": "root(x, n) - n-th root of x; an odd n also takes x < 0: root(-8, 3) = -2",
    "abs": "abs(x) - absolute value",
    "min": "min(a, b) - the smaller of a and b, at every moment",
    "max": "max(a, b) - the larger of a and b, at every moment",
    "pi": "pi - 3.14159...", "e": "e - Euler's number, 2.71828...",
    "t": "t - time in seconds after :AWG:RUN",
    "tau": "tau - time in seconds from the start of the segment",
    "ms": "ms - millisecond, 0.001 s", "us": "us - microsecond, 0.000001 s", "s": "s - second",
    "kHz": "kHz - kilohertz, 1000 Hz", "Hz": "Hz - hertz",
    "step": "step(t0) - 0 before t0, 1 from t0 on (t0 from the start of the segment)",
    "pulse": "pulse(t0, t1) - 1 from t0 to t1, 0 elsewhere (times from the start of the segment)",
    "ramp": "ramp(t0, t1) - rises in a straight line from 0 at t0 to 1 at t1, then stays 1 "
            "(times from the start of the segment)",
    "gauss": "gauss(t0, sigma) - Gaussian bell centred at t0, height 1 (t0 from the start of the segment)",
    "hann": "hann(t0, t1) - smooth window: 0 at t0, 1 in the middle, 0 at t1 (times from the start of the segment)",
    "chirp": "chirp(f0, f1, t0, t1) - sine sweeping from f0 to f1 between t0 and t1, 0 outside "
             "(times from the start of the segment)",
    "data": "data(n) - channel n of the file opened with Open...",
    "<": "delete the character before the cursor", "C": "clear the whole formula",
}

_EXAMPLES = ("200*pulse(5ms, 15ms)", "5*sin(2*pi*300Hz*t)*exp(-t/15ms)",
             "500*ramp(40ms, 45ms) - 500*ramp(65ms, 70ms)", "300*chirp(100Hz, 800Hz, 10ms, 60ms)")


class _SegmentVars:
    """What is typed in one segment row."""

    def __init__(self, formula: str = "", start: str = "", stop: str = "") -> None:
        self.formula = tk.StringVar(value=formula)
        self.start = tk.StringVar(value=start)
        self.stop = tk.StringVar(value=stop)

    def segment(self) -> Segment:
        return Segment(self.formula.get(), self.start.get(), self.stop.get())


class AwgEditorPanel(ttk.Frame):
    """Formulas for the 4 AWG channels, a live preview and Save as .npz.

    Every channel is made of one or more segments (piecewise): a formula with its start
    and stop time in ms; an empty start is 0, an empty stop the end of the sequence.
    "+ segment" adds a row. The file is written only when the user presses Save and
    every channel is valid; it is then read back with the same checks as an experiment
    uses (load_awg_npz).
    """

    def __init__(self, parent: tk.Misc, initial_dir: Path, max_sample_rate_hz: float,
                 max_amplitude_A: float, on_saved=None) -> None:
        super().__init__(parent, padding=(18, 16, 18, 16))
        self._initial_dir = initial_dir
        self._max_rate = max_sample_rate_hz
        self._max_amp = max_amplitude_A
        self._on_saved = on_saved
        self._pending: str | None = None
        self._waveform: AwgWaveform | None = None
        self._pending_preview: str | None = None
        self._drawn: AwgWaveform | None = None
        self._errors: list[str] = []
        self._source: SourceData | None = None

        saved = user_settings.load_section(_SETTINGS)
        segments = _saved_segments(saved)
        ranges = saved.get("ranges")
        if not (isinstance(ranges, list) and len(ranges) == N_CHANNELS and all(r in _RANGES for r in ranges)):
            ranges = ["HIGH (1 A)"] * N_CHANNELS
        self._rate = tk.StringVar(value=user_settings.text(saved, "sample_rate_hz", "10000"))
        self._duration = tk.StringVar(value=user_settings.text(saved, "duration_ms", "50"))
        self._segments: list[list[_SegmentVars]] = [[] for _ in range(N_CHANNELS)]
        self._ranges = [tk.StringVar(value=r) for r in ranges]
        self._entries: list[ttk.Entry] = []
        self._active: ttk.Entry | None = None

        self._build_ui()
        for var in (self._rate, self._duration, *self._ranges):
            var.trace_add("write", lambda *_: self._schedule_update())
        for i, channel in enumerate(segments):
            self._set_segments(i, channel)
        self._entries[0].focus_set()
        self._update()

    def destroy(self) -> None:
        """Closed with the main window: remember what was typed."""
        self.save_settings()
        super().destroy()

    def restyle(self) -> None:
        """Draw the preview again in the colours of the current theme."""
        self._drawn = None
        self._figure.clear()
        self._figure.set_facecolor(plot_style.PANEL)
        if self._waveform is not None:
            self._draw_preview()
        else:
            self._canvas.draw_idle()

    def save_settings(self) -> None:
        user_settings.save_section(_SETTINGS, {
            "sample_rate_hz": self._rate.get(),
            "duration_ms": self._duration.get(),
            "ranges": [r.get() for r in self._ranges],
            "segments": [[[s.formula.get(), s.start.get(), s.stop.get()] for s in channel]
                         for channel in self._segments],
        })

    def _build_ui(self) -> None:
        head = theme.card(self, padding=12)
        head.pack(side="top", fill="x")
        theme.heading(head.body, "AWG generator").pack(side="left")
        self._summary = theme.Badge(head.body, "", theme.MUTED)
        self._summary.pack(side="left", padx=10)
        ttk.Button(head.body, text="Save…", style="Accent.TButton", command=self._save).pack(side="right")
        ttk.Button(head.body, text="Fit formulas", command=self._fit).pack(side="right", padx=(0, 6))
        ttk.Button(head.body, text="Open…", command=self._open).pack(side="right", padx=(0, 6))
        self._source_label = ttk.Label(head.body, text="", style="Card.TLabel", foreground=theme.MUTED,
                                       font=theme.mono(9))
        self._source_label.pack(side="right", padx=(0, 12))

        top_card = theme.card(self, padding=12)
        top_card.pack(side="top", fill="x", pady=(12, 0))
        top = top_card.body

        keypad = ttk.Frame(top, style="Card.TFrame")
        keypad.pack(side="right", anchor="n", padx=(16, 0))
        left = ttk.Frame(top, style="Card.TFrame")
        left.pack(side="left", fill="both", expand=True)
        params = ttk.Frame(left, style="Card.TFrame")
        params.pack(side="top", fill="x")
        theme.caption(params, "Sample rate (Hz)").pack(side="left")
        ttk.Entry(params, textvariable=self._rate, width=10, font=theme.mono(10)).pack(side="left", padx=(6, 18))
        theme.caption(params, "Duration (ms)").pack(side="left")
        ttk.Entry(params, textvariable=self._duration, width=10, font=theme.mono(10)).pack(side="left", padx=(6, 18))
        self._general = ttk.Label(params, text="", style="Card.TLabel", font=theme.mono(9))
        self._general.pack(side="left")
        ttk.Label(left, text="Each channel is made of segments: a formula from t_start to t_stop (ms). An empty "
                             "t_start is where the previous segment stops, an empty t_stop where the next one "
                             "starts (or the end). t is the time after :AWG:RUN, tau the time from the start of "
                             "the segment; the times in step, pulse, ramp, gauss, hann and chirp also count "
                             "from the start of the segment. Outside the segments the channel is 0.",
                  style="Card.TLabel", foreground=theme.MUTED, font=theme.sans(9), wraplength=760,
                  justify="left").pack(side="top", anchor="w", pady=(8, 0))

        self._page = theme.ScrollPage(left, background=theme.PANEL, style="Card.TFrame")
        self._page.view.config(height=CHANNELS_HEIGHT_PX)
        self._page.pack(side="top", fill="x", pady=(8, 0))
        self._channel_frames: list[ttk.Frame] = []
        self._status: list[ttk.Label] = []
        for i in range(N_CHANNELS):
            block = ttk.Frame(self._page.inner, style="Card.TFrame", padding=(0, 4, 8, 6))
            block.pack(side="top", fill="x")
            head_row = ttk.Frame(block, style="Card.TFrame")
            head_row.pack(side="top", fill="x")
            ttk.Label(head_row, text=f"CH{i + 1}", style="Card.TLabel", foreground=plot_style.series(i),
                      font=theme.mono(10, "bold"), width=5).pack(side="left")
            ttk.Combobox(head_row, textvariable=self._ranges[i], values=list(_RANGES), state="readonly",
                         width=12, font=theme.mono(9)).pack(side="left")
            add = ttk.Button(head_row, text="+ segment", style="Small.TButton",
                             command=lambda c=i: self._add_segment(c))
            add.pack(side="left", padx=(10, 0))
            Tooltip(add, "Add a segment: the next piece of this channel, with its own formula")
            status = ttk.Label(head_row, text="", style="Card.TLabel", foreground=theme.MUTED, font=theme.mono(8))
            status.pack(side="left", padx=(12, 0))
            self._status.append(status)
            rows = ttk.Frame(block, style="Card.TFrame")
            rows.pack(side="top", fill="x", pady=(4, 0))
            rows.columnconfigure(1, weight=1)
            self._channel_frames.append(rows)
            for widget in (block, head_row, rows):
                self._page.bind_wheel(widget)

        for g, (title, rows) in enumerate(_KEYPAD):
            group = ttk.LabelFrame(keypad, text=title, padding=4)
            group.grid(row=0, column=g, sticky="ns", padx=(0 if g == 0 else 6, 0))
            for r, row in enumerate(rows):
                for c, key in enumerate(row):
                    if not key:
                        continue
                    button = ttk.Button(group, text=_KEY_LABELS.get(key, key), width=6, takefocus=False,
                                        style="Key.TButton", command=lambda k=key: self._key(k))
                    button.grid(row=r, column=c, padx=1, pady=1)
                    if key in _KEY_NAMES:
                        Tooltip(button, _KEY_NAMES[key])

        preview = theme.card(self, padding=12)
        preview.pack(side="top", fill="both", expand=True, pady=(12, 0))
        theme.caption(preview.body, "Preview - current in mA versus time after :AWG:RUN, as the AWG outputs it "
                                    "(every sample held until the next)").pack(side="top", anchor="w")
        self._figure = Figure(figsize=(11, 6), layout="constrained")
        self._canvas = FigureCanvasTkAgg(self._figure, master=preview.body)
        toolbar = NavigationToolbar2Tk(self._canvas, preview.body, pack_toolbar=False)
        toolbar.update()
        theme.style_toolbar(toolbar)
        toolbar.pack(side="bottom", fill="x")
        self._axis_drag = AxisDrag(self._canvas, toolbar)
        theme.figure_widget(self._canvas).pack(side="top", fill="both", expand=True, pady=(6, 0))

    def _set_segments(self, channel: int, segments: list[Segment]) -> None:
        """Replace the segments of a channel (at least one row stays)."""
        self._segments[channel] = [_SegmentVars(s.formula, s.start_ms, s.stop_ms) for s in segments] or [
            _SegmentVars()]
        for vars_ in self._segments[channel]:
            for var in (vars_.formula, vars_.start, vars_.stop):
                var.trace_add("write", lambda *_: self._schedule_update())
        self._render_channel(channel)
        self._schedule_update()

    def _render_channel(self, channel: int) -> None:
        """Build the rows of one channel's segments again."""
        frame = self._channel_frames[channel]
        for child in frame.winfo_children():
            if child in self._entries_of(frame):
                self._entries.remove(child)
            child.destroy()
        segments = self._segments[channel]
        for r, vars_ in enumerate(segments):
            ttk.Label(frame, text=f"{r + 1}." if len(segments) > 1 else "", style="Card.TLabel",
                      foreground=theme.DIM, font=theme.mono(8), width=4).grid(row=r, column=0, sticky="w")
            entry = ttk.Entry(frame, textvariable=vars_.formula, font=theme.mono(10))
            entry.grid(row=r, column=1, sticky="ew", padx=(0, 8), pady=1)
            entry.bind("<FocusIn>", lambda _e, w=entry: setattr(self, "_active", w))
            entry._channel_entry = True
            self._entries.append(entry)
            example = _EXAMPLES[channel] if r == 0 else "e.g. 100*sin(2*pi*1kHz*tau)"
            ghost = GhostText(entry, example if example.startswith("e.g.") else f"e.g. {example}")
            vars_.formula.trace_add("write", lambda *_, g=ghost: self.after_idle(g.update))
            theme.caption(frame, "t_start").grid(row=r, column=2, padx=(0, 4))
            start = ttk.Entry(frame, textvariable=vars_.start, width=8, font=theme.mono(10))
            start.grid(row=r, column=3)
            theme.caption(frame, "t_stop").grid(row=r, column=4, padx=(8, 4))
            stop = ttk.Entry(frame, textvariable=vars_.stop, width=8, font=theme.mono(10))
            stop.grid(row=r, column=5)
            theme.caption(frame, "ms").grid(row=r, column=6, padx=(4, 6))
            remove = ttk.Button(frame, text="✕", style="Small.TButton", width=3,
                                command=lambda c=channel, n=r: self._remove_segment(c, n))
            remove.grid(row=r, column=7)
            if len(segments) == 1:
                remove.state(["disabled"])
            for widget in (entry, start, stop, remove):
                self._page.bind_wheel(widget)
        if self._active is None or not self._active.winfo_exists():
            self._active = self._entries[0] if self._entries else None

    @staticmethod
    def _entries_of(frame: ttk.Frame) -> list[tk.Misc]:
        return [w for w in frame.winfo_children() if getattr(w, "_channel_entry", False)]

    def _add_segment(self, channel: int) -> None:
        """A new row after the last one; its t_start (to be typed) ends the previous segment."""
        segments = [v.segment() for v in self._segments[channel]]
        segments.append(Segment("", segments[-1].stop_ms.strip(), ""))
        self._set_segments(channel, segments)
        self._entries_of(self._channel_frames[channel])[-1].focus_set()

    def _remove_segment(self, channel: int, index: int) -> None:
        segments = [v.segment() for v in self._segments[channel]]
        if len(segments) > 1:
            del segments[index]
            self._set_segments(channel, segments)

    def _key(self, key: str) -> None:
        entry = self._active
        if entry is None or not entry.winfo_exists():
            return
        entry.focus_set()
        if key == "C":
            entry.delete(0, "end")
        elif key == "<":
            pos = entry.index("insert")
            if pos > 0:
                entry.delete(pos - 1)
        else:
            entry.insert("insert", f"{key}(" if key in _FUNCTIONS else key)

    def _schedule_update(self) -> None:
        if self._pending is None:
            self._pending = self.after_idle(self._update)

    def _channels(self) -> list[ChannelFormula]:
        return [ChannelFormula([v.segment() for v in self._segments[i]], _RANGES[self._ranges[i].get()])
                for i in range(N_CHANNELS)]

    def _has_formula(self, channel: int) -> bool:
        return any(v.formula.get().strip() for v in self._segments[channel])

    def _update(self) -> None:
        """Evaluate the formulas, show every problem at once and redraw the preview."""
        self._pending = None
        try:
            rate, duration = float(self._rate.get().replace(",", ".")), float(self._duration.get().replace(",", "."))
        except ValueError:
            self._show(["sample rate and duration must be numbers"], [None] * N_CHANNELS, None)
            return
        waveform, errors, channel_errors = build_waveform(
            self._channels(), rate, duration, self._max_rate, self._max_amp, self._source)
        self._show(errors, channel_errors, waveform)

    def _show(self, errors: list[str], channel_errors: list[str | None], waveform: AwgWaveform | None) -> None:
        self._errors = errors + [f"CH{i + 1}: {e}" for i, e in enumerate(channel_errors) if e]
        self._waveform = waveform if not self._errors else None
        self._general.config(text="; ".join(errors), foreground=theme.DANGER)
        for i, label in enumerate(self._status):
            if channel_errors[i]:
                label.config(text=channel_errors[i], foreground=theme.DANGER)
            elif waveform is not None:
                peak = abs(waveform.currents_A[:, i]).max() * 1000
                label.config(text=f"OK, peak {peak:.4g} mA" if self._has_formula(i) else "0 mA (empty)",
                             foreground=theme.OK if self._has_formula(i) else theme.MUTED)
            else:
                label.config(text="")
        if not self._errors:
            self._summary.set(
                f"{waveform.n_samples} samples / {waveform.sample_rate_hz:g} Hz / {waveform.duration_s * 1000:g} ms",
                theme.ACCENT)
            if self._pending_preview is not None:
                self.after_cancel(self._pending_preview)
            self._pending_preview = self.after(PREVIEW_DELAY_MS, self._draw_preview)
        else:
            self._summary.set("fix the errors - the preview shows the last valid version", theme.DANGER)

    def _draw_preview(self) -> None:
        self._pending_preview = None
        waveform = self._waveform
        if waveform is None or _same_waveform(waveform, self._drawn):
            return
        self._drawn = waveform
        self._figure.clear()
        self._figure.set_facecolor(plot_style.PANEL)
        plot_awg(waveform, fig=self._figure, show_title=False)
        self._canvas.draw_idle()

    def _save(self) -> None:
        if self._pending is not None:
            self.after_cancel(self._pending)
            self._update()
        if self._waveform is None:
            messagebox.showerror("Cannot save", "Fix these problems first:\n\n" + "\n".join(self._errors),
                                 parent=self)
            return
        path = filedialog.asksaveasfilename(
            parent=self, title="Save AWG sequence", initialdir=self._initial_dir,
            defaultextension=".npz", filetypes=[("AWG sequence", "*.npz")],
        )
        if not path:
            return
        try:
            saved = save_waveform(path, self._waveform, self._channels(), float(self._duration.get()),
                                  self._max_rate, self._max_amp, self._source)
        except (OSError, ValueError) as e:
            messagebox.showerror("Cannot save", english_error(e), parent=self)
            return
        self._initial_dir = Path(path).parent
        if self._on_saved:
            self._on_saved(Path(path), saved)
        messagebox.showinfo("Saved", f"{Path(path).name}\n{saved.n_samples} samples, {saved.sample_rate_hz:g} Hz, "
                                     f"{saved.duration_s * 1000:g} ms\n\nUse it in an experiment file:\n"
                                     f'"awg_sequence": {{"type": "npz", "file_path": "sequences/{Path(path).name}"}}',
                            parent=self)

    def _fit(self) -> None:
        """Replace every data(n) formula with a guessed formula, where one fits the samples."""
        if self._source is None:
            messagebox.showinfo("Fit formulas", "Open an AWG file first - its channels (data(n)) are "
                                                "what the formulas are fitted to.", parent=self)
            return
        self.config(cursor="watch")
        self.update_idletasks()
        try:
            lines = self._fit_channels()
        finally:
            self.config(cursor="")
        messagebox.showinfo("Fit formulas", "\n".join(lines), parent=self)

    def _fit_channels(self) -> list[str]:
        lines = []
        for i, channel_formula in enumerate(self._channels()):
            text = channel_formula.formula.strip().replace(" ", "")
            n = i + 1
            if not (text.startswith("data(") and text.endswith(")") and text[5:-1].isdigit()):
                lines.append(f"CH{n}: unchanged (not one segment with a plain data(n) formula)")
                continue
            channel = int(text[5:-1])
            if not 1 <= channel <= N_CHANNELS:
                lines.append(f"CH{n}: unchanged ({text} is not a channel)")
                continue
            step_mA = RANGE_FULL_SCALE_A[_RANGES[self._ranges[i].get()]] * 1000 / np.iinfo(np.int16).max
            result = fit_channel(self._source.currents_mA[:, channel - 1], self._source.sample_rate_hz, step_mA)
            if result.formula is None:
                lines.append(f"CH{n}: kept {text} - {result.note}")
            else:
                self._segments[i][0].formula.set(result.formula)
                lines.append(f"CH{n}: {result.note}, max error {result.max_error_mA:.2g} mA")
        return lines

    def _open(self) -> None:
        path = filedialog.askopenfilename(parent=self, title="Open AWG sequence", initialdir=self._initial_dir,
                                          filetypes=[("AWG sequence", "*.npz")])
        if not path:
            return
        try:
            channels, rate, duration, source = open_for_editing(path, self._max_rate, self._max_amp)
        except (OSError, ValueError, KeyError) as e:
            messagebox.showerror("Cannot open", english_error(e), parent=self)
            return
        self._source = source
        self._source_label.config(text=f"data(n) = channels of {source.name}" if source else "")
        self._rate.set(f"{rate:g}")
        self._duration.set(f"{duration:g}")
        for i, ch in enumerate(channels[:N_CHANNELS]):
            self._set_segments(i, ch.segments)
            self._ranges[i].set(next(k for k, v in _RANGES.items() if v == ch.range_))
        self._initial_dir = Path(path).parent


def _same_waveform(a: AwgWaveform, b: AwgWaveform | None) -> bool:
    return (b is not None and a.sample_rate_hz == b.sample_rate_hz and a.ranges == b.ranges
            and a.source_path == b.source_path and np.array_equal(a.currents_A, b.currents_A))


def _saved_segments(saved: dict) -> list[list[Segment]]:
    """The segments typed last time; the formulas of older settings become one segment each."""
    segments = saved.get("segments")
    if (isinstance(segments, list) and len(segments) == N_CHANNELS
            and all(isinstance(ch, list) and ch and all(isinstance(x, list) and len(x) == 3
                                                         and all(isinstance(v, str) for v in x) for x in ch)
                    for ch in segments)):
        return [[Segment(*x) for x in ch] for ch in segments]
    formulas = saved.get("formulas")
    if not (isinstance(formulas, list) and len(formulas) == N_CHANNELS and all(isinstance(f, str) for f in formulas)):
        formulas = ["200*pulse(5*ms, 15*ms)"] + [""] * (N_CHANNELS - 1)
    return [[Segment(f)] for f in formulas]
