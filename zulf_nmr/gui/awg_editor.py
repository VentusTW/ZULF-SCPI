"""Window for writing an AWG sequence (.npz) from formulas, with a calculator keypad."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import numpy as np

from zulf_nmr.hardware.awg import N_CHANNELS, RANGE_FULL_SCALE_A, AwgRange, AwgWaveform
from zulf_nmr.hardware.awg_fit import fit_channel
from zulf_nmr.hardware.awg_formula import (
    ChannelFormula,
    SourceData,
    build_waveform,
    open_for_editing,
    save_waveform,
)
from zulf_nmr.utils.visualization.awg_plotter import plot_awg

_COLOR_ERROR = "#d0342c"
_RANGES = {"HIGH (1 A)": AwgRange.HIGH, "LOW (10 mA)": AwgRange.LOW}

# Calculator keypad: (label, text inserted). "<" = backspace, "C" = clear the formula.
_KEYPAD = [
    ["7", "8", "9", "/", "sin", "cos", "tan", "pi"],
    ["4", "5", "6", "*", "exp", "log", "sqrt", "e"],
    ["1", "2", "3", "-", "abs", "min", "max", "t"],
    ["0", ".", ",", "+", "^", "(", ")", "ms"],
    ["step", "pulse", "ramp", "gauss", "hann", "chirp", "us", "kHz"],
]
_FUNCTIONS = {"sin", "cos", "tan", "exp", "log", "sqrt", "abs", "min", "max",
              "step", "pulse", "ramp", "gauss", "hann", "chirp", "data"}
_HELP = (
    "Current in mA as a function of t (seconds after :AWG:RUN). A unit can follow a number: 1.6ms, 2 s, 20us,\n"
    "300Hz, 5kHz, 0.2A (= 200 mA). Units: s sec ms us ns  Hz kHz MHz  mA uA A.\n"
    "step(t0)   pulse(t0, t1)   ramp(t0, t1)   gauss(t0, sigma)   hann(t0, t1)   chirp(f0, f1, t0, t1)\n"
    "data(n) = channel n of the file opened with Open... (any AWG .npz)\n"
    "e.g.  200*pulse(5ms, 15ms)      8*sin(2*pi*300Hz*t)*exp(-t/15ms)      300*chirp(100Hz, 800Hz, 10ms, 60ms)"
)


class AwgEditor(tk.Toplevel):
    """Formulas for the 4 AWG channels, a live preview and Save as .npz.

    The file is written only when the user presses Save and every channel is valid;
    it is then read back with the same checks as an experiment uses (load_awg_npz).
    """

    def __init__(self, parent: tk.Misc, initial_dir: Path, max_sample_rate_hz: float,
                 max_amplitude_A: float, on_saved=None) -> None:
        super().__init__(parent)
        self.title("AWG sequence editor")
        self.geometry(f"{min(1250, self.winfo_screenwidth() - 80)}x{min(900, self.winfo_screenheight() - 100)}+60+30")
        self._initial_dir = initial_dir
        self._max_rate = max_sample_rate_hz
        self._max_amp = max_amplitude_A
        self._on_saved = on_saved
        self._pending: str | None = None
        self._waveform: AwgWaveform | None = None
        self._errors: list[str] = []
        self._source: SourceData | None = None  # samples of the opened file, for data(n)

        self._rate = tk.StringVar(value="10000")
        self._duration = tk.StringVar(value="50")
        self._formulas = [tk.StringVar() for _ in range(N_CHANNELS)]
        self._ranges = [tk.StringVar(value="HIGH (1 A)") for _ in range(N_CHANNELS)]
        self._formulas[0].set("200*pulse(5*ms, 15*ms)")

        self._build_ui()
        for var in (self._rate, self._duration, *self._formulas, *self._ranges):
            var.trace_add("write", lambda *_: self._schedule_update())
        self._active = self._entries[0]
        self._entries[0].focus_set()
        self._update()

    # -------------------------------------------------------------------- UI

    def _build_ui(self) -> None:
        bottom = ttk.Frame(self, padding=8)
        bottom.pack(side="bottom", fill="x")
        ttk.Button(bottom, text="Open…", command=self._open).pack(side="left")
        ttk.Button(bottom, text="Fit formulas", command=self._fit).pack(side="left", padx=(6, 0))
        self._source_label = ttk.Label(bottom, text="", foreground="#666666")
        self._source_label.pack(side="left", padx=(8, 0))
        self._summary = ttk.Label(bottom, text="")
        self._summary.pack(side="left", padx=12)
        ttk.Button(bottom, text="Close", command=self.destroy).pack(side="right")
        ttk.Button(bottom, text="Save…", command=self._save).pack(side="right", padx=(0, 6))

        top = ttk.Frame(self, padding=(8, 8, 8, 0))
        top.pack(side="top", fill="x")

        left = ttk.Frame(top)
        left.pack(side="left", fill="both", expand=True)
        params = ttk.Frame(left)
        params.pack(side="top", fill="x")
        ttk.Label(params, text="Sample rate (Hz):").pack(side="left")
        ttk.Entry(params, textvariable=self._rate, width=10).pack(side="left", padx=(4, 16))
        ttk.Label(params, text="Duration (ms):").pack(side="left")
        ttk.Entry(params, textvariable=self._duration, width=10).pack(side="left", padx=(4, 16))
        self._general = ttk.Label(params, text="")
        self._general.pack(side="left")

        grid = ttk.Frame(left)
        grid.pack(side="top", fill="x", pady=8)
        grid.columnconfigure(2, weight=1)
        self._entries: list[ttk.Entry] = []
        self._status: list[ttk.Label] = []
        for i in range(N_CHANNELS):
            ttk.Label(grid, text=f"CH{i + 1}").grid(row=2 * i, column=0, sticky="w", padx=(0, 6))
            ttk.Combobox(grid, textvariable=self._ranges[i], values=list(_RANGES), state="readonly",
                         width=12).grid(row=2 * i, column=1, sticky="w")
            entry = ttk.Entry(grid, textvariable=self._formulas[i], font=("Consolas", 10))
            entry.grid(row=2 * i, column=2, sticky="ew", padx=6)
            entry.bind("<FocusIn>", lambda _e, w=entry: setattr(self, "_active", w))
            self._entries.append(entry)
            status = ttk.Label(grid, text="", foreground="#666666")
            status.grid(row=2 * i + 1, column=2, sticky="w", padx=6, pady=(0, 4))
            self._status.append(status)
        ttk.Label(left, text=_HELP, foreground="#444444", font=("Consolas", 9), justify="left").pack(
            side="top", anchor="w")

        keypad = ttk.LabelFrame(top, text="Calculator", padding=6)
        keypad.pack(side="right", anchor="n", padx=(12, 0))
        for r, row in enumerate(_KEYPAD):
            for c, key in enumerate(row):
                ttk.Button(keypad, text=key, width=6, takefocus=False,
                           command=lambda k=key: self._key(k)).grid(row=r, column=c, padx=1, pady=1)
        last = len(_KEYPAD)
        ttk.Button(keypad, text="data", width=6, takefocus=False,
                   command=lambda: self._key("data")).grid(row=last, column=0, padx=1, pady=1)
        for column, unit in ((1, "s"), (2, "Hz")):
            ttk.Button(keypad, text=unit, width=6, takefocus=False,
                       command=lambda k=unit: self._key(k)).grid(row=last, column=column, padx=1, pady=1)
        ttk.Button(keypad, text="⌫", width=6, takefocus=False,
                   command=lambda: self._key("<")).grid(row=last, column=6, padx=1, pady=1)
        ttk.Button(keypad, text="C", width=6, takefocus=False,
                   command=lambda: self._key("C")).grid(row=last, column=7, padx=1, pady=1)
        ttk.Label(keypad, text="Types at the cursor.", foreground="#666666").grid(
            row=last, column=3, columnspan=3, sticky="w")

        self._figure = Figure(figsize=(11, 6), layout="constrained")
        self._canvas = FigureCanvasTkAgg(self._figure, master=self)
        self._canvas.get_tk_widget().pack(side="top", fill="both", expand=True, padx=8, pady=8)

    def _key(self, key: str) -> None:
        entry = self._active
        entry.focus_set()
        if key == "C":
            entry.delete(0, "end")
        elif key == "<":
            pos = entry.index("insert")
            if pos > 0:
                entry.delete(pos - 1)
        else:
            entry.insert("insert", f"{key}(" if key in _FUNCTIONS else key)

    # ---------------------------------------------------------------- update

    def _schedule_update(self) -> None:
        # At once, as soon as Tk is idle: several changes in one event (e.g. Open... setting
        # every field) still give one update.
        if self._pending is None:
            self._pending = self.after_idle(self._update)

    def _channels(self) -> list[ChannelFormula]:
        return [ChannelFormula(self._formulas[i].get(), _RANGES[self._ranges[i].get()]) for i in range(N_CHANNELS)]

    def _update(self) -> None:
        """Evaluate the formulas, show every problem at once and redraw the preview."""
        self._pending = None
        try:
            rate, duration = float(self._rate.get().replace(",", ".")), float(self._duration.get().replace(",", "."))
        except ValueError:
            self._show([f"sample rate and duration must be numbers"], [None] * N_CHANNELS, None)
            return
        waveform, errors, channel_errors = build_waveform(
            self._channels(), rate, duration, self._max_rate, self._max_amp, self._source)
        self._show(errors, channel_errors, waveform)

    def _show(self, errors: list[str], channel_errors: list[str | None], waveform: AwgWaveform | None) -> None:
        self._errors = errors + [f"CH{i + 1}: {e}" for i, e in enumerate(channel_errors) if e]
        self._waveform = waveform if not self._errors else None
        self._general.config(text="; ".join(errors), foreground=_COLOR_ERROR)
        for i, label in enumerate(self._status):
            if channel_errors[i]:
                label.config(text=channel_errors[i], foreground=_COLOR_ERROR)
            elif waveform is not None:
                peak = abs(waveform.currents_A[:, i]).max() * 1000
                label.config(text=f"OK, peak {peak:.4g} mA" if self._formulas[i].get().strip() else "0 mA (empty)",
                             foreground="#2e9e44" if self._formulas[i].get().strip() else "#666666")
            else:
                label.config(text="")
        if not self._errors:
            # Every channel is OK: draw it now.
            self._summary.config(
                text=f"{waveform.n_samples} samples, {waveform.sample_rate_hz:g} Hz, {waveform.duration_s * 1000:g} ms",
                foreground="")
            self._figure.clear()
            plot_awg(waveform, fig=self._figure)
            self._canvas.draw_idle()
        else:
            # E.g. a formula half typed: keep the last valid preview instead of a broken one.
            self._summary.config(text="fix the errors - the preview shows the last valid version",
                                 foreground=_COLOR_ERROR)

    # ------------------------------------------------------------ open, save

    def _save(self) -> None:
        if self._pending is not None:  # the user typed and pressed Save at once
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
            messagebox.showerror("Cannot save", str(e), parent=self)
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
        lines = []
        for i, var in enumerate(self._formulas):
            text = var.get().strip().replace(" ", "")
            n = i + 1
            if not (text.startswith("data(") and text.endswith(")") and text[5:-1].isdigit()):
                lines.append(f"CH{n}: unchanged (not a plain data(n) formula)")
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
                var.set(result.formula)
                lines.append(f"CH{n}: {result.note}, max error {result.max_error_mA:.2g} mA")
        messagebox.showinfo("Fit formulas", "\n".join(lines), parent=self)

    def _open(self) -> None:
        path = filedialog.askopenfilename(parent=self, title="Open AWG sequence", initialdir=self._initial_dir,
                                          filetypes=[("AWG sequence", "*.npz")])
        if not path:
            return
        try:
            channels, rate, duration, source = open_for_editing(path, self._max_rate, self._max_amp)
        except (OSError, ValueError, KeyError) as e:
            messagebox.showerror("Cannot open", str(e), parent=self)
            return
        self._source = source
        self._source_label.config(text=f"data(n) = channels of {source.name}" if source else "")
        self._rate.set(f"{rate:g}")
        self._duration.set(f"{duration:g}")
        for i, ch in enumerate(channels[:N_CHANNELS]):
            self._formulas[i].set(ch.formula)
            self._ranges[i].set(next(k for k, v in _RANGES.items() if v == ch.range_))
        self._initial_dir = Path(path).parent
