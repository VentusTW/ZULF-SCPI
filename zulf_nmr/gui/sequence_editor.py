"""Sequence editor panel: the events of an experiment as blocks on a time axis, dragged
with the mouse, with the experiment's settings; saved as an experiment JSON file."""

from __future__ import annotations

import json
import os
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable

from zulf_nmr.core.errors import english_error
from zulf_nmr.core.scpi import header_matches
from zulf_nmr.experiment.parser import ConfigError, SystemConfig, load_experiment_config, parse_experiment
from zulf_nmr.gui import theme
from zulf_nmr.gui.tooltip import Tooltip
from zulf_nmr.hardware.acquisition import ADC_CHANNELS
from zulf_nmr.hardware.awg import load_awg_npz
from zulf_nmr.hardware.digital import Digital
from zulf_nmr.hardware.motor import MotorPosition
from zulf_nmr.hardware.shim import Shim
from zulf_nmr.utils.visualization.events_plotter import event_category

GUTTER_PX = 230
ROW_PX = 30
AXIS_PX = 34
MIN_BLOCK_PX = 10
SNAPS_MS = ("1", "10", "50", "100", "500", "1000")
ZOOM_STEP = 1.25
POSITIONS = tuple(p.value for p in MotorPosition if p is not MotorPosition.OTHER)

KINDS: dict[str, tuple[str, tuple[str, ...]]] = {
    "Shim current": (":SHIM:CURRent", ("channel", "mA")),
    "Motor move": (":MOTor:MOVE", ("position",)),
    "Digital state": (":DIGital:STATe", ("channel", "state")),
    "AWG run": (":AWG:RUN", ()),
    "AWG reset": (":AWG:RST", ()),
    "Acquire run": (":ACQuire:RUN", ()),
}
DEFAULT_ARGS = {"Shim current": ("1", "0.0"), "Motor move": ("MAGNET",), "Digital state": ("1", "1")}


def action_kind(action: str) -> tuple[str | None, list[str]]:
    """(kind of KINDS, arguments) of a command; None for a command that is none of them."""
    header, _, rest = action.strip().partition(" ")
    args = [a.strip() for a in rest.split(",")] if rest.strip() else []
    for kind, (pattern, _params) in KINDS.items():
        if header and header_matches(header if header.startswith(":") else ":" + header, pattern):
            return kind, args
    return None, args


def build_action(kind: str, args: list[str]) -> str:
    pattern, params = KINDS[kind]
    values = [a.strip() for a in args[:len(params)]]
    return f"{pattern} {', '.join(values)}" if params else pattern


class _Event:
    def __init__(self, time_ms: int, action: str) -> None:
        self.time_ms = time_ms
        self.action = action


class SequenceEditorPanel(ttk.Frame):
    """Build or change an experiment file.

    The timeline shows one row per event: a dot at its time and a block as long as its
    duration (from system.json; :AWG:RUN lasts as long as the AWG sequence), coloured
    by subsystem; drag a block to move the event (snapped to the grid). The acquisition
    is the last row. The file is checked as it is edited, with the checks used when an
    experiment is loaded, and saved only when it is valid.
    """

    def __init__(self, parent: tk.Misc, system: Callable[[], SystemConfig], data_dir: Path,
                 on_saved: Callable[[Path], None] | None = None,
                 on_add_to_queue: Callable[[Path], None] | None = None) -> None:
        super().__init__(parent, padding=(18, 16, 18, 16))
        self._system = system
        self._data_dir = data_dir
        self._on_saved = on_saved
        self._on_add_to_queue = on_add_to_queue
        self._path: Path | None = None
        self._metadata: dict[str, Any] = {}
        self._output: dict[str, Any] = {"format": "csv", "auto_save": True}
        self._events: list[_Event] = []
        self._selected: int | None = None
        self._scale = 0.05
        self._drag: tuple[int, float, int] | None = None
        self._filling = False
        self._pending: str | None = None
        self._errors: list[str] = []
        self._awg_cache: tuple[Path, float, float | None] | None = None
        self._auto_fit = True

        self._experiment = tk.StringVar(value="New_Experiment")
        self._transients = tk.StringVar(value="1")
        self._rate = tk.StringVar(value="1000")
        self._acq_start = tk.StringVar(value="0")
        self._samples = tk.StringVar(value="1000")
        self._channels = {ch: tk.BooleanVar(value=ch == 1) for ch in ADC_CHANNELS}
        self._awg_file = tk.StringVar(value="")
        self._snap = tk.StringVar(value="10")
        self._time = tk.StringVar()
        self._kind = tk.StringVar()
        self._arg1 = tk.StringVar()
        self._arg2 = tk.StringVar()
        self._command = tk.StringVar()

        self._build_ui()
        for var in (self._experiment, self._transients, self._rate, self._acq_start, self._samples, self._awg_file,
                    *self._channels.values()):
            var.trace_add("write", lambda *_: self._changed())
        for var in (self._time, self._arg1, self._arg2):
            var.trace_add("write", lambda *_: self._form_changed())
        self._kind.trace_add("write", lambda *_: self._kind_changed())
        self._command.trace_add("write", lambda *_: self._command_changed())
        self._new(ask=False)

    def _build_ui(self) -> None:
        head = theme.card(self, padding=12)
        head.pack(side="top", fill="x")
        theme.heading(head.body, "Sequence editor").pack(side="left")
        self._file_badge = theme.Badge(head.body, "new file", theme.MUTED)
        self._file_badge.pack(side="left", padx=10)
        self._valid_badge = theme.Badge(head.body, "", theme.OK)
        self._valid_badge.pack(side="left")
        add = ttk.Button(head.body, text="Add to queue", command=self._add_to_queue)
        add.pack(side="right")
        Tooltip(add, "Save the file and add it to the experiment queue")
        ttk.Button(head.body, text="Save…", style="Accent.TButton", command=self._save).pack(side="right", padx=(0, 6))
        ttk.Button(head.body, text="Open…", command=self._open).pack(side="right", padx=(0, 6))
        ttk.Button(head.body, text="New", command=self._new).pack(side="right", padx=(0, 6))

        settings = theme.card(self, padding=12)
        settings.pack(side="top", fill="x", pady=(12, 0))
        grid = settings.body
        fields = ((0, 0, "Experiment name", self._experiment, 26), (0, 1, "Transients", self._transients, 8),
                  (0, 2, "ADC rate (Hz)", self._rate, 9), (0, 3, "Samples", self._samples, 9),
                  (1, 0, "Acquisition start (us after :AWG:RUN)", self._acq_start, 10))
        for row, col, label, var, width in fields:
            theme.caption(grid, label).grid(row=2 * row, column=col, sticky="w", padx=(0, 16))
            ttk.Entry(grid, textvariable=var, width=width, font=theme.mono(10)).grid(
                row=2 * row + 1, column=col, sticky="w", padx=(0, 16), pady=(2, 8))
        theme.caption(grid, "ADC channels").grid(row=2, column=1, columnspan=3, sticky="w")
        boxes = ttk.Frame(grid, style="Card.TFrame")
        boxes.grid(row=3, column=1, columnspan=3, sticky="w", pady=(2, 8))
        for ch, var in self._channels.items():
            ttk.Checkbutton(boxes, text=str(ch), variable=var, style="Card.TCheckbutton").pack(side="left")
        theme.caption(grid, "AWG sequence file (relative to the experiment file)").grid(
            row=0, column=4, sticky="w")
        awg = ttk.Frame(grid, style="Card.TFrame")
        awg.grid(row=1, column=4, sticky="ew", pady=(2, 8))
        ttk.Entry(awg, textvariable=self._awg_file, font=theme.mono(10)).pack(side="left", fill="x", expand=True)
        ttk.Button(awg, text="Browse…", style="Small.TButton", command=self._browse_awg).pack(side="left", padx=4)
        ttk.Button(awg, text="None", style="Small.TButton", command=lambda: self._awg_file.set("")).pack(side="left")
        self._awg_info = ttk.Label(grid, text="", style="Card.TLabel", foreground=theme.MUTED, font=theme.mono(8))
        self._awg_info.grid(row=2, column=4, rowspan=2, sticky="nw")
        grid.columnconfigure(4, weight=1)

        bottom = ttk.Frame(self)
        bottom.pack(side="bottom", fill="x", pady=(12, 0))
        form = theme.card(bottom, padding=12)
        form.pack(side="left", fill="both", expand=True)
        self._build_form(form.body)
        problems = theme.card(bottom, padding=12)
        problems.pack(side="left", fill="both", expand=True, padx=(12, 0))
        theme.caption(problems.body, "Check of the experiment file").pack(side="top", anchor="w")
        self._problems = tk.Text(problems.body, height=7, wrap="word", state="disabled", font=theme.mono(9),
                                 background=theme.FIELD, foreground=theme.TEXT, borderwidth=0,
                                 highlightthickness=0, padx=8, pady=6)
        self._problems.tag_config("error", foreground=theme.DANGER)
        self._problems.tag_config("ok", foreground=theme.OK)
        self._problems.pack(side="top", fill="both", expand=True, pady=(6, 0))

        timeline = theme.card(self, padding=12)
        timeline.pack(side="top", fill="both", expand=True, pady=(12, 0))
        bar = ttk.Frame(timeline.body, style="Card.TFrame")
        bar.pack(side="top", fill="x", pady=(0, 8))
        theme.heading(bar, "Sequence timeline", size=13).pack(side="left")
        add = ttk.Menubutton(bar, text="+ Add event", style="Accent.TButton")
        menu = tk.Menu(add, tearoff=False, background=theme.PANEL, foreground=theme.TEXT,
                       activebackground=theme.SELECTION, activeforeground=theme.TEXT, font=theme.mono(9))
        for kind in KINDS:
            menu.add_command(label=kind, command=lambda k=kind: self._add_event(k))
        add["menu"] = menu
        add.pack(side="left", padx=(16, 6))
        for text, command, tip in (("Duplicate", self._duplicate, "Copy the selected event (one grid step later)"),
                                   ("Delete", self._delete, "Remove the selected event (Delete key)"),
                                   ("Sort by time", self._sort, "Order the rows by time")):
            button = ttk.Button(bar, text=text, style="Small.TButton", command=command)
            button.pack(side="left", padx=(0, 6))
            Tooltip(button, tip)
        ttk.Button(bar, text="Fit", style="Small.TButton", command=self._fit).pack(side="right")
        ttk.Button(bar, text="+", style="Small.TButton", width=3, command=lambda: self._zoom(ZOOM_STEP)).pack(
            side="right", padx=(0, 6))
        ttk.Button(bar, text="−", style="Small.TButton", width=3, command=lambda: self._zoom(1 / ZOOM_STEP)).pack(
            side="right", padx=(0, 4))
        ttk.Combobox(bar, textvariable=self._snap, values=SNAPS_MS, width=6, state="readonly",
                     font=theme.mono(9)).pack(side="right", padx=(0, 16))
        theme.caption(bar, "Grid (ms)").pack(side="right", padx=(0, 6))

        ttk.Label(timeline.body, text="Click an event to edit it, drag its block to move it (on the grid). "
                                      "Ctrl + wheel: zoom   Shift + wheel: scroll in time   Delete: remove",
                  style="Card.TLabel", foreground=theme.MUTED, font=theme.mono(8)).pack(side="bottom", anchor="w",
                                                                                        pady=(6, 0))
        holder = ttk.Frame(timeline.body, style="Card.TFrame")
        holder.pack(side="top", fill="both", expand=True)
        xscroll = ttk.Scrollbar(holder, orient="horizontal")
        xscroll.pack(side="bottom", fill="x")
        yscroll = ttk.Scrollbar(holder, orient="vertical")
        yscroll.pack(side="right", fill="y")
        self._canvas = tk.Canvas(holder, background=theme.FIELD, highlightthickness=0, height=260,
                                 xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)
        self._canvas.pack(side="left", fill="both", expand=True)
        xscroll.config(command=self._canvas.xview)
        yscroll.config(command=self._canvas.yview)
        self._canvas.bind("<ButtonPress-1>", self._press)
        self._canvas.bind("<B1-Motion>", self._motion)
        self._canvas.bind("<ButtonRelease-1>", self._release)
        self._canvas.bind("<MouseWheel>", self._wheel)
        self._canvas.bind("<Configure>", lambda _e: self._fit() if self._auto_fit else self._draw())
        self._canvas.bind("<Delete>", lambda _e: self._delete())
        self._snap.trace_add("write", lambda *_: self._draw())

    def _build_form(self, body: ttk.Frame) -> None:
        theme.caption(body, "Selected event").grid(row=0, column=0, columnspan=4, sticky="w")
        theme.caption(body, "Time (ms)").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self._time_entry = ttk.Entry(body, textvariable=self._time, width=10, font=theme.mono(10))
        self._time_entry.grid(row=2, column=0, sticky="w", padx=(0, 12))
        theme.caption(body, "Type").grid(row=1, column=1, sticky="w", pady=(8, 0))
        self._kind_box = ttk.Combobox(body, textvariable=self._kind, values=list(KINDS), state="readonly",
                                      width=16, font=theme.mono(10))
        self._kind_box.grid(row=2, column=1, sticky="w", padx=(0, 12))
        self._arg1_label = theme.caption(body, "")
        self._arg1_label.grid(row=1, column=2, sticky="w", pady=(8, 0))
        self._arg1_box = ttk.Combobox(body, textvariable=self._arg1, width=12, font=theme.mono(10))
        self._arg1_box.grid(row=2, column=2, sticky="w", padx=(0, 12))
        self._arg2_label = theme.caption(body, "")
        self._arg2_label.grid(row=1, column=3, sticky="w", pady=(8, 0))
        self._arg2_box = ttk.Combobox(body, textvariable=self._arg2, width=12, font=theme.mono(10))
        self._arg2_box.grid(row=2, column=3, sticky="w")
        theme.caption(body, "SCPI command").grid(row=3, column=0, columnspan=4, sticky="w", pady=(10, 0))
        self._command_entry = ttk.Entry(body, textvariable=self._command, font=theme.mono(10))
        self._command_entry.grid(row=4, column=0, columnspan=4, sticky="ew")
        body.columnconfigure(3, weight=1)
        self._event_note = ttk.Label(body, text="", style="Card.TLabel", foreground=theme.MUTED, font=theme.mono(8))
        self._event_note.grid(row=5, column=0, columnspan=4, sticky="w", pady=(6, 0))
        self._show_form()

    def restyle(self) -> None:
        self._canvas.config(background=theme.FIELD)
        self._draw()

    def _new(self, ask: bool = True) -> None:
        if ask and self._events and not messagebox.askyesno(
                "New experiment", "Start a new experiment? Changes not saved are lost.", parent=self):
            return
        self._path, self._metadata = None, {}
        self._output = {"format": "csv", "auto_save": True}
        self._filling = True
        self._experiment.set("New_Experiment")
        self._transients.set("1")
        self._rate.set("1000")
        self._acq_start.set("1000")
        self._samples.set("1000")
        for ch, var in self._channels.items():
            var.set(ch == 1)
        self._awg_file.set("")
        self._filling = False
        self._events = [_Event(0, ":AWG:RUN")]
        self._selected = 0
        self._file_badge.set("new file", theme.MUTED)
        self._after_change(fit=True)

    def load(self, path: str | Path) -> None:
        """Open an experiment file; it may contain errors (they are listed)."""
        path = Path(path)
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("the top level is not a JSON object")
        except (OSError, ValueError) as e:
            messagebox.showerror("Cannot open", f"{path.name}: {english_error(e)}", parent=self)
            return
        self._path = path
        self._metadata = dict(data.get("metadata") or {}) if isinstance(data.get("metadata"), dict) else {}
        output = data.get("output") if isinstance(data.get("output"), dict) else {}
        self._output = {k: v for k, v in output.items() if k != "average_count"} or {"format": "csv", "auto_save": True}
        hardware = data.get("hardware_parameters") if isinstance(data.get("hardware_parameters"), dict) else {}
        acquisition = data.get("acquisition") if isinstance(data.get("acquisition"), dict) else {}
        awg = data.get("awg_sequence") if isinstance(data.get("awg_sequence"), dict) else {}
        self._filling = True
        self._experiment.set(str(data.get("experiment_name", path.stem)))
        self._transients.set(str(output.get("average_count", 1)))
        self._rate.set(_number_text(hardware.get("acq_sample_rate_hz", "")))
        self._acq_start.set(str(acquisition.get("start", "")))
        self._samples.set(str(acquisition.get("number_of_samples", "")))
        channels = acquisition.get("channels") if isinstance(acquisition.get("channels"), list) else []
        for ch, var in self._channels.items():
            var.set(ch in channels)
        self._awg_file.set(str(awg.get("file_path", "")))
        self._filling = False
        events = data.get("events") if isinstance(data.get("events"), list) else []
        self._events = []
        for event in events:
            if isinstance(event, dict):
                try:
                    time_ms = int(event.get("time_ms", 0))
                except (TypeError, ValueError):
                    time_ms = 0
                self._events.append(_Event(max(time_ms, 0), str(event.get("action", ""))))
        self._selected = 0 if self._events else None
        self._file_badge.set(path.name, theme.ACCENT)
        self._after_change(fit=True)

    def _open(self) -> None:
        path = filedialog.askopenfilename(parent=self, title="Open experiment file",
                                          initialdir=self._path.parent if self._path else self._data_dir,
                                          filetypes=[("JSON", "*.json"), ("All files", "*.*")])
        if path:
            self.load(path)

    def _base_dir(self) -> Path:
        return self._path.parent if self._path else self._data_dir

    def _browse_awg(self) -> None:
        sequences = self._data_dir / "sequences"
        path = filedialog.askopenfilename(parent=self, title="AWG sequence of the experiment",
                                          initialdir=sequences if sequences.is_dir() else self._data_dir,
                                          filetypes=[("AWG sequence", "*.npz")])
        if path:
            self._awg_file.set(_relative(Path(path), self._base_dir()))

    def experiment_data(self) -> dict[str, Any]:
        """The experiment as the dictionary written to the JSON file."""
        data: dict[str, Any] = {"experiment_name": self._experiment.get().strip()}
        if self._metadata:
            data["metadata"] = self._metadata
        data["output"] = {"average_count": _int_or_text(self._transients.get()), **self._output}
        data["hardware_parameters"] = {"acq_sample_rate_hz": _number_or_text(self._rate.get())}
        data["acquisition"] = {
            "start": _int_or_text(self._acq_start.get()),
            "number_of_samples": _int_or_text(self._samples.get()),
            "channels": [ch for ch, var in self._channels.items() if var.get()],
        }
        data["events"] = [{"time_ms": e.time_ms, "action": e.action}
                          for e in sorted(self._events, key=lambda e: e.time_ms)]
        if self._awg_file.get().strip():
            data["awg_sequence"] = {"type": "npz", "file_path": self._awg_file.get().strip().replace("\\", "/")}
        return data

    def _check(self) -> list[str]:
        try:
            parse_experiment(self.experiment_data(), self._system().hardware_limits, base_dir=self._base_dir())
        except ConfigError as e:
            return list(e.errors)
        return []

    def _changed(self) -> None:
        if not self._filling:
            self._after_change()

    def _after_change(self, fit: bool = False) -> None:
        """Check the file and redraw (soon, once for many changes). fit: show the whole
        sequence (also when the timeline gets its size), until the user zooms."""
        if fit:
            self._auto_fit = True
            self.after_idle(self._fit)
        if self._pending is None:
            self._pending = self.after_idle(self._refresh)

    def _refresh(self) -> None:
        self._pending = None
        self._errors = self._check()
        self._problems.config(state="normal")
        self._problems.delete("1.0", "end")
        if self._errors:
            for error in self._errors:
                self._problems.insert("end", f"• {error}\n", "error")
            self._valid_badge.set(f"{len(self._errors)} problem(s)", theme.DANGER)
        else:
            self._problems.insert("end", "The experiment file is valid.", "ok")
            self._valid_badge.set("valid", theme.OK)
        self._problems.config(state="disabled")
        self._show_awg_info()
        self._draw()
        self._show_form()

    def _awg_duration_ms(self) -> float | None:
        """Length of the AWG sequence of the file in ms; None when there is none or it cannot be read."""
        text = self._awg_file.get().strip()
        if not text:
            return None
        path = (self._base_dir() / text).resolve()
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return None
        if self._awg_cache is not None and self._awg_cache[:2] == (path, mtime):
            return self._awg_cache[2]
        limits = self._system().hardware_limits
        try:
            duration = load_awg_npz(path, limits.max_awg_sample_rate_hz, limits.max_awg_amplitude_A).duration_s * 1000
        except ValueError:
            duration = None
        self._awg_cache = (path, mtime, duration)
        return duration

    def _show_awg_info(self) -> None:
        if not self._awg_file.get().strip():
            self._awg_info.config(text="no AWG sequence", foreground=theme.MUTED)
            return
        duration = self._awg_duration_ms()
        if duration is None:
            self._awg_info.config(text="file not found or not valid", foreground=theme.DANGER)
        else:
            self._awg_info.config(text=f"{duration:g} ms - :AWG:RUN lasts as long", foreground=theme.MUTED)

    def _duration_ms(self, action: str) -> float | None:
        if header_matches(action.strip().split(" ", 1)[0], ":AWG:RUN"):
            awg = self._awg_duration_ms()
            if awg is not None:
                return awg
        return self._system().event_duration_ms(action)

    def _acquisition_ms(self) -> tuple[float, float] | None:
        """Start and end of the acquisition in ms of the sequence, None if it cannot be computed."""
        try:
            rate = float(self._rate.get().replace(",", "."))
            start_us = float(self._acq_start.get())
            samples = int(self._samples.get())
        except ValueError:
            return None
        if rate <= 0:
            return None
        for event in sorted(self._events, key=lambda e: e.time_ms):
            header = event.action.strip().split(" ", 1)[0]
            if header_matches(header, ":AWG:RUN"):
                start = event.time_ms + start_us / 1000
                return start, start + samples / rate * 1000
        for event in sorted(self._events, key=lambda e: e.time_ms):
            if header_matches(event.action.strip().split(" ", 1)[0], ":ACQuire:RUN"):
                return event.time_ms, event.time_ms + samples / rate * 1000
        return None

    def _end_ms(self) -> float:
        ends = [e.time_ms + (self._duration_ms(e.action) or 0) for e in self._events]
        window = self._acquisition_ms()
        if window:
            ends.append(window[1])
        return max(ends, default=0.0)

    def _x(self, ms: float) -> float:
        return GUTTER_PX + ms * self._scale

    def _ms(self, x: float) -> float:
        return (x - GUTTER_PX) / self._scale

    def _fit(self) -> None:
        width = max(self._canvas.winfo_width() - GUTTER_PX - 40, 200)
        self._scale = width / max(self._end_ms() * 1.05, 10.0)
        self._draw()

    def _zoom(self, factor: float, at_x: float | None = None) -> None:
        canvas = self._canvas
        if at_x is None:
            at_x = canvas.winfo_width() / 2
        ms = self._ms(canvas.canvasx(at_x))
        self._auto_fit = False
        self._scale = min(max(self._scale * factor, 1e-5), 200.0)
        self._draw()
        total = float(canvas.cget("scrollregion").split()[2]) if canvas.cget("scrollregion") else 1
        canvas.xview_moveto(max(self._x(ms) - at_x, 0) / max(total, 1))

    def _grid_step_ms(self) -> float:
        """A round step of the time axis giving labels ~110 px apart."""
        target = 110 / self._scale
        candidates = [b * 10 ** p for p in range(-3, 9) for b in (1, 2, 5) if b * 10 ** p >= target]
        return min(candidates) if candidates else 1e8

    def _draw(self) -> None:
        canvas = self._canvas
        canvas.delete("all")
        rows = len(self._events) + 1
        end_ms = max(self._end_ms() * 1.1, 10.0)
        width = max(self._x(end_ms) + 40, canvas.winfo_width())
        height = max(AXIS_PX + rows * ROW_PX + 10, canvas.winfo_height())
        canvas.config(scrollregion=(0, 0, width, height))

        step = self._grid_step_ms()
        tick = 0.0
        while self._x(tick) < width:
            x = self._x(tick)
            canvas.create_line(x, AXIS_PX - 6, x, height, fill=theme.BORDER, dash=(2, 4))
            canvas.create_text(x + 3, AXIS_PX - 12, text=f"{tick:g} ms", anchor="w", fill=theme.MUTED,
                               font=theme.mono(8))
            tick += step
        canvas.create_line(GUTTER_PX, AXIS_PX - 6, width, AXIS_PX - 6, fill=theme.BORDER)
        canvas.create_text(10, AXIS_PX - 12, text="EVENT / SCPI COMMAND", anchor="w", fill=theme.MUTED,
                           font=theme.mono(8, "bold"))

        for i, event in enumerate(self._events):
            y = AXIS_PX + i * ROW_PX + ROW_PX / 2
            selected = i == self._selected
            if selected:
                canvas.create_rectangle(0, y - ROW_PX / 2, width, y + ROW_PX / 2, fill=theme.SELECTION, outline="")
            _name, colour = event_category(event.action)
            canvas.create_text(10, y, text=_short(event.action, 30), anchor="w", fill=colour, font=theme.mono(9))
            duration = self._duration_ms(event.action) or 0
            x0 = self._x(event.time_ms)
            x1 = max(self._x(event.time_ms + duration), x0 + MIN_BLOCK_PX)
            canvas.create_line(GUTTER_PX, y, x0, y, fill=theme.BORDER)
            canvas.create_rectangle(x0, y - 9, x1, y + 9, fill=_mix(colour, theme.FIELD, 0.35), outline=colour,
                                    width=2 if selected else 1, tags=("block", f"event{i}"))
            canvas.create_oval(x0 - 6, y - 6, x0 + 6, y + 6, fill=colour, outline=theme.TEXT if selected else colour,
                               width=2 if selected else 1, tags=("block", f"event{i}"))
            label = f"{event.time_ms:g} ms" + (f"  (+{duration:g} ms)" if duration else "")
            canvas.create_text(x1 + 8, y, text=label, anchor="w", fill=theme.MUTED, font=theme.mono(8),
                               tags=(f"event{i}",))

        window = self._acquisition_ms()
        y = AXIS_PX + len(self._events) * ROW_PX + ROW_PX / 2
        acquisition_colour = event_category(":ACQuire:RUN")[1]
        channels = ", ".join(str(ch) for ch, var in self._channels.items() if var.get()) or "–"
        canvas.create_text(10, y, text=f"Acquisition, CH {channels}", anchor="w", fill=acquisition_colour,
                           font=theme.mono(9, "bold"))
        if window:
            x0, x1 = self._x(window[0]), max(self._x(window[1]), self._x(window[0]) + MIN_BLOCK_PX)
            canvas.create_rectangle(x0, y - 9, x1, y + 9, fill=_mix(acquisition_colour, theme.FIELD, 0.25),
                                    outline=acquisition_colour)
            canvas.create_text(x0 + 6, y, text=f"{window[0]:g} – {window[1]:g} ms", anchor="w", fill=theme.TEXT,
                               font=theme.mono(8))
        canvas.create_line(GUTTER_PX, AXIS_PX - 6, GUTTER_PX, height, fill=theme.BORDER)

    def _event_at(self, x: float, y: float) -> int | None:
        for item in reversed(self._canvas.find_overlapping(x - 2, y - 2, x + 2, y + 2)):
            for tag in self._canvas.gettags(item):
                if tag.startswith("event"):
                    return int(tag[5:])
        row = int((y - AXIS_PX) // ROW_PX)
        if 0 <= row < len(self._events) and x < GUTTER_PX:
            return row
        return None

    def _press(self, event) -> None:
        self._canvas.focus_set()
        x, y = self._canvas.canvasx(event.x), self._canvas.canvasy(event.y)
        index = self._event_at(x, y)
        self._select(index)
        if index is not None and x >= GUTTER_PX:
            self._drag = (index, x, self._events[index].time_ms)
            self._canvas.config(cursor="sb_h_double_arrow")

    def _motion(self, event) -> None:
        if self._drag is None:
            return
        index, x0, time0 = self._drag
        snap = float(self._snap.get() or 1)
        new = time0 + (self._canvas.canvasx(event.x) - x0) / self._scale
        new = max(0, int(round(round(new / snap) * snap)))
        if new != self._events[index].time_ms:
            self._events[index].time_ms = new
            self._filling = True
            self._time.set(str(new))
            self._filling = False
            self._draw()

    def _release(self, _event) -> None:
        if self._drag is not None:
            self._drag = None
            self._canvas.config(cursor="")
            self._after_change()

    def _wheel(self, event) -> None:
        ctrl, shift = bool(event.state & 0x4), bool(event.state & 0x1)
        notches = event.delta / 120
        if ctrl:
            self._zoom(ZOOM_STEP if notches > 0 else 1 / ZOOM_STEP, at_x=event.x)
        elif shift:
            self._canvas.xview_scroll(int(-notches * 3), "units")
        else:
            self._canvas.yview_scroll(int(-notches * 3), "units")

    def _select(self, index: int | None) -> None:
        self._selected = index
        self._draw()
        self._show_form()

    def _show_form(self) -> None:
        """The fields of the selected event."""
        event = self._events[self._selected] if self._selected is not None and self._selected < len(
            self._events) else None
        self._filling = True
        state = "normal" if event else "disabled"
        for widget in (self._time_entry, self._command_entry):
            widget.config(state=state)
        self._kind_box.config(state="readonly" if event else "disabled")
        if event is None:
            for var in (self._time, self._kind, self._arg1, self._arg2, self._command):
                var.set("")
            self._event_note.config(text="Click an event on the timeline, or add one with + Add event.")
        else:
            if self.focus_get() is not self._time_entry:
                self._time.set(str(event.time_ms))
            kind, args = action_kind(event.action)
            self._kind.set(kind or "")
            self._set_arg_widgets(kind, args)
            if self.focus_get() is not self._command_entry:
                self._command.set(event.action)
            problems = [e.split(": ", 1)[-1] for e in self._errors if e.startswith(f"events[{self._sorted_index()}]")]
            self._event_note.config(text=problems[0] if problems else
                                    f"Duration on the timeline: {self._duration_ms(event.action) or 0:g} ms",
                                    foreground=theme.DANGER if problems else theme.MUTED)
        self._filling = False

    def _sorted_index(self) -> int | None:
        """Index of the selected event in the saved (time-sorted) order, as the check reports it."""
        if self._selected is None:
            return None
        order = sorted(range(len(self._events)), key=lambda i: self._events[i].time_ms)
        return order.index(self._selected)

    def _set_arg_widgets(self, kind: str | None, args: list[str]) -> None:
        params = KINDS[kind][1] if kind else ()
        for i, (label, box, var) in enumerate(((self._arg1_label, self._arg1_box, self._arg1),
                                               (self._arg2_label, self._arg2_box, self._arg2))):
            if i < len(params):
                name = params[i]
                values = {"channel": [str(c) for c in (Shim.CHANNELS if kind == "Shim current" else Digital.CHANNELS)],
                          "position": list(POSITIONS), "state": ["0", "1"], "mA": []}[name]
                label.config(text=name.upper())
                box.config(values=values, state="readonly" if values else "normal")
                if self.focus_get() is not box:
                    var.set(args[i] if i < len(args) else "")
                box.grid()
                label.grid()
            else:
                var.set("")
                box.grid_remove()
                label.grid_remove()

    def _current(self) -> _Event | None:
        if self._selected is None or self._selected >= len(self._events):
            return None
        return self._events[self._selected]

    def _form_changed(self) -> None:
        if self._filling:
            return
        event = self._current()
        if event is None:
            return
        text = self._time.get().strip()
        if text.isdigit():
            event.time_ms = int(text)
        kind = self._kind.get()
        if kind in KINDS:
            event.action = build_action(kind, [self._arg1.get(), self._arg2.get()])
            self._filling = True
            self._command.set(event.action)
            self._filling = False
        self._after_change()

    def _kind_changed(self) -> None:
        if self._filling:
            return
        event = self._current()
        kind = self._kind.get()
        if event is None or kind not in KINDS:
            return
        event.action = build_action(kind, list(DEFAULT_ARGS.get(kind, ())))
        self._show_form()
        self._after_change()

    def _command_changed(self) -> None:
        if self._filling:
            return
        event = self._current()
        if event is None:
            return
        event.action = self._command.get()
        kind, args = action_kind(event.action)
        self._filling = True
        self._kind.set(kind or "")
        self._set_arg_widgets(kind, args)
        self._filling = False
        self._after_change()

    def _add_event(self, kind: str) -> None:
        current = self._current()
        snap = int(float(self._snap.get() or 1))
        time_ms = current.time_ms + snap if current else 0
        self._events.append(_Event(time_ms, build_action(kind, list(DEFAULT_ARGS.get(kind, ())))))
        self._selected = len(self._events) - 1
        self._after_change()

    def _duplicate(self) -> None:
        current = self._current()
        if current is None:
            return
        snap = int(float(self._snap.get() or 1))
        self._events.insert(self._selected + 1, _Event(current.time_ms + snap, current.action))
        self._selected += 1
        self._after_change()

    def _delete(self) -> None:
        if self._current() is None:
            return
        del self._events[self._selected]
        self._selected = min(self._selected, len(self._events) - 1) if self._events else None
        self._after_change()

    def _sort(self) -> None:
        current = self._current()
        self._events.sort(key=lambda e: e.time_ms)
        self._selected = self._events.index(current) if current else None
        self._after_change()

    def _save(self) -> Path | None:
        """Write the file (only when it is valid) and read it back with the experiment checks."""
        if self._pending is not None:
            self.after_cancel(self._pending)
            self._refresh()
        if self._errors:
            messagebox.showerror("Cannot save", "Fix these problems first:\n\n" + "\n".join(self._errors[:15]),
                                 parent=self)
            return None
        initial = self._path.name if self._path else f"{self._experiment.get().strip() or 'experiment'}.json"
        path = filedialog.asksaveasfilename(parent=self, title="Save experiment file", defaultextension=".json",
                                            initialdir=self._base_dir(), initialfile=initial,
                                            filetypes=[("JSON", "*.json")])
        if not path:
            return None
        path = Path(path)
        awg_text = self._awg_file.get().strip()
        if awg_text:
            absolute = (self._base_dir() / awg_text).resolve()
            self._filling = True
            self._awg_file.set(_relative(absolute, path.parent))
            self._filling = False
        try:
            path.write_text(json.dumps(self.experiment_data(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            load_experiment_config(path, self._system().hardware_limits)
        except ConfigError as e:
            messagebox.showerror("Saved, but not valid", "\n".join(e.errors[:15]), parent=self)
            return None
        except OSError as e:
            messagebox.showerror("Cannot save", english_error(e), parent=self)
            return None
        self._path = path
        self._file_badge.set(path.name, theme.ACCENT)
        self._after_change()
        if self._on_saved:
            self._on_saved(path)
        return path

    def _add_to_queue(self) -> None:
        path = self._save()
        if path is not None and self._on_add_to_queue:
            self._on_add_to_queue(path)


def _relative(path: Path, base: Path) -> str:
    """path relative to base (with /), or absolute when it is on another drive."""
    try:
        return os.path.relpath(path, base).replace("\\", "/")
    except ValueError:
        return str(path)


def _int_or_text(text: str):
    text = text.strip()
    try:
        return int(text)
    except ValueError:
        return text


def _number_or_text(text: str):
    text = text.strip().replace(",", ".")
    try:
        value = float(text)
    except ValueError:
        return text
    return int(value) if value.is_integer() else value


def _number_text(value) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _short(text: str, length: int) -> str:
    return text if len(text) <= length else text[:length - 1] + "…"


def _mix(colour: str, background: str, share: float) -> str:
    def rgb(c: str) -> tuple[int, int, int]:
        return int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)
    a, b = rgb(colour), rgb(background)
    return "#" + "".join(f"{round(share * x + (1 - share) * y):02x}" for x, y in zip(a, b))
