"""Main window: a top bar with the connection and run state, a navigation between the
instrument consoles (terminal, queue, set review, results, AWG generator, system) and
the logic that ties them together - connection, queue review, run and results."""

from __future__ import annotations

import ctypes
import logging
import queue
import sys
import threading
import time
import tkinter as tk
from dataclasses import asdict
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageTk
from serial.tools import list_ports

from zulf_nmr.core.connection import APP_DIR, DEFAULT_SYSTEM_CONFIG, Message, SCPIConnection
from zulf_nmr.core.errors import english_error
from zulf_nmr.core.spectrometer import Spectrometer
from zulf_nmr.experiment.history import HistoryError, load_results
from zulf_nmr.experiment.parser import (
    ConfigError,
    ConnectionSettings,
    HardwareLimits,
    SystemConfig,
    load_system_config,
)
from zulf_nmr.experiment.runner import ExperimentAborted
from zulf_nmr.experiment.schedule import QueueItem, QueueResult, QueueRunner, ResultGroup, Transient
from zulf_nmr.gui import theme, user_settings
from zulf_nmr.gui.awg_editor import AwgEditorPanel
from zulf_nmr.gui.console_panel import MAX_SAVED_HISTORY, ConsolePanel
from zulf_nmr.gui.queue_editor import QueuePanel
from zulf_nmr.gui.results_viewer import ResultsPanel, ResultsView
from zulf_nmr.gui.review import ReviewPanel
from zulf_nmr.gui.sequence_editor import SequenceEditorPanel
from zulf_nmr.gui.system_panel import SystemPanel
from zulf_nmr.gui.tooltip import Tooltip
from zulf_nmr.version import version_text

log = logging.getLogger(__name__)

DEFAULT_SYSTEM = SystemConfig(
    connection=ConnectionSettings(port="COM3", baudrate=115200, timeout_s=2.0),
    hardware_limits=HardwareLimits(),
    data_output=Path("data/results/"),
)

ICON_DIR = Path(__file__).resolve().parent / "assets"

_STATUS_POLL_MS = 500
_LOG_POLL_MS = 100
_DRAIN_BUDGET_S = 0.05
_CLOSE_WAIT_S = 5.0
_SETTINGS = "main_window"
_LOGO_PX = 34
_SIDEBAR_PX = 250

PANELS = (
    ("console", "Terminal & SCPI", "›_"),
    ("queue", "Experiment queue", "≡"),
    ("sequence", "Sequence editor", "▥"),
    ("review", "Set review", "✓"),
    ("results", "Results", "▤"),
    ("awg", "AWG generator", "≈"),
    ("system", "System", "⚙"),
)


def load_system_settings(path: Path = DEFAULT_SYSTEM_CONFIG) -> tuple[SystemConfig, str]:
    """Read system.json, falling back to DEFAULT_SYSTEM.

    Returns the settings and a message describing where they came from.
    """
    try:
        return load_system_config(path), f"Loaded connection settings from {path}"
    except ConfigError as e:
        reason = "; ".join(e.errors)
        return DEFAULT_SYSTEM, f"No valid {path} ({reason}) - using default values"


class NavItem(tk.Frame):
    """One entry of the sidebar: an accent bar when active, an icon, a name and a badge."""

    def __init__(self, parent: tk.Misc, text: str, icon: str, shortcut: str, command: Callable[[], None]) -> None:
        super().__init__(parent, background=theme.SURFACE, cursor="hand2")
        self._bar = tk.Frame(self, width=3, background=theme.SURFACE)
        self._bar.pack(side="left", fill="y")
        self._icon = tk.Label(self, text=icon, width=3, font=theme.mono(11, "bold"), anchor="center")
        self._icon.pack(side="left", padx=(10, 4), pady=9)
        self._text = tk.Label(self, text=text, font=theme.sans(11), anchor="w")
        self._text.pack(side="left", fill="x", expand=True)
        self._badge = tk.Label(self, text="", font=theme.mono(8, "bold"), padx=6)
        self._badge.pack(side="right", padx=(0, 12))
        self._active = self._hover = False
        self._badge_colour = theme.MUTED
        for widget in (self, self._icon, self._text, self._badge):
            widget.bind("<Button-1>", lambda _e: command())
            widget.bind("<Enter>", lambda _e: self._set_hover(True))
            widget.bind("<Leave>", lambda _e: self._set_hover(False))
        Tooltip(self, f"{text}  ({shortcut})")
        self._paint()

    def set_active(self, active: bool) -> None:
        self._active = active
        self._paint()

    def set_badge(self, text: str, colour: str | None = None) -> None:
        self._badge_colour = colour or theme.MUTED
        self._badge.config(text=text.upper())
        self._paint()

    def _set_hover(self, hover: bool) -> None:
        self._hover = hover
        self._paint()

    def _paint(self) -> None:
        background = theme.PANEL if self._active else theme.RAISED if self._hover else theme.SURFACE
        for widget in (self, self._icon, self._text, self._badge):
            widget.config(background=background)
        self._bar.config(background=theme.ACCENT if self._active else background)
        self._icon.config(foreground=theme.ACCENT if self._active else theme.MUTED)
        self._text.config(foreground=theme.TEXT if self._active or self._hover else theme.NAV_TEXT)
        self._badge.config(foreground=self._badge_colour)

    def recolor(self, mapping: dict[str, str]) -> None:
        self._badge_colour = mapping.get(self._badge_colour.lower(), self._badge_colour)
        self._paint()


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        saved_theme = user_settings.load_section(_SETTINGS).get("theme")
        theme.apply_theme(self, saved_theme if saved_theme in theme.THEMES else "dark")
        self.title("ZULF-SCPI")
        self._set_icon()
        self.geometry("1280x820")
        self.minsize(1040, 660)
        self._maximize()

        self._incoming: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._outgoing: queue.Queue[tuple[SCPIConnection, str]] = queue.Queue()
        threading.Thread(target=self._send_worker, name="SCPI sender", daemon=True).start()
        self._connection: SCPIConnection | None = None
        self._last_connected: bool | None = None

        self._queue: list[QueueItem] = []
        self._queue_accepted = False
        self._review_items: list[QueueItem] | None = None
        self._review_index = 0
        self._runner: QueueRunner | None = None
        self._worker: threading.Thread | None = None
        self._viewer: ResultsView | None = None
        self._transients_done = 0
        self._progress_text = ""

        self._system, self._system_note = load_system_settings()
        self._settings = self._system.connection

        self._panels: dict[str, ttk.Frame | None] = {}
        self._current: str | None = None
        self._build_ui()
        theme.title_bar(self)
        theme.hide_title_icon(self)
        self._log("info", f"ZULF-SCPI {version_text()}")
        self._log("info", self._system_note)
        self._apply_settings_to_fields(self._settings)
        self._restore_typed()
        try:
            self._ensure_results_dir()
        except OSError:
            pass
        self._system_panel.show(self._system, self._system_note)
        self._set_queue([], accepted=False)
        self._connect()

        self.after(_STATUS_POLL_MS, self._poll_status)
        self.after(_LOG_POLL_MS, self._drain_log)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        self.columnconfigure(1, weight=1)
        self.rowconfigure(2, weight=1)

        self._build_top_bar().grid(row=0, column=0, columnspan=2, sticky="ew")
        tk.Frame(self, height=1, background=theme.BORDER).grid(row=1, column=0, columnspan=2, sticky="ew")
        self._sidebar = self._build_sidebar()
        self._sidebar.grid(row=2, column=0, sticky="ns")
        self.sidebar_shown = True
        self._content = ttk.Frame(self)
        self._content.grid(row=2, column=1, sticky="nsew")
        tk.Frame(self, height=1, background=theme.BORDER).grid(row=3, column=0, columnspan=2, sticky="ew")
        self._build_status_bar().grid(row=4, column=0, columnspan=2, sticky="ew")

        self._console = ConsolePanel(self._content, ports=self._available_ports, on_send=self._send,
                                     on_connect=self._connect, on_toggle_connection=self._toggle_connection,
                                     on_load_connection=self._load_connection_file)
        self._queue_panel = QueuePanel(self._content, self._system.hardware_limits,
                                       initial_dir=DEFAULT_SYSTEM_CONFIG.parent.parent / "data",
                                       on_review=self._start_review, on_changed=self._queue_edited)
        self._review = ReviewPanel(self._content, on_ok=self._review_ok, on_exit=self._review_exit,
                                   on_queue=lambda: self.show_panel("queue"))
        self._results = ResultsPanel(self._content, on_load=self._load_results)
        self._system_panel = SystemPanel(self._content, results_dir=self._results_dir,
                                         on_load_connection=self._load_connection_file)
        self._panels = {"console": self._console, "queue": self._queue_panel, "review": self._review,
                        "results": self._results, "awg": None, "sequence": None, "system": self._system_panel}
        for number, (name, _text, _icon) in enumerate(PANELS, start=1):
            self.bind_all(f"<Control-Key-{number}>", lambda _e, n=name: self.show_panel(n))
        self.bind_all("<Control-Key-b>", lambda _e: self._toggle_sidebar())
        self.show_panel("console")
        if user_settings.load_section(_SETTINGS).get("sidebar") is False:
            self._toggle_sidebar()
        self._console.command.focus_set()

    def _build_top_bar(self) -> ttk.Frame:
        bar = ttk.Frame(self, style="Surface.TFrame", padding=(10, 8, 14, 8))
        menu = ttk.Button(bar, text="☰", style="Small.TButton", width=3, command=self._toggle_sidebar)
        menu.pack(side="left", padx=(0, 12))
        Tooltip(menu, "Hide or show the menu on the left  (Ctrl+B)")
        try:
            image = Image.open(ICON_DIR / "icon.png").convert("RGBA").resize((_LOGO_PX, _LOGO_PX), Image.LANCZOS)
            self._logo = ImageTk.PhotoImage(image, master=self)
            ttk.Label(bar, image=self._logo, style="Surface.TLabel").pack(side="left")
        except OSError:
            log.warning("Logo not found in %s", ICON_DIR)
        name = ttk.Frame(bar, style="Surface.TFrame")
        name.pack(side="left", padx=(10, 18))
        ttk.Label(name, text="ZULF-SCPI", style="Surface.TLabel", foreground=theme.ACCENT,
                  font=theme.mono(12, "bold")).pack(anchor="w")
        ttk.Label(name, text=f"v{version_text()}", style="Surface.TLabel", foreground=theme.MUTED,
                  font=theme.mono(8)).pack(anchor="w")
        tk.Frame(bar, width=1, background=theme.BORDER).pack(side="left", fill="y", padx=(0, 16))

        chip = tk.Frame(bar, background=theme.FIELD, highlightthickness=1, highlightbackground=theme.BORDER)
        chip.pack(side="left")
        self._link_dot = theme.StatusDot(chip, theme.FIELD, size=9)
        self._link_dot.pack(side="left", padx=(10, 4), pady=6)
        self._link_text = tk.Label(chip, text="", background=theme.FIELD, foreground=theme.DANGER,
                                   font=theme.mono(9, "bold"))
        self._link_text.pack(side="left", padx=(0, 10))
        self._mode_badge = theme.Badge(bar, "", theme.WARNING)
        self._mode_badge.pack(side="left", padx=(12, 0), ipady=3)
        self._queue_badge = theme.Badge(bar, "", theme.MUTED)
        self._queue_badge.pack(side="left", padx=(8, 0), ipady=3)
        self._run_badge = theme.Badge(bar, "running", theme.ACCENT)

        self._abort_btn = ttk.Button(bar, text="■  ABORT", style="Danger.TButton", width=11, command=self._abort)
        self._abort_btn.pack(side="right")
        self._run_btn = ttk.Button(bar, text="▶  RUN QUEUE", style="Accent.TButton", width=15,
                                   command=self._start_run)
        self._run_btn.pack(side="right", padx=(0, 8))
        self._simulate = tk.BooleanVar(value=True)
        simulate = ttk.Checkbutton(bar, text="SIMULATION", variable=self._simulate, style="Surface.TCheckbutton",
                                   command=self._update_run_button)
        simulate.pack(side="right", padx=(0, 18))
        self._theme_btn = ttk.Button(bar, text="", style="Small.TButton", width=9, command=self._toggle_theme)
        self._theme_btn.pack(side="right", padx=(0, 14))
        self._show_theme_button()
        Tooltip(self._theme_btn, "Switch between the dark and the light colours")
        Tooltip(simulate, "Run the queue without the instrument: the data are simulated")
        Tooltip(self._abort_btn, "Abort the running queue (sends *RST to the instrument)")
        return bar

    def _build_sidebar(self) -> tk.Frame:
        side = tk.Frame(self, background=theme.SURFACE, width=_SIDEBAR_PX)
        side.pack_propagate(False)
        tk.Frame(side, width=1, background=theme.BORDER).pack(side="right", fill="y")
        theme.caption(side, "Instrument consoles", style="Surface.TLabel").pack(side="top", anchor="w",
                                                                              padx=16, pady=(16, 8))
        self._nav: dict[str, NavItem] = {}
        for number, (name, text, icon) in enumerate(PANELS, start=1):
            item = NavItem(side, text, icon, f"Ctrl+{number}", command=lambda n=name: self.show_panel(n))
            item.pack(side="top", fill="x")
            self._nav[name] = item

        box = tk.Frame(side, background=theme.SURFACE)
        box.pack(side="bottom", fill="x", padx=16, pady=(0, 16))
        tk.Frame(side, height=1, background=theme.BORDER).pack(side="bottom", fill="x", pady=(0, 10))
        self._state: dict[str, tk.Label] = {}
        for key in ("Link", "Mode", "Queue", "Run", "Transients"):
            row = tk.Frame(box, background=theme.SURFACE)
            row.pack(side="top", fill="x", pady=1)
            tk.Label(row, text=key.upper(), background=theme.SURFACE, foreground=theme.MUTED,
                     font=theme.mono(8)).pack(side="left")
            value = tk.Label(row, text="", background=theme.SURFACE, foreground=theme.TEXT, font=theme.mono(8))
            value.pack(side="right")
            self._state[key] = value
        return side

    def _build_status_bar(self) -> ttk.Frame:
        bar = ttk.Frame(self, style="Surface.TFrame", padding=(14, 5, 14, 5))
        self._progress_label = ttk.Label(bar, text="Ready", style="Surface.TLabel", foreground=theme.MUTED,
                                         font=theme.mono(9))
        self._progress_label.pack(side="left")
        ttk.Label(bar, text="Ctrl+1…7 switch consoles   Ctrl+B menu", style="Surface.TLabel", foreground=theme.DIM,
                  font=theme.mono(8)).pack(side="right")
        self._progress_bar = ttk.Progressbar(bar, length=260, mode="determinate")
        self._progress_bar.pack(side="right", padx=(0, 18))
        return bar

    def show_panel(self, name: str) -> None:
        """Show one console in the main area; the AWG generator is built the first time."""
        if name == self._current:
            return
        panel = self._panels.get(name)
        if panel is None and name == "awg":
            panel = self._panels["awg"] = self._build_awg_editor()
        if panel is None and name == "sequence":
            panel = self._panels["sequence"] = SequenceEditorPanel(
                self._content, lambda: self._system, DEFAULT_SYSTEM_CONFIG.parent.parent / "data",
                on_saved=lambda path: self._log("info", f"Experiment file saved: {path}"),
                on_add_to_queue=self._add_to_queue)
        if self._current is not None and self._panels.get(self._current) is not None:
            self._panels[self._current].pack_forget()
        panel.pack(fill="both", expand=True)
        self._current = name
        for key, item in self._nav.items():
            item.set_active(key == name)

    def _build_awg_editor(self) -> AwgEditorPanel:
        limits = self._system.hardware_limits
        return AwgEditorPanel(self._content, APP_DIR / "data" / "sequences", limits.max_awg_sample_rate_hz,
                              limits.max_awg_amplitude_A,
                              on_saved=lambda path, w: self._log(
                                  "info", f"AWG sequence saved: {path} ({w.n_samples} samples, "
                                          f"{w.sample_rate_hz:g} Hz)"))

    @staticmethod
    def _available_ports() -> list[str]:
        return [p.device for p in list_ports.comports()] + ["loop://"]

    def _apply_settings_to_fields(self, settings: ConnectionSettings) -> None:
        self._console.port.set(settings.port)
        self._console.baud.delete(0, "end")
        self._console.baud.insert(0, str(settings.baudrate))

    def _restore_typed(self) -> None:
        """Put back what the user typed last time: it wins over system.json."""
        saved = user_settings.load_section(_SETTINGS)
        port_field, baud_field = self._console.port, self._console.baud
        port = user_settings.text(saved, "port", port_field.get())
        baud = user_settings.text(saved, "baudrate", baud_field.get())
        if (port, baud) != (port_field.get(), baud_field.get()):
            port_field.set(port)
            baud_field.delete(0, "end")
            baud_field.insert(0, baud)
            self._log("info", f"Using the last port: {port}, {baud} baud")
        if isinstance(saved.get("simulation"), bool):
            self._simulate.set(saved["simulation"])
        history = saved.get("command_history")
        if isinstance(history, list):
            self._console.set_history(history)

    def _save_typed(self) -> None:
        user_settings.save_section(_SETTINGS, {
            "port": self._console.port.get().strip(),
            "baudrate": self._console.baud.get().strip(),
            "simulation": self._simulate.get(),
            "command_history": self._console.history[-MAX_SAVED_HISTORY:],
            "theme": theme.current(),
            "sidebar": self.sidebar_shown,
        })

    def _toggle_sidebar(self) -> None:
        """Hide the menu on the left (the consoles get the whole width) or show it again."""
        if self.sidebar_shown:
            self._sidebar.grid_remove()
        else:
            self._sidebar.grid()
        self.sidebar_shown = not self.sidebar_shown

    def _show_theme_button(self) -> None:
        self._theme_btn.config(text="☀  Light" if theme.current() == "dark" else "☾  Dark")

    def _toggle_theme(self) -> None:
        """Dark <-> light: repaint the window and draw the plots again."""
        theme.switch(self, "light" if theme.current() == "dark" else "dark")
        self._show_theme_button()
        self._review.restyle()
        for name in ("awg", "sequence"):
            if self._panels.get(name) is not None:
                self._panels[name].restyle()
        self._results.restyle()
        self._last_connected = None
        self._poll_status(reschedule=False)
        self._progress_label.config(foreground=theme.TEXT if self._runner else theme.MUTED)
        self._refresh_state()

    def _set_status(self, connected: bool) -> None:
        port, baud = self._settings.port, self._settings.baudrate
        text = f"CONNECTED  {port} ({baud})" if connected else f"DISCONNECTED  {port}"
        colour = theme.OK if connected else theme.DANGER
        self._link_dot.set(colour)
        self._link_text.config(text=text, foreground=colour)
        self._console.set_connected(connected, f"connected {port} ({baud})" if connected else "offline")
        self._state["Link"].config(text="CONNECTED" if connected else "OFFLINE", foreground=colour)
        self._update_run_button()

    def _toggle_connection(self) -> None:
        if self._is_connected():
            self._disconnect()
        else:
            self._connect()

    def _is_connected(self) -> bool:
        return self._connection is not None and self._connection.is_connected

    def _connect(self) -> None:
        self._disconnect(quiet=True)
        port = self._console.port.get().strip()
        try:
            baudrate = int(self._console.baud.get())
        except ValueError:
            self._log("error", f"Invalid baud rate: {self._console.baud.get()!r}")
            return
        self._settings = ConnectionSettings(port, baudrate, self._settings.timeout_s)
        self._connection = SCPIConnection(
            port=port,
            baudrate=baudrate,
            timeout_s=self._settings.timeout_s,
            on_message=self._on_message,
        )
        try:
            self._connection.connect()
        except Exception as e:
            self._log("error", f"Cannot connect to {port}: {english_error(e)}")
        else:
            self._log("info", f"Connected to {port} ({baudrate} baud)")
        self._last_connected = None
        self._poll_status(reschedule=False)

    def _disconnect(self, quiet: bool = False) -> None:
        if self._connection is not None:
            was_connected = self._connection.is_connected
            self._connection.close()
            self._connection = None
            if was_connected and not quiet:
                self._log("info", "Disconnected")
        self._poll_status(reschedule=False)

    def _poll_status(self, reschedule: bool = True) -> None:
        connected = self._is_connected()
        if self._last_connected and not connected and self._connection is not None:
            self._log("error", "Connection lost")
        if connected != self._last_connected:
            self._set_status(connected)
            self._last_connected = connected
        if reschedule:
            self.after(_STATUS_POLL_MS, self._poll_status)

    def _on_message(self, message: Message) -> None:
        if isinstance(message, bytes):
            self._incoming.put(("recv", f"<< <binary block, {len(message)} B>"))
        else:
            self._incoming.put(("recv", f"<< {message}"))

    def _send(self, command: str) -> None:
        if not self._is_connected():
            self._log("error", f"Not connected - not sent: {command}")
            return
        self._log("sent", f">> {command}")
        self._outgoing.put((self._connection, command))

    def _send_worker(self) -> None:
        """Send queued commands one by one, in order, off the Tk thread.

        Queries wait for the reply; the reply itself is shown by _on_message.
        """
        while True:
            conn, command = self._outgoing.get()
            try:
                if "?" in command.split(" ", 1)[0]:
                    conn.query(command)
                else:
                    conn.write(command)
            except TypeError:
                pass
            except TimeoutError:
                self._incoming.put(("error", f"No reply to {command!r}"))
            except Exception as e:
                self._incoming.put(("error", f"Error sending {command!r}: {english_error(e)}"))

    def _load_connection_file(self) -> None:
        """Load connection settings from a system.json-style file and connect at once."""
        path = filedialog.askopenfilename(
            parent=self,
            title="Select connection file",
            initialdir=DEFAULT_SYSTEM_CONFIG.parent,
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        path = Path(path)
        try:
            system = load_system_config(path)
        except ConfigError as e:
            self._log("error", f"{path.name} contains errors:")
            for message in e.errors:
                self._log("error", f"   - {message}")
            return
        settings = system.connection
        self._log("info", f"Loaded connection settings from {path.name}: {settings.port}, {settings.baudrate} baud")
        limits_changed = system.hardware_limits != self._system.hardware_limits
        self._system = system
        self._system_note = f"Loaded from {path}"
        self._queue_panel.set_limits(system.hardware_limits)
        if limits_changed and (self._queue_accepted or self._review_items is not None):
            self._invalidate_review("Hardware limits changed - review the queue again")
        self._settings = settings
        self._apply_settings_to_fields(settings)
        self._system_panel.show(system, self._system_note)
        self._connect()

    def _add_to_queue(self, path: Path) -> None:
        """An experiment file from the sequence editor goes to the end of the queue."""
        if self._runner is not None:
            self._log("error", "The queue is running - abort it before changing it")
            return
        self._queue_panel.add_files([path])
        self._log("info", f"{path.name} added to the queue - review the sets to accept it")
        self.show_panel("queue")

    def _start_review(self, items: list[QueueItem]) -> None:
        """Show every set of the queue in turn; OK on the last one accepts the queue."""
        if self._runner is not None:
            self._log("error", "The queue is running - abort it before changing it")
            return
        self._review_items, self._review_index = items, 0
        self._set_queue(items, accepted=False)
        self._log("info", f"Queue with {len(items)} set(s) loaded - review every set")
        self._review.show(items, 0, self._system, accepted=False)
        self.show_panel("review")

    def _review_ok(self) -> None:
        items, index = self._review_items, self._review_index
        if items is None:
            return
        n = len(items)
        self._log("info", f"Set {index + 1}/{n} ({items[index].path.name}) accepted")
        if index + 1 < n:
            self._review_index = index + 1
            self._refresh_state()
            self._review.show(items, index + 1, self._system, accepted=False)
            return
        self._review_items = None
        self._set_queue(items, accepted=True)
        self._log("info", "All sets accepted - press Run to start the queue")
        self._review.show(items, 0, self._system, accepted=True)

    def _review_exit(self) -> None:
        items, index = self._review_items, self._review_index
        if items is None:
            return
        n = len(items)
        self._log("info", f"Set {index + 1}/{n} ({items[index].path.name}) rejected - the queue is not accepted")
        self._review_items = None
        self._set_queue(items, accepted=False)
        self._review.show_empty(f"Set {index + 1}/{n} ({items[index].path.name}) was rejected - the queue is "
                                f"not accepted. Change it and press Review sets again.")
        self.show_panel("queue")

    def _queue_edited(self) -> None:
        """The sets in the queue panel changed: an accepted queue, or one in review, must be reviewed again."""
        if self._queue_accepted or self._review_items is not None:
            self._invalidate_review("Queue changed - press Review sets to review it again")
        self._set_queue(self._queue_panel.items, accepted=False)

    def _invalidate_review(self, message: str) -> None:
        self._review_items = None
        self._set_queue(self._queue, accepted=False)
        self._review.show_empty(message + ".")
        self._log("info", message)

    def _set_queue(self, items: list[QueueItem], accepted: bool) -> None:
        self._queue, self._queue_accepted = items, accepted
        self._update_run_button()

    def _update_run_button(self) -> None:
        running = self._runner is not None
        can_run = self._queue_accepted and (self._simulate.get() or self._is_connected())
        self._run_btn.config(state="normal" if can_run and not running else "disabled")
        self._abort_btn.config(state="normal" if running else "disabled")
        self._refresh_state()

    def _refresh_state(self) -> None:
        """The chips of the top bar, the state box and the badges of the sidebar."""
        running = self._runner is not None
        simulate = self._simulate.get()
        self._mode_badge.set("simulation" if simulate else "hardware", theme.WARNING if simulate else theme.INFO)
        self._state["Mode"].config(text="SIMULATION" if simulate else "HARDWARE",
                                   foreground=theme.WARNING if simulate else theme.INFO)
        n = len(self._queue)
        if self._review_items is not None:
            queue_text = f"queue {n} set(s) · review {self._review_index + 1}/{len(self._review_items)}"
            colour = theme.WARNING
        elif not n:
            queue_text, colour = "queue empty", theme.MUTED
        elif self._queue_accepted:
            queue_text, colour = f"queue {n} set(s) · accepted", theme.OK
        else:
            queue_text, colour = f"queue {n} set(s) · not accepted", theme.WARNING
        self._queue_badge.set(queue_text, colour)
        self._state["Queue"].config(text=queue_text.removeprefix("queue ").upper(), foreground=colour)
        if running and not self._run_badge.winfo_manager():
            self._run_badge.set("running", theme.ACCENT)
            self._run_badge.pack(side="left", padx=(8, 0), ipady=3, after=self._queue_badge)
        elif not running and self._run_badge.winfo_manager():
            self._run_badge.pack_forget()
        self._state["Run"].config(text="RUNNING" if running else "IDLE",
                                  foreground=theme.ACCENT if running else theme.MUTED)
        total = float(self._progress_bar["maximum"])
        self._state["Transients"].config(text=f"{self._transients_done}/{int(total)}" if running or
                                         self._transients_done else "–")

        panel_count = len(self._queue_panel.items)
        self._nav["queue"].set_badge(str(panel_count) if panel_count else "", theme.MUTED)
        if self._review_items is not None:
            self._nav["review"].set_badge(f"{self._review_index + 1}/{len(self._review_items)}", theme.WARNING)
        else:
            self._nav["review"].set_badge("ok" if self._queue_accepted else "", theme.OK)
        views = len(self._results.views)
        self._nav["results"].set_badge("live" if running else str(views) if views else "",
                                       theme.ACCENT if running else theme.MUTED)
        self._nav["console"].set_badge("" if self._is_connected() else "off", theme.DANGER)

    def _start_run(self) -> None:
        if self._runner is not None:
            return
        simulate = self._simulate.get()
        if not self._queue_accepted or not (simulate or self._is_connected()):
            return
        spec = None
        if not simulate:
            spec = Spectrometer(
                self._connection,
                hardware_limits=asdict(self._system.hardware_limits),
                data_output=self._results_dir(),
            )
        try:
            self._runner = QueueRunner(
                spec, self._queue, self._ensure_results_dir(), simulate=simulate,
                on_status=lambda text: self._incoming.put(("status", text)),
                on_progress=lambda *args: self._incoming.put(("progress", args)),
                on_result=lambda group, transient: self._incoming.put(
                    ("result", (group, transient, group.average, len(group.transients)))),
                on_live=lambda group, number, data: self._incoming.put(("live", (group, number, data))),
            )
        except (ValueError, OSError) as e:
            self._log("error", f"Cannot start the queue: {english_error(e)}")
            return
        self._transients_done = 0
        total = sum(i.config.output.average_count for i in self._queue)
        self._progress_bar.config(maximum=total, value=0)
        self._viewer = None
        mode = " - SIMULATION" if simulate else ""
        self._log("info", f"Starting the queue ({len(self._queue)} set(s)){mode}")
        self._queue_panel.set_locked(True)
        self._review.set_locked(True)
        self._update_run_button()
        self._worker = threading.Thread(target=self._run_worker, args=(self._runner,), name="Queue", daemon=True)
        self._worker.start()

    def _abort(self) -> None:
        if self._runner is not None:
            self._runner.abort()
            self._log("info", "Aborting...")

    def _run_worker(self, runner: QueueRunner) -> None:
        try:
            runner.run()
        except ExperimentAborted:
            outcome = "Queue aborted" + ("" if runner.simulate else " - *RST sent")
            self._incoming.put(("error", outcome))
        except Exception as e:
            outcome = f"Queue failed: {english_error(e)}"
            self._incoming.put(("error", outcome))
        else:
            outcome = f"Queue finished ({len(runner.items)} set(s))"
            self._incoming.put(("info", outcome))
        result = runner.result
        if result is not None:
            self._incoming.put(("info", f"Results in {result.folder}"))
            for path in result.files:
                self._incoming.put(("info", f"   {path.name}"))
        self._incoming.put(("finished", outcome))

    def _finished(self, outcome: str) -> None:
        self._runner = None
        self._set_progress(outcome)
        self._queue_panel.set_locked(False)
        self._review.set_locked(False)
        if self._viewer is not None and self._viewer.winfo_exists():
            self._results.rename(self._viewer, self._viewer.folder.name)
        self._update_run_button()

    def _set_progress(self, text: str, value: float | None = None) -> None:
        """Progress bar and text in the status bar; the text also in the live results tab."""
        if value is not None:
            self._progress_bar.config(value=value)
        self._progress_label.config(text=text, foreground=theme.TEXT if self._runner else theme.MUTED)
        if self._viewer is not None and self._viewer.winfo_exists():
            self._viewer.set_status(text)
        self._refresh_state()

    def _show_progress(self, set_index: int, transient: int, received: int, expected: int) -> None:
        """Samples of the running transient received so far."""
        total = self._queue[set_index - 1].config.output.average_count
        self._set_progress(f"Set {set_index}/{len(self._queue)}, transient {transient}/{total}: "
                           f"{received}/{expected} samples",
                           self._transients_done + (received / expected if expected else 0))

    def _results_window(self, group: ResultGroup) -> ResultsView | None:
        """The live results tab: opened (and shown) with the first data, not reopened if the user closed it."""
        if self._viewer is None:
            folder = group.config_file.parent
            self._viewer = self._results.new_view(folder, f"● LIVE  {folder.name}")
            self.show_panel("results")
        return self._viewer if self._viewer.winfo_exists() else None

    def _show_live(self, group: ResultGroup, number: int, data: np.ndarray) -> None:
        viewer = self._results_window(group)
        if viewer is not None:
            total = self._queue[group.set_index - 1].config.acquisition.number_of_samples
            viewer.show_live(group, number, data, total)

    def _show_result(self, group: ResultGroup, transient: Transient, average: np.ndarray, count: int) -> None:
        """A transient was saved: advance the progress bar and add it to the live results tab."""
        self._transients_done += 1
        viewer = self._results_window(group)
        self._set_progress(f"Set {group.set_index}/{len(self._queue)}: transient {transient.number}/"
                           f"{group.expected_transients} done", self._transients_done)
        if viewer is not None:
            viewer.add_transient(group, transient, average, count)

    def _load_results(self) -> None:
        """Show the results of an earlier run: the user picks its whole folder."""
        try:
            initial = self._ensure_results_dir()
        except OSError:
            initial = APP_DIR
        folder = filedialog.askdirectory(parent=self, title="Select the folder of a run (e.g. 20261019_105039)",
                                         initialdir=initial, mustexist=True)
        if not folder:
            return
        self.load_results_folder(folder)

    def load_results_folder(self, folder: str | Path) -> None:
        self._log("info", f"Loading results from {folder}...")
        threading.Thread(target=self._load_results_worker, args=(str(folder),), name="Load results",
                         daemon=True).start()

    def _load_results_worker(self, folder: str) -> None:
        """Read a results folder off the Tk thread: hundreds of files take seconds."""
        try:
            self._incoming.put(("history", load_results(folder)))
        except (HistoryError, OSError, ValueError) as e:
            self._incoming.put(("history_error", english_error(e)))

    def _show_history(self, result: QueueResult, warnings: list[str]) -> None:
        for warning in warnings:
            self._log("error", f"   {warning}")
        sets = ", ".join(f"{g.name} ({len(g.transients)})" for g in result.groups)
        self._log("info", f"Loaded results from {result.folder}: {sets}")
        view = self._results.new_view(result.folder, result.folder.name)
        self.show_panel("results")
        view.show_saved(result)
        self._refresh_state()

    def _results_dir(self) -> Path:
        path = self._system.data_output
        return path if path.is_absolute() else APP_DIR / path

    def _ensure_results_dir(self) -> Path:
        """Create the results folder (and its parents, e.g. data/) if it does not exist."""
        path = self._results_dir()
        if not path.is_dir():
            try:
                path.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                self._log("error", f"Cannot create the results folder {path}: {english_error(e)}")
                raise
            self._log("info", f"Created the results folder {path}")
        return path

    def _maximize(self) -> None:
        """Start on the whole screen; 1280x820 is the size the window gets when restored."""
        try:
            if sys.platform == "win32":
                self.state("zoomed")
            else:
                self.attributes("-zoomed", True)
        except tk.TclError:
            self.geometry(f"{self.winfo_screenwidth()}x{self.winfo_screenheight()}+0+0")

    def _set_icon(self) -> None:
        """DM Technologies logo instead of Tk's feather, for this and every later window."""
        try:
            if sys.platform == "win32":
                self.iconbitmap(default=str(ICON_DIR / "icon.ico"))
            self._icon = tk.PhotoImage(file=str(ICON_DIR / "icon.png"))
            self.iconphoto(True, self._icon)
        except (tk.TclError, OSError):
            log.warning("Window icon not found in %s", ICON_DIR)

    def _log(self, tag: str, text: str) -> None:
        self._console.log(tag, text)

    def _drain_log(self) -> None:
        """Handle what the worker threads sent. Always scheduled again - an error in one
        message must not stop the console, the progress and the plots for good - and
        never longer than _DRAIN_BUDGET_S at a time, so a flood of messages (e.g. a device
        sending without end) cannot freeze the window."""
        try:
            self._handle_incoming()
        except Exception:
            log.exception("Handling a message from a worker thread failed")
            try:
                self._log("error", "Internal error while showing data - see the log; the program goes on")
            except tk.TclError:
                pass
        finally:
            self.after(_LOG_POLL_MS, self._drain_log)

    def _handle_incoming(self) -> None:
        live = None
        deadline = time.monotonic() + _DRAIN_BUDGET_S
        try:
            while time.monotonic() < deadline:
                try:
                    tag, text = self._incoming.get_nowait()
                except queue.Empty:
                    break
                if tag == "live":
                    live = text
                elif tag == "progress":
                    self._show_progress(*text)
                elif tag == "result":
                    group, transient = text[0], text[1]
                    if live is not None and (live[0].set_index, live[1]) == (group.set_index, transient.number):
                        live = None
                    self._show_result(*text)
                elif tag == "status":
                    self._set_progress(text)
                    self._log("info", text)
                elif tag == "history":
                    self._show_history(*text)
                elif tag == "history_error":
                    self._log("error", f"Cannot load results: {text}")
                    messagebox.showerror("Cannot load results", text, parent=self)
                elif tag == "finished":
                    self._finished(text)
                else:
                    self._log(tag, text)
        finally:
            if live is not None:
                self._show_live(*live)

    def _on_close(self) -> None:
        if self._runner is not None:
            self._runner.abort()
            if self._worker is not None:
                self._worker.join(timeout=_CLOSE_WAIT_S)
        self._disconnect(quiet=True)
        self._save_typed()
        self.destroy()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    if sys.platform == "win32":
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("DMTechnologies.ZULF-SCPI")
        except (AttributeError, OSError):
            pass
    App().mainloop()


if __name__ == "__main__":
    main()
