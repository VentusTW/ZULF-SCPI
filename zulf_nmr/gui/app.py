"""Main window: connection indicator, SCPI console, experiment queue and its run."""

from __future__ import annotations

import logging
import queue
import threading
import tkinter as tk
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText
from typing import Any

import numpy as np
from matplotlib.figure import Figure
from serial.tools import list_ports

from zulf_nmr.core.connection import APP_DIR, DEFAULT_SYSTEM_CONFIG, Message, SCPIConnection
from zulf_nmr.core.scpi import header_matches
from zulf_nmr.core.spectrometer import Spectrometer
from zulf_nmr.experiment.parser import (
    ConfigError,
    ConnectionSettings,
    HardwareLimits,
    SystemConfig,
    load_system_config,
)
from zulf_nmr.experiment.history import HistoryError, load_results
from zulf_nmr.experiment.runner import ExperimentAborted
from zulf_nmr.experiment.sequence import acquisition_window_ms
from zulf_nmr.experiment.schedule import QueueItem, QueueRunner, ResultGroup, Transient, format_delay
from zulf_nmr.gui.awg_editor import AwgEditor
from zulf_nmr.gui.preview import PlotPreview
from zulf_nmr.gui.queue_editor import QueueEditor
from zulf_nmr.gui.results_viewer import ResultsViewer
from zulf_nmr.utils.visualization.awg_plotter import plot_awg
from zulf_nmr.utils.visualization.events_plotter import AcquisitionWindow, plot_events

log = logging.getLogger(__name__)

# Used when system.json is missing or invalid.
DEFAULT_SYSTEM = SystemConfig(
    connection=ConnectionSettings(port="COM3", baudrate=115200, timeout_s=2.0),
    hardware_limits=HardwareLimits(),
    data_output=Path("data/results/"),
)

_STATUS_POLL_MS = 500
_LOG_POLL_MS = 100
_MAX_LOG_LINE = 300  # longer console lines (e.g. echoed binary data) are cut
_COLOR_OK = "#2e9e44"
_COLOR_FAIL = "#d0342c"


def load_system_settings(path: Path = DEFAULT_SYSTEM_CONFIG) -> tuple[SystemConfig, str]:
    """Read system.json, falling back to DEFAULT_SYSTEM.

    Returns the settings and a message describing where they came from.
    """
    try:
        return load_system_config(path), f"Loaded connection settings from {path}"
    except ConfigError as e:
        reason = "; ".join(e.errors)
        return DEFAULT_SYSTEM, f"No valid {path} ({reason}) - using default values"


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("ZULF-SCPI")
        self.geometry("820x560")
        self.minsize(640, 400)

        # (tag, payload) from worker threads; payload is text, or a tuple for "progress" and "result"
        self._incoming: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._outgoing: queue.Queue[tuple[SCPIConnection, str]] = queue.Queue()
        threading.Thread(target=self._send_worker, name="SCPI sender", daemon=True).start()
        self._connection: SCPIConnection | None = None
        self._last_connected: bool | None = None

        # Queue of experiment sets; Run is possible once the user accepted every set.
        self._queue: list[QueueItem] = []
        self._queue_accepted = False
        self._runner: QueueRunner | None = None
        self._viewer: ResultsViewer | None = None  # live results of the current run
        self._transients_done = 0

        self._system, source = load_system_settings()
        self._settings = self._system.connection

        self._build_ui()
        self._log("info", source)
        self._apply_settings_to_fields(self._settings)
        try:
            self._ensure_results_dir()
        except OSError:
            pass  # already logged; checked again before every run
        self._connect()

        self.after(_STATUS_POLL_MS, self._poll_status)
        self.after(_LOG_POLL_MS, self._drain_log)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # -------------------------------------------------------------------- UI

    def _build_ui(self) -> None:
        top = ttk.Frame(self, padding=8)
        top.pack(fill="x")

        self._status_icon = tk.Canvas(top, width=18, height=18, highlightthickness=0)
        self._status_dot = self._status_icon.create_oval(2, 2, 16, 16, fill=_COLOR_FAIL, outline="")
        self._status_icon.pack(side="left")
        self._status_label = ttk.Label(top, text="Disconnected", width=14)
        self._status_label.pack(side="left", padx=(6, 12))

        ttk.Label(top, text="Port:").pack(side="left")
        self._port = ttk.Combobox(top, width=12, values=self._available_ports())
        self._port.pack(side="left", padx=(4, 8))
        ttk.Label(top, text="Baud:").pack(side="left")
        self._baud = ttk.Entry(top, width=8)
        self._baud.pack(side="left", padx=(4, 8))
        # Choosing a port (or Enter in either field) connects at once.
        self._port.bind("<<ComboboxSelected>>", lambda _e: self._connect())
        self._port.bind("<Return>", lambda _e: self._connect())
        self._baud.bind("<Return>", lambda _e: self._connect())

        self._connect_btn = ttk.Button(top, text="Connect", command=self._toggle_connection)
        self._connect_btn.pack(side="left")

        ttk.Button(top, text="Load config", command=self._load_config_file).pack(
            side="right"
        )
        ttk.Button(top, text="Load connection", command=self._load_connection_file).pack(
            side="right", padx=(0, 6)
        )
        ttk.Button(top, text="Load results", command=self._load_results).pack(side="right", padx=(0, 6))
        ttk.Button(top, text="AWG editor", command=self._open_awg_editor).pack(side="right", padx=(0, 6))

        run_bar = ttk.Frame(self, padding=(8, 0, 8, 8))
        run_bar.pack(fill="x")
        self._experiment_label = ttk.Label(run_bar, text="Queue: empty")
        self._experiment_label.pack(side="left")
        self._run_btn = ttk.Button(run_bar, text="Run", command=self._toggle_run, state="disabled")
        # Until the device is available the results are simulated.
        self._simulate = tk.BooleanVar(value=True)
        ttk.Checkbutton(run_bar, text="Simulation", variable=self._simulate,
                        command=self._update_run_button).pack(side="right", padx=(0, 8))
        self._run_btn.pack(side="right")
        self._progress_bar = ttk.Progressbar(run_bar, length=180, mode="determinate")
        self._progress_bar.pack(side="right", padx=(8, 8))
        self._progress_label = ttk.Label(run_bar, text="")
        self._progress_label.pack(side="right")

        self._console = ScrolledText(self, state="disabled", font=("Consolas", 10), wrap="word")
        self._console.pack(fill="both", expand=True, padx=8)
        self._console.tag_config("sent", foreground="#1f5fbf")
        self._console.tag_config("recv", foreground="#1b7a33")
        self._console.tag_config("error", foreground=_COLOR_FAIL)
        self._console.tag_config("info", foreground="#666666")

        bottom = ttk.Frame(self, padding=8)
        bottom.pack(fill="x")
        ttk.Label(bottom, text="Command:").pack(side="left")
        self._command = ttk.Entry(bottom, font=("Consolas", 10))
        self._command.pack(side="left", fill="x", expand=True, padx=6)
        self._command.bind("<Return>", lambda _e: self._send_command())
        self._command.bind("<Up>", lambda _e: self._history_step(-1))
        self._command.bind("<Down>", lambda _e: self._history_step(1))
        ttk.Button(bottom, text="Send", command=self._send_command).pack(side="left")
        self._command.focus_set()

        self._history: list[str] = []
        self._history_pos = 0

    @staticmethod
    def _available_ports() -> list[str]:
        return [p.device for p in list_ports.comports()] + ["loop://"]

    def _apply_settings_to_fields(self, settings: ConnectionSettings) -> None:
        self._port.set(settings.port)
        self._baud.delete(0, "end")
        self._baud.insert(0, str(settings.baudrate))

    def _set_status(self, connected: bool) -> None:
        self._status_icon.itemconfig(self._status_dot, fill=_COLOR_OK if connected else _COLOR_FAIL)
        self._status_label.config(text="Connected" if connected else "Disconnected")
        self._connect_btn.config(text="Disconnect" if connected else "Connect")
        self._update_run_button()

    # ------------------------------------------------------------ connection

    def _toggle_connection(self) -> None:
        if self._is_connected():
            self._disconnect()
        else:
            self._connect()

    def _is_connected(self) -> bool:
        return self._connection is not None and self._connection.is_connected

    def _connect(self) -> None:
        self._disconnect(quiet=True)
        port = self._port.get().strip()
        try:
            baudrate = int(self._baud.get())
        except ValueError:
            self._log("error", f"Invalid baud rate: {self._baud.get()!r}")
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
        except Exception as e:  # SerialException, ValueError for a bad URL...
            self._log("error", f"Cannot connect to {port}: {e}")
        else:
            self._log("info", f"Connected to {port} ({baudrate} baud)")
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
        # Called from the listener thread - hand over to the Tk thread via the queue.
        if isinstance(message, bytes):
            self._incoming.put(("recv", f"<< <binary block, {len(message)} B>"))
        else:
            self._incoming.put(("recv", f"<< {message}"))

    # -------------------------------------------------------------- commands

    def _send_command(self) -> None:
        command = self._command.get().strip()
        if not command:
            return
        self._history.append(command)
        self._history_pos = len(self._history)
        self._command.delete(0, "end")
        self._send(command)

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
                pass  # binary reply to a query - already shown by _on_message
            except TimeoutError:
                self._incoming.put(("error", f"No reply to {command!r}"))
            except Exception as e:
                self._incoming.put(("error", f"Error sending {command!r}: {e}"))

    def _history_step(self, step: int) -> str:
        if self._history:
            self._history_pos = max(0, min(len(self._history), self._history_pos + step))
            self._command.delete(0, "end")
            if self._history_pos < len(self._history):
                self._command.insert(0, self._history[self._history_pos])
        return "break"

    # --------------------------------------------------------- configuration

    def _load_connection_file(self) -> None:
        """Load connection settings from a system.json-style file and connect at once."""
        path = filedialog.askopenfilename(
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
        if system.hardware_limits != self._system.hardware_limits and self._queue_accepted:
            # The queue was validated against the old limits.
            self._set_queue(self._queue, accepted=False)
            self._log("info", "Hardware limits changed - load and accept the queue again")
        self._system = system
        self._settings = settings
        self._apply_settings_to_fields(settings)
        self._connect()

    def _load_config_file(self) -> None:
        """Edit the queue of experiment sets, then review and accept every set."""
        if self._runner is not None:
            self._log("error", "The queue is running - abort it before changing it")
            return
        editor = QueueEditor(self, self._queue, self._system.hardware_limits,
                             initial_dir=DEFAULT_SYSTEM_CONFIG.parent.parent / "data")
        items = editor.ask()
        if items is None:
            return
        self._set_queue(items, accepted=False)
        n = len(items)
        self._log("info", f"Queue with {n} set(s) loaded - review every set")
        for index, item in enumerate(items, start=1):
            if not self._review(item, index, n):
                self._log("info", f"Set {index}/{n} ({item.path.name}) rejected - the queue is not accepted")
                return
            self._log("info", f"Set {index}/{n} ({item.path.name}) accepted")
        self._set_queue(items, accepted=True)
        self._log("info", "All sets accepted - press Run to start the queue")

    def _review(self, item: QueueItem, index: int, total: int) -> bool:
        """Show the event timeline (and the AWG channels below it) of one set; True if accepted."""
        config, awg = item.config, item.awg
        # Bars: configured event durations; :AWG:RUN lasts as long as the AWG sequence.
        durations = [
            awg.duration_s * 1000 if awg is not None and header_matches(e.action.strip().split(" ", 1)[0], ":AWG:RUN")
            else self._system.event_duration_ms(e.action)
            for e in config.events
        ]
        window = acquisition_window_ms(config)
        channels = config.acquisition.channels
        timeline_acq = AcquisitionWindow(window[0], window[1], channels) if window else None
        rows = len(config.events) + (1 if timeline_acq else 0)
        events_height = max(4.0, 0.45 * rows + 2)

        def timeline(ax):
            plot_events(config.events, title=config.experiment_name, ax=ax,
                        durations_ms=durations, acquisition=timeline_acq)

        if awg is None:
            fig = Figure(figsize=(10, events_height), layout="constrained")
            timeline(fig.add_subplot())
        else:
            fig = Figure(figsize=(10, events_height + 6), layout="constrained")
            top, bottom = fig.subfigures(2, 1, height_ratios=[events_height, 6])
            timeline(top.add_subplot())
            acq = config.acquisition
            end_ms = (acq.start_us / 1000
                      + acq.duration_s(config.hardware_parameters.acq_sample_rate_hz) * 1000)
            plot_awg(awg, fig=bottom, acquisition=(acq.start_us / 1000, end_ms, channels))
        if index < total:
            hint = f"Set {index}/{total}. Next set starts {format_delay(item.delay_s)} after this one ends."
        else:
            hint = f"Set {index}/{total} (last)."
        hint += " Press OK to accept."
        return PlotPreview(self, f"Set {index}/{total}: {item.path.name}", fig, hint).ask()

    def _set_queue(self, items: list[QueueItem], accepted: bool) -> None:
        self._queue, self._queue_accepted = items, accepted
        if not items:
            text = "Queue: empty"
        else:
            names = ", ".join(item.path.stem for item in items[:3]) + (", …" if len(items) > 3 else "")
            text = f"Queue: {len(items)} set(s) [{names}] - {'accepted' if accepted else 'not accepted'}"
        self._experiment_label.config(text=text)
        self._update_run_button()

    # ------------------------------------------------------------------- run

    def _update_run_button(self) -> None:
        if self._runner is not None:
            self._run_btn.config(text="Abort", state="normal")
        else:
            ready = self._queue_accepted and (self._simulate.get() or self._is_connected())
            self._run_btn.config(text="Run", state="normal" if ready else "disabled")

    def _toggle_run(self) -> None:
        if self._runner is not None:
            self._runner.abort()
            self._log("info", "Aborting...")
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
                # The worker may already be on the next transient when the GUI draws this one,
                # so the average and its count are taken now, in the worker thread.
                on_result=lambda group, transient: self._incoming.put(
                    ("result", (group, transient, group.average, len(group.transients)))),
                on_live=lambda group, number, data: self._incoming.put(("live", (group, number, data))),
            )
        except (ValueError, OSError) as e:
            self._log("error", f"Cannot start the queue: {e}")
            return
        # One step per transient of every set; a running transient adds its share of points.
        self._transients_done = 0
        self._progress_bar.config(maximum=sum(i.config.output.average_count for i in self._queue), value=0)
        self._viewer = None
        mode = " - SIMULATION" if simulate else ""
        self._log("info", f"Starting the queue ({len(self._queue)} set(s)){mode}")
        self._update_run_button()
        threading.Thread(target=self._run_worker, args=(self._runner,), name="Queue", daemon=True).start()

    def _run_worker(self, runner: QueueRunner) -> None:
        try:
            runner.run()
        except ExperimentAborted:
            outcome = "Queue aborted" + ("" if runner.simulate else " - *RST sent")
            self._incoming.put(("error", outcome))
        except Exception as e:
            outcome = f"Queue failed: {e}"
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

    def _set_progress(self, text: str, value: float | None = None) -> None:
        """Progress bar and text in the main window and at the bottom of the results window."""
        if value is not None:
            self._progress_bar.config(value=value)
        self._progress_label.config(text=text)
        if self._viewer is not None and self._viewer.winfo_exists():
            self._viewer.set_progress(float(self._progress_bar["value"]),
                                      float(self._progress_bar["maximum"]), text)

    def _show_progress(self, set_index: int, transient: int, received: int, expected: int) -> None:
        """Samples of the running transient received so far."""
        total = self._queue[set_index - 1].config.output.average_count
        self._set_progress(f"Set {set_index}/{len(self._queue)}, transient {transient}/{total}: "
                           f"{received}/{expected} samples",
                           self._transients_done + (received / expected if expected else 0))

    def _results_window(self, group: ResultGroup) -> ResultsViewer | None:
        """The live results window: opened with the first data, not reopened if the user closed it."""
        if self._viewer is None:
            self._viewer = ResultsViewer(self, group.config_file.parent)
        return self._viewer if self._viewer.winfo_exists() else None

    def _show_live(self, group: ResultGroup, number: int, data: np.ndarray) -> None:
        viewer = self._results_window(group)
        if viewer is not None:
            total = self._queue[group.set_index - 1].config.acquisition.number_of_samples
            viewer.show_live(group, number, data, total)

    def _show_result(self, group: ResultGroup, transient: Transient, average: np.ndarray, count: int) -> None:
        """A transient was saved: advance the progress bar and add it to the live results window."""
        self._transients_done += 1
        viewer = self._results_window(group)
        self._set_progress(f"Set {group.set_index}/{len(self._queue)}: transient {transient.number}/"
                           f"{group.expected_transients} done", self._transients_done)
        if viewer is not None:
            viewer.add_transient(group, transient, average, count)

    # ------------------------------------------------------------ AWG editor

    def _open_awg_editor(self) -> None:
        limits = self._system.hardware_limits
        AwgEditor(self, APP_DIR / "data" / "sequences", limits.max_awg_sample_rate_hz, limits.max_awg_amplitude_A,
                  on_saved=lambda path, w: self._log(
                      "info", f"AWG sequence saved: {path} ({w.n_samples} samples, {w.sample_rate_hz:g} Hz)"))

    # --------------------------------------------------------------- results

    def _load_results(self) -> None:
        """Show the results of an earlier run: the user picks its whole folder."""
        try:
            initial = self._ensure_results_dir()
        except OSError:
            initial = APP_DIR
        folder = filedialog.askdirectory(title="Select the folder of a run (e.g. 20261019_105039)",
                                         initialdir=initial, mustexist=True)
        if not folder:
            return
        try:
            result, warnings = load_results(folder)
        except (HistoryError, OSError) as e:
            self._log("error", f"Cannot load results: {e}")
            messagebox.showerror("Cannot load results", str(e), parent=self)
            return
        for warning in warnings:
            self._log("error", f"   {warning}")
        sets = ", ".join(f"{g.name} ({len(g.transients)})" for g in result.groups)
        self._log("info", f"Loaded results from {result.folder}: {sets}")
        ResultsViewer(self, result.folder).show_saved(result)

    def _results_dir(self) -> Path:
        path = self._system.data_output
        # Relative paths are relative to APP_DIR, next to config/ and data/.
        return path if path.is_absolute() else APP_DIR / path

    def _ensure_results_dir(self) -> Path:
        """Create the results folder (and its parents, e.g. data/) if it does not exist."""
        path = self._results_dir()
        if not path.is_dir():
            try:
                path.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                self._log("error", f"Cannot create the results folder {path}: {e}")
                raise
            self._log("info", f"Created the results folder {path}")
        return path

    # ------------------------------------------------------------------- log

    def _log(self, tag: str, text: str) -> None:
        if len(text) > _MAX_LOG_LINE:
            text = text[:_MAX_LOG_LINE] + f"... ({len(text)} characters)"
        stamp = datetime.now().strftime("%H:%M:%S")
        self._console.config(state="normal")
        self._console.insert("end", f"[{stamp}] {text}\n", tag)
        self._console.see("end")
        self._console.config(state="disabled")

    def _drain_log(self) -> None:
        live = None  # only the newest live data is drawn - older parts are already outdated
        while True:
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
                    live = None  # the complete transient replaces it
                self._show_result(*text)
            elif tag == "status":
                self._set_progress(text)
                self._log("info", text)
            elif tag == "finished":
                self._runner = None
                self._set_progress(text)
                self._update_run_button()
            else:
                self._log(tag, text)
        if live is not None:
            self._show_live(*live)
        self.after(_LOG_POLL_MS, self._drain_log)

    def _on_close(self) -> None:
        if self._runner is not None:
            self._runner.abort()
        self._disconnect(quiet=True)
        self.destroy()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    App().mainloop()


if __name__ == "__main__":
    main()
