"""Main window: connection indicator, SCPI console, configuration loading and experiment run."""

from __future__ import annotations

import json
import logging
import queue
import threading
import tkinter as tk
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, ttk
from tkinter.scrolledtext import ScrolledText

from matplotlib.figure import Figure
from serial.tools import list_ports

from zulf_nmr.core.connection import DEFAULT_SYSTEM_CONFIG, Message, SCPIConnection
from zulf_nmr.core.spectrometer import Spectrometer
from zulf_nmr.experiment.parser import (
    ConfigError,
    ConnectionSettings,
    ExperimentConfig,
    HardwareLimits,
    SystemConfig,
    load_experiment_config,
    load_system_config,
)
from zulf_nmr.experiment.runner import ExperimentAborted, ExperimentRunner
from zulf_nmr.gui.preview import PlotPreview
from zulf_nmr.hardware.awg import AwgWaveform, load_awg_npz
from zulf_nmr.utils.visualization.awg_plotter import plot_awg
from zulf_nmr.utils.visualization.events_plotter import plot_events

log = logging.getLogger(__name__)

# Used when system.json is missing or invalid.
DEFAULT_SYSTEM = SystemConfig(
    connection=ConnectionSettings(port="COM3", baudrate=115200, timeout_s=2.0),
    hardware_limits=HardwareLimits(),
    data_output=Path("data/"),
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

        self._incoming: queue.Queue[tuple[str, str]] = queue.Queue()
        self._outgoing: queue.Queue[tuple[SCPIConnection, str]] = queue.Queue()
        threading.Thread(target=self._send_worker, name="SCPI sender", daemon=True).start()
        self._connection: SCPIConnection | None = None
        self._last_connected: bool | None = None

        # Experiment and AWG sequence accepted by the user, ready to run.
        self._experiment: ExperimentConfig | None = None
        self._awg: AwgWaveform | None = None
        self._runner: ExperimentRunner | None = None

        self._system, source = load_system_settings()
        self._settings = self._system.connection

        self._build_ui()
        self._log("info", source)
        self._apply_settings_to_fields(self._settings)
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

        self._connect_btn = ttk.Button(top, text="Connect", command=self._toggle_connection)
        self._connect_btn.pack(side="left")

        ttk.Button(top, text="Load config", command=self._load_config_file).pack(
            side="right"
        )

        run_bar = ttk.Frame(self, padding=(8, 0, 8, 8))
        run_bar.pack(fill="x")
        self._experiment_label = ttk.Label(run_bar, text="Experiment: none loaded")
        self._experiment_label.pack(side="left")
        self._run_btn = ttk.Button(run_bar, text="Run", command=self._toggle_run, state="disabled")
        self._run_btn.pack(side="right")
        self._progress_label = ttk.Label(run_bar, text="")
        self._progress_label.pack(side="right", padx=8)

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

    def _load_config_file(self) -> None:
        """Load a system file (connect at once) or an experiment file (preview, then accept)."""
        path = filedialog.askopenfilename(
            title="Select configuration file",
            initialdir=DEFAULT_SYSTEM_CONFIG.parent.parent,
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        path = Path(path)
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            self._log("error", f"Cannot read {path.name}: {e}")
            return

        if isinstance(data, dict) and "connection" in data:
            self._load_system_file(path)
        elif isinstance(data, dict) and "events" in data:
            self._load_experiment_file(path)
        else:
            self._log("error", f"{path.name}: unknown file - expected a 'connection' section "
                               "(system file) or 'events' (experiment file)")

    def _load_system_file(self, path: Path) -> None:
        try:
            system = load_system_config(path)
        except ConfigError as e:
            self._log_config_error(path, e)
            return
        settings = system.connection
        self._log("info", f"Loaded connection settings from {path.name}: {settings.port}, {settings.baudrate} baud")
        self._system = system
        self._settings = settings
        self._apply_settings_to_fields(settings)
        self._connect()

    def _load_experiment_file(self, path: Path) -> None:
        """Validate the experiment and its AWG sequence, then show both for acceptance."""
        if self._runner is not None:
            self._log("error", "An experiment is running - abort it before loading another one")
            return
        self._set_experiment(None, None)
        limits = self._system.hardware_limits
        try:
            config = load_experiment_config(path, limits=limits)
            awg = None
            if config.awg_sequence is not None:
                awg = load_awg_npz(
                    config.awg_sequence.file_path,
                    max_sample_rate_hz=limits.max_awg_sample_rate_hz,
                    max_amplitude_A=limits.max_awg_amplitude_A,
                )
        except ConfigError as e:
            self._log_config_error(path, e)
            return
        except ValueError as e:
            self._log("error", f"AWG sequence: {e}")
            return
        self._log("info", f"Loaded experiment {config.experiment_name!r} from {path.name} "
                          f"({len(config.events)} events)")

        # Event timeline on top, AWG channels below, so both are checked together.
        events_height = max(4.0, 0.45 * len(config.events) + 2)
        if awg is None:
            fig = Figure(figsize=(10, events_height), layout="constrained")
            plot_events(config.events, title=config.experiment_name, ax=fig.add_subplot())
            hint = "Check the event timeline. Press OK to enable Run."
        else:
            fig = Figure(figsize=(10, events_height + 6), layout="constrained")
            top, bottom = fig.subfigures(2, 1, height_ratios=[events_height, 6])
            plot_events(config.events, title=config.experiment_name, ax=top.add_subplot())
            plot_awg(awg, fig=bottom)
            hint = "Check the event timeline and the AWG waveforms. Press OK to enable Run."
        if not PlotPreview(self, "Experiment configuration", fig, hint).ask():
            self._log("info", "Experiment configuration rejected")
            return
        accepted = "Experiment configuration"
        accepted += f" and AWG sequence {awg.source_path.name}" if awg is not None else ""
        self._log("info", f"{accepted} accepted")

        self._set_experiment(config, awg)
        self._log("info", "Ready - press Run to start the experiment")

    def _log_config_error(self, path: Path, error: ConfigError) -> None:
        self._log("error", f"{path.name} contains errors:")
        for message in error.errors:
            self._log("error", f"   - {message}")

    def _set_experiment(self, config: ExperimentConfig | None, awg: AwgWaveform | None) -> None:
        self._experiment, self._awg = config, awg
        if config is None:
            text = "Experiment: none loaded"
        else:
            text = f"Experiment: {config.experiment_name} (accepted"
            text += f", AWG: {awg.source_path.name})" if awg is not None else ", no AWG sequence)"
        self._experiment_label.config(text=text)
        self._update_run_button()

    # ------------------------------------------------------------------- run

    def _update_run_button(self) -> None:
        if self._runner is not None:
            self._run_btn.config(text="Abort", state="normal")
        else:
            ready = self._experiment is not None and self._is_connected()
            self._run_btn.config(text="Run", state="normal" if ready else "disabled")

    def _toggle_run(self) -> None:
        if self._runner is not None:
            self._runner.abort()
            self._log("info", "Aborting...")
            return
        if self._experiment is None or not self._is_connected():
            return
        spec = Spectrometer(
            self._connection,
            hardware_limits=asdict(self._system.hardware_limits),
            data_output=self._system.data_output,
        )
        self._runner = ExperimentRunner(
            spec, self._experiment, awg_waveform=self._awg, on_progress=self._on_progress
        )
        self._log("info", f"Starting experiment {self._experiment.experiment_name!r}")
        self._update_run_button()
        threading.Thread(target=self._run_worker, args=(self._runner,), name="Experiment", daemon=True).start()

    def _run_worker(self, runner: ExperimentRunner) -> None:
        try:
            result = runner.run()
        except ExperimentAborted:
            self._incoming.put(("error", "Experiment aborted - *RST sent"))
        except Exception as e:
            self._incoming.put(("error", f"Experiment failed: {e}"))
        else:
            self._incoming.put(("info", f"Experiment finished ({len(result.transients)} transient(s))"))
            for path in result.files:
                self._incoming.put(("info", f"   saved {path}"))
        self._incoming.put(("finished", ""))

    def _on_progress(self, transient: int, received: int, expected: int) -> None:
        total = self._experiment.output.average_count if self._experiment else 1
        self._incoming.put(("progress", f"Transient {transient}/{total}: {received}/{expected} points"))

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
        while True:
            try:
                tag, text = self._incoming.get_nowait()
            except queue.Empty:
                break
            if tag == "progress":
                self._progress_label.config(text=text)
            elif tag == "finished":
                self._runner = None
                self._progress_label.config(text="")
                self._update_run_button()
            else:
                self._log(tag, text)
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
