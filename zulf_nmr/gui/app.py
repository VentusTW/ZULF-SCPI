"""Main window: connection indicator, SCPI command console, configuration loading."""

from __future__ import annotations

import logging
import queue
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, ttk
from tkinter.scrolledtext import ScrolledText

from serial.tools import list_ports

from zulf_nmr.core.connection import DEFAULT_SYSTEM_CONFIG, Message, SCPIConnection
from zulf_nmr.experiment.parser import ConfigError, ConnectionSettings, load_system_config

log = logging.getLogger(__name__)

# Used when system.json is missing or invalid.
DEFAULT_CONNECTION = ConnectionSettings(port="COM3", baudrate=115200, timeout_s=2.0)

_STATUS_POLL_MS = 500
_LOG_POLL_MS = 100
_COLOR_OK = "#2e9e44"
_COLOR_FAIL = "#d0342c"


def load_connection_settings(path: Path = DEFAULT_SYSTEM_CONFIG) -> tuple[ConnectionSettings, str]:
    """Read connection settings from system.json, falling back to DEFAULT_CONNECTION.

    Returns the settings and a message describing where they came from.
    """
    try:
        return load_system_config(path).connection, f"Loaded connection settings from {path}"
    except ConfigError as e:
        reason = "; ".join(e.errors)
        return DEFAULT_CONNECTION, f"No valid {path} ({reason}) - using default values"


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

        settings, source = load_connection_settings()
        self._settings = settings

        self._build_ui()
        self._log("info", source)
        self._apply_settings_to_fields(settings)
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
        """Load connection settings from a system.json-style file and connect at once."""
        path = filedialog.askopenfilename(
            title="Select configuration file",
            initialdir=DEFAULT_SYSTEM_CONFIG.parent,
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        path = Path(path)
        try:
            settings = load_system_config(path).connection
        except ConfigError as e:
            self._log("error", f"{path.name} contains errors:")
            for message in e.errors:
                self._log("error", f"   - {message}")
            return
        self._log("info", f"Loaded connection settings from {path.name}: {settings.port}, {settings.baudrate} baud")
        self._settings = settings
        self._apply_settings_to_fields(settings)
        self._connect()

    # ------------------------------------------------------------------- log

    def _log(self, tag: str, text: str) -> None:
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
            self._log(tag, text)
        self.after(_LOG_POLL_MS, self._drain_log)

    def _on_close(self) -> None:
        self._disconnect(quiet=True)
        self.destroy()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    App().mainloop()


if __name__ == "__main__":
    main()
