"""Terminal & SCPI panel: the connection settings, the console of everything sent,
received and reported, and the command line with its history."""

from __future__ import annotations

import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Callable

from zulf_nmr.core.errors import english_error
from zulf_nmr.gui import theme
from zulf_nmr.gui.tooltip import Tooltip

MAX_LINES = 5000
MAX_LINE = 300
MAX_SAVED_HISTORY = 100


def tag_colours() -> dict[str, str]:
    """Colours of the console lines in the current theme."""
    return {"sent": theme.ACCENT, "recv": theme.OK, "error": theme.DANGER, "info": theme.MUTED}


class ConsolePanel(ttk.Frame):
    """The console and the connection fields; the App connects and sends.

    on_send(command) sends one command; on_connect() connects with the fields' values,
    on_toggle_connection() connects or disconnects, on_load_connection() reads a file.
    """

    def __init__(self, parent: tk.Misc, ports: Callable[[], list[str]], on_send: Callable[[str], None],
                 on_connect: Callable[[], None], on_toggle_connection: Callable[[], None],
                 on_load_connection: Callable[[], None]) -> None:
        super().__init__(parent, padding=(18, 16, 18, 16))
        self._ports = ports
        self._on_send = on_send
        self.history: list[str] = []
        self._history_pos = 0
        self._build(on_connect, on_toggle_connection, on_load_connection)

    def _build(self, on_connect, on_toggle_connection, on_load_connection) -> None:
        connection = theme.card(self)
        connection.pack(side="top", fill="x")
        body = connection.body
        title = ttk.Frame(body, style="Card.TFrame")
        title.pack(side="top", fill="x")
        theme.heading(title, "Connection").pack(side="left")
        self.state_badge = theme.Badge(title, "offline", theme.DANGER)
        self.state_badge.pack(side="left", padx=10)
        theme.caption(title, "Serial port of the instrument (or loop:// to test without one)").pack(side="right")

        row = ttk.Frame(body, style="Card.TFrame")
        row.pack(side="top", fill="x", pady=(12, 0))
        theme.caption(row, "Port").pack(side="left")
        self.port = ttk.Combobox(row, width=16, values=self._ports(), font=theme.mono(10),
                                 postcommand=lambda: self.port.config(values=self._ports()))
        self.port.pack(side="left", padx=(6, 16))
        theme.caption(row, "Baud").pack(side="left")
        self.baud = ttk.Entry(row, width=10, font=theme.mono(10))
        self.baud.pack(side="left", padx=(6, 16))
        self.port.bind("<<ComboboxSelected>>", lambda _e: on_connect())
        self.port.bind("<Return>", lambda _e: on_connect())
        self.baud.bind("<Return>", lambda _e: on_connect())
        self.connect_btn = ttk.Button(row, text="Connect", style="Accent.TButton", width=12,
                                      command=on_toggle_connection)
        self.connect_btn.pack(side="left")
        load = ttk.Button(row, text="Load connection…", command=on_load_connection)
        load.pack(side="left", padx=(8, 0))
        Tooltip(load, "Read the port and the hardware limits from a system.json-style file and connect")

        console = theme.card(self, padding=0)
        console.pack(side="top", fill="both", expand=True, pady=(14, 0))
        head = ttk.Frame(console.body, style="Card.TFrame", padding=(14, 10, 14, 8))
        head.pack(side="top", fill="x")
        theme.heading(head, "SCPI console").pack(side="left")
        legend = ttk.Frame(head, style="Card.TFrame")
        legend.pack(side="left", padx=16)
        for name, tag in (("sent", "sent"), ("received", "recv"), ("error", "error"), ("info", "info")):
            ttk.Label(legend, text="●", style="Card.TLabel", foreground=tag_colours()[tag],
                      font=theme.sans(9)).pack(side="left")
            ttk.Label(legend, text=name, style="Card.TLabel", foreground=theme.MUTED,
                      font=theme.mono(8)).pack(side="left", padx=(2, 10))
        ttk.Button(head, text="Save log…", style="Small.TButton", command=self._save_log).pack(side="right")
        ttk.Button(head, text="Clear", style="Small.TButton", command=self.clear).pack(side="right", padx=(0, 6))

        self._log_frame = tk.Frame(console.body, background=theme.FIELD, borderwidth=0, highlightthickness=0)
        self._log_frame.pack(side="top", fill="both", expand=True)
        scroll = ttk.Scrollbar(self._log_frame, orient="vertical", style="Field.Vertical.TScrollbar")
        scroll.pack(side="right", fill="y")
        self._text = tk.Text(self._log_frame, state="disabled", font=theme.mono(10), wrap="word",
                             background=theme.FIELD, foreground=theme.TEXT, borderwidth=0,
                             highlightthickness=0, padx=10, pady=8, insertbackground=theme.TEXT,
                             yscrollcommand=scroll.set)
        self._text.pack(side="left", fill="both", expand=True)
        scroll.config(command=self._text.yview)
        for tag, colour in tag_colours().items():
            self._text.tag_config(tag, foreground=colour)
        self._text.tag_config("stamp", foreground=theme.DIM)

        command = ttk.Frame(console.body, style="Card.TFrame", padding=(14, 10, 14, 12))
        command.pack(side="bottom", fill="x", before=self._log_frame)
        ttk.Label(command, text="›", style="Card.TLabel", foreground=theme.ACCENT,
                  font=theme.mono(14, "bold")).pack(side="left")
        self.command = ttk.Entry(command, font=theme.mono(10))
        self.command.pack(side="left", fill="x", expand=True, padx=8)
        self.command.bind("<Return>", lambda _e: self._send_typed())
        self.command.bind("<Up>", lambda _e: self._history_step(-1))
        self.command.bind("<Down>", lambda _e: self._history_step(1))
        ttk.Button(command, text="Send", style="Accent.TButton", command=self._send_typed).pack(side="left")

    def log(self, tag: str, text: str) -> None:
        if len(text) > MAX_LINE:
            text = text[:MAX_LINE] + f"... ({len(text)} characters)"
        stamp = datetime.now().strftime("%H:%M:%S")
        self._text.config(state="normal")
        self._text.insert("end", f"[{stamp}] ", "stamp")
        self._text.insert("end", f"{text}\n", tag)
        lines = int(self._text.index("end-1c").split(".")[0])
        if lines > MAX_LINES:
            self._text.delete("1.0", f"{lines - MAX_LINES + 1}.0")
        self._text.see("end")
        self._text.config(state="disabled")

    def clear(self) -> None:
        self._text.config(state="normal")
        self._text.delete("1.0", "end")
        self._text.config(state="disabled")

    def _save_log(self) -> None:
        path = filedialog.asksaveasfilename(
            parent=self, title="Save the console log", defaultextension=".txt",
            initialfile=f"zulf_scpi_log_{datetime.now():%Y%m%d_%H%M%S}.txt",
            filetypes=[("Text", "*.txt"), ("All files", "*.*")])
        if not path:
            return
        try:
            Path(path).write_text(self._text.get("1.0", "end-1c"), encoding="utf-8")
        except OSError as e:
            messagebox.showerror("Cannot save the log", english_error(e), parent=self)
            return
        self.log("info", f"Console log saved to {path}")

    def set_history(self, commands: list[str]) -> None:
        self.history = [c for c in commands if isinstance(c, str)][-MAX_SAVED_HISTORY:]
        self._history_pos = len(self.history)

    def _send_typed(self) -> None:
        command = self.command.get().strip()
        if not command:
            return
        self.history.append(command)
        self._history_pos = len(self.history)
        self.command.delete(0, "end")
        self._on_send(command)

    def _history_step(self, step: int) -> str:
        if self.history:
            self._history_pos = max(0, min(len(self.history), self._history_pos + step))
            self.command.delete(0, "end")
            if self._history_pos < len(self.history):
                self.command.insert(0, self.history[self._history_pos])
        return "break"

    def set_connected(self, connected: bool, text: str) -> None:
        self.state_badge.set(text, theme.OK if connected else theme.DANGER)
        self.connect_btn.config(text="Disconnect" if connected else "Connect",
                                style="TButton" if connected else "Accent.TButton")
