"""System panel: the settings the program works with - where they come from, the
connection, the hardware limits, the event durations and the folders."""

from __future__ import annotations

import os
import subprocess
import sys
import tkinter as tk
from dataclasses import fields
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Callable

from zulf_nmr.core.errors import english_error
from zulf_nmr.experiment.parser import SystemConfig
from zulf_nmr.gui import theme, user_settings
from zulf_nmr.version import version_text

_LIMIT_NAMES = {
    "max_shim_current_mA": ("Max shim current", "mA"),
    "max_awg_amplitude_A": ("Max AWG amplitude", "A"),
    "max_awg_sample_rate_hz": ("Max AWG sample rate", "Hz"),
    "motor_timeout_s": ("Motor timeout", "s"),
}


def open_folder(path: Path) -> None:
    """Show a folder in the system's file manager."""
    if sys.platform == "win32":
        os.startfile(path)
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


class SystemPanel(ttk.Frame):
    """Read-only view of the system settings, refreshed by show()."""

    def __init__(self, parent: tk.Misc, results_dir: Callable[[], Path],
                 on_load_connection: Callable[[], None]) -> None:
        super().__init__(parent, padding=(18, 16, 18, 16))
        self._results_dir = results_dir
        page = theme.ScrollPage(self)
        page.pack(fill="both", expand=True)
        self._page = page

        head = theme.card(page.inner, padding=12)
        head.pack(side="top", fill="x")
        theme.heading(head.body, "System").pack(side="left")
        theme.Badge(head.body, f"ZULF-SCPI {version_text()}", theme.ACCENT).pack(side="left", padx=10)
        ttk.Button(head.body, text="Load connection…", command=on_load_connection).pack(side="right")

        columns = ttk.Frame(page.inner)
        columns.pack(side="top", fill="x", pady=(12, 0))
        columns.columnconfigure((0, 1), weight=1, uniform="col")
        self._source = self._section(columns, "Settings file", 0, 0)
        self._connection = self._section(columns, "Connection", 0, 1)
        self._limits = self._section(columns, "Hardware limits (checked in every experiment file)", 1, 0)
        self._durations = self._section(columns, "Event durations used on the timeline", 1, 1)
        self._folders = self._section(columns, "Folders", 2, 0)

    def _section(self, parent: tk.Misc, title: str, row: int, column: int) -> ttk.Frame:
        card = theme.card(parent, padding=12)
        card.grid(row=row, column=column, sticky="nsew", padx=(0 if column == 0 else 6, 6 if column == 0 else 0),
                  pady=(0, 12))
        theme.caption(card.body, title).pack(side="top", anchor="w", pady=(0, 8))
        self._page.bind_wheel(card.body)
        body = ttk.Frame(card.body, style="Card.TFrame")
        body.pack(side="top", fill="x")
        return body

    def _pair(self, parent: ttk.Frame, key: str, value: str, colour: str | None = None) -> None:
        row = ttk.Frame(parent, style="Card.TFrame")
        row.pack(side="top", fill="x", pady=1)
        ttk.Label(row, text=key, style="Card.TLabel", foreground=theme.MUTED, font=theme.mono(9),
                  width=30).pack(side="left")
        ttk.Label(row, text=value, style="Card.TLabel", foreground=colour or theme.TEXT, font=theme.mono(9)).pack(side="left")

    @staticmethod
    def _clear(frame: ttk.Frame) -> None:
        for child in frame.winfo_children():
            child.destroy()

    def show(self, system: SystemConfig, source_note: str) -> None:
        for frame in (self._source, self._connection, self._limits, self._durations, self._folders):
            self._clear(frame)
        self._pair(self._source, "File", str(system.source_path) if system.source_path else "none - defaults",
                   theme.ACCENT if system.source_path else theme.WARNING)
        ttk.Label(self._source, text=source_note, style="Card.TLabel", foreground=theme.MUTED,
                  font=theme.sans(9), wraplength=520, justify="left").pack(side="top", anchor="w", pady=(6, 0))
        connection = system.connection
        self._pair(self._connection, "Port", connection.port, theme.ACCENT)
        self._pair(self._connection, "Baud rate", f"{connection.baudrate}")
        self._pair(self._connection, "Reply timeout", f"{connection.timeout_s:g} s")
        for f in fields(system.hardware_limits):
            name, unit = _LIMIT_NAMES.get(f.name, (f.name, ""))
            self._pair(self._limits, name, f"{getattr(system.hardware_limits, f.name):g} {unit}".strip())
        for command, ms in system.event_durations_ms.items():
            self._pair(self._durations, command, f"{ms:g} ms")
        results = self._results_dir()
        self._folder_row("Results", results)
        self._folder_row("Saved typed values", user_settings.SETTINGS_FILE.parent)

    def _folder_row(self, name: str, path: Path) -> None:
        row = ttk.Frame(self._folders, style="Card.TFrame")
        row.pack(side="top", fill="x", pady=2)
        ttk.Label(row, text=name, style="Card.TLabel", foreground=theme.MUTED, font=theme.mono(9),
                  width=20).pack(side="left")
        ttk.Button(row, text="Open", style="Small.TButton", command=lambda: self._open(path)).pack(side="right")
        ttk.Label(row, text=str(path), style="Card.TLabel", font=theme.mono(9)).pack(side="left")

    def _open(self, path: Path) -> None:
        try:
            open_folder(path)
        except OSError as e:
            messagebox.showerror("Cannot open the folder", f"{path}\n\n{english_error(e)}", parent=self)
